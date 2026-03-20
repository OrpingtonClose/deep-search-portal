/**
 * Deep Search Portal — Open WebUI Sidebar Report Dropdown
 *
 * Injected via Open WebUI's static/loader.js hook.
 * Adds a collapsible "View Report" link beneath sidebar conversations
 * that have a matching research report on the portal.
 *
 * How matching works:
 *   - Fetches /research/reports (proxied via nginx to the persistent proxy)
 *   - Normalises both the chat title and report query to lowercase trimmed text
 *   - If the chat title starts with the first 60 chars of a report query (or vice
 *     versa), considers it a match.  Open WebUI often truncates long titles, so
 *     prefix matching is intentional.
 *
 * Lifecycle:
 *   - Runs once on DOMContentLoaded
 *   - Refreshes the report list every 60 seconds
 *   - Uses a MutationObserver to re-scan whenever sidebar DOM changes
 */

(function () {
  "use strict";

  // ---- configuration -------------------------------------------------------
  const REPORTS_API    = "/research/reports";
  const REPORT_PATH    = "/research/report/";   // + session_id
  const METRICS_PATH   = "/research/metrics/";   // + session_id
  const POLL_INTERVAL  = 60_000; // ms between report-list refreshes
  const MATCH_PREFIX   = 60;     // chars of prefix used for fuzzy title match
  const PROCESSED_ATTR  = "data-dsp-processed"; // marker to avoid re-processing
  const BADGE_CLASS    = "dsp-report-badge";
  const DROPDOWN_CLASS = "dsp-report-dropdown";

  // ---- state ---------------------------------------------------------------
  let reportsByNorm = new Map();   // normalisedQuery -> report object
  let observer      = null;

  // ---- helpers -------------------------------------------------------------

  /** Escape HTML special characters to prevent XSS. */
  function escHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  /** Normalise text for matching: lowercase, collapse whitespace, trim. */
  function norm(s) {
    return (s || "").toLowerCase().replace(/\s+/g, " ").trim();
  }

  /** Try to match a sidebar title to a report.  Returns report or null. */
  function matchReport(title) {
    const nt = norm(title);
    if (!nt || nt.length < 10) return null;

    // Exact match first
    if (reportsByNorm.has(nt)) return reportsByNorm.get(nt);

    // Prefix match (Open WebUI truncates long titles)
    // Only match if the prefix is long enough to be meaningful
    const prefix = nt.slice(0, MATCH_PREFIX);
    if (prefix.length < 10) return null;

    let bestMatch = null;
    let bestLen = 0;
    for (const [q, r] of reportsByNorm) {
      const qPrefix = q.slice(0, MATCH_PREFIX);
      if (q.startsWith(prefix) || prefix.startsWith(qPrefix)) {
        // Pick the longest overlap to avoid short-prefix false positives
        const overlap = Math.min(prefix.length, qPrefix.length);
        if (overlap > bestLen) {
          bestLen = overlap;
          bestMatch = r;
        }
      }
    }
    return bestMatch;
  }

  // ---- fetch reports -------------------------------------------------------

  async function refreshReports() {
    try {
      const resp = await fetch(REPORTS_API);
      if (!resp.ok) return;
      const data = await resp.json();
      const reports = data.reports || [];

      const m = new Map();
      for (const r of reports) {
        const key = norm(r.query);
        if (key) m.set(key, r);
      }
      reportsByNorm = m;
      scanSidebar();
    } catch (_) {
      // silently ignore — API may not be reachable yet
    }
  }

  // ---- sidebar scanning ----------------------------------------------------

  /**
   * Find all chat-item elements in the sidebar and inject report badges
   * for any that match a known report.
   *
   * Open WebUI renders chat items as <a> or <div> elements inside #sidebar.
   * Each item has a child that shows the truncated title text.  We look for
   * elements that hold the title string and inject a badge after them.
   */
  function scanSidebar() {
    if (reportsByNorm.size === 0) return;

    const sidebar = document.getElementById("sidebar");
    if (!sidebar) return;

    // Open WebUI renders chat items as clickable elements containing a span
    // with the chat title.  The structure varies by version, so we search
    // broadly for elements that look like chat titles.
    const candidates = sidebar.querySelectorAll(
      'a[href^="/c/"], div[data-chat-id], [class*="chat-item"]'
    );

    // Fallback: if no structured selectors match, walk all <a> in sidebar
    const items = candidates.length
      ? candidates
      : sidebar.querySelectorAll("a[href]");

    for (const el of items) {
      // Skip items we've already processed (use attribute on element itself)
      if (el.hasAttribute(PROCESSED_ATTR)) continue;

      // Extract title text — try the most specific child first
      const titleEl =
        el.querySelector(".line-clamp-1") ||
        el.querySelector('[class*="title"]') ||
        el.querySelector("span") ||
        el;
      const title = (titleEl.textContent || "").trim();
      if (!title || title.length < 5) continue;

      const report = matchReport(title);
      if (!report) continue;

      // Mark as processed BEFORE injecting to prevent MutationObserver loops
      el.setAttribute(PROCESSED_ATTR, "1");
      injectBadge(el, report);
    }
  }

  /** Insert a small "Report" badge/dropdown beneath a chat item element. */
  function injectBadge(chatEl, report) {
    const badge = document.createElement("div");
    badge.className = BADGE_CLASS;

    const toggle = document.createElement("button");
    toggle.className = "dsp-report-toggle";
    toggle.innerHTML =
      '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>' +
      ' <span>Report</span>';
    toggle.title = "View research report";

    const dropdown = document.createElement("div");
    dropdown.className = DROPDOWN_CLASS;
    dropdown.style.display = "none";

    const dur = report.total_duration_secs
      ? Math.round(report.total_duration_secs) + "s"
      : "?";
    const conds = report.total_conditions ?? "?";

    const safeSid  = escHtml(report.session_id);
    const safeConds = escHtml(String(conds));
    const safeDur   = escHtml(String(dur));

    dropdown.innerHTML =
      '<div class="dsp-report-meta">' +
        '<span>' + safeConds + " findings</span> &middot; <span>" + safeDur + "</span>" +
      "</div>" +
      '<a class="dsp-report-link" href="' + REPORT_PATH + safeSid +
        '" target="_blank" rel="noopener">Open Report</a>' +
      '<a class="dsp-report-link dsp-report-link--secondary" href="' +
        METRICS_PATH + safeSid +
        '" target="_blank" rel="noopener">Metrics JSON</a>';

    toggle.addEventListener("click", function (e) {
      e.preventDefault();
      e.stopPropagation();
      const open = dropdown.style.display !== "none";
      dropdown.style.display = open ? "none" : "block";
      toggle.classList.toggle("dsp-open", !open);
    });

    badge.appendChild(toggle);
    badge.appendChild(dropdown);

    // Insert badge inside the chat element (not as sibling) to keep it contained
    chatEl.appendChild(badge);
  }

  // ---- observer ------------------------------------------------------------

  function startObserver() {
    const target = document.getElementById("sidebar") || document.body;
    observer = new MutationObserver(function () {
      // Debounce: only scan once per animation frame
      if (!observer._raf) {
        observer._raf = requestAnimationFrame(function () {
          observer._raf = null;
          scanSidebar();
        });
      }
    });
    observer.observe(target, { childList: true, subtree: true });
  }

  // ---- init ----------------------------------------------------------------

  function init() {
    refreshReports();
    setInterval(refreshReports, POLL_INTERVAL);
    startObserver();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
