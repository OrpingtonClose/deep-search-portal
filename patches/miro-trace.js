/**
 * Miro Execution Trace & Knowledge Corpus — Client-Side Capture
 *
 * Injected into LibreChat's index.html alongside the YouTube embed script.
 *
 * 1. Hides `miro-trace` fenced code blocks via CSS (never visible to the user).
 * 2. Uses MutationObserver to capture JSON trace data from those code blocks.
 * 3. Stores trace data in sessionStorage (per-conversation) and knowledge
 *    corpus in localStorage (persistent across sessions).
 * 4. Overrides the Bookmark icon  → opens /trace.html in a new window.
 * 5. Overrides the Plus-Circle icon → opens /knowledge.html in a new window.
 * 6. Broadcasts updates via BroadcastChannel so the viewer pages update live.
 */
(function () {
  'use strict';

  // ── CSS: hide miro-trace code blocks before they ever paint ──────────
  var style = document.createElement('style');
  style.textContent = [
    'pre:has(> code.language-miro-trace) { display:none !important; height:0 !important; overflow:hidden !important; }',
    'code.language-miro-trace { display:none !important; }',
    // Fallback for browsers without :has()
    '.miro-trace-hidden { display:none !important; height:0 !important; overflow:hidden !important; }',
  ].join('\n');
  document.head.appendChild(style);

  // ── BroadcastChannel for live updates to viewer pages ────────────────
  var traceChannel = null;
  try {
    traceChannel = new BroadcastChannel('miro-trace');
  } catch (_e) {
    // BroadcastChannel not available — viewers will poll storage instead
  }

  function broadcast(msg) {
    if (traceChannel) {
      try { traceChannel.postMessage(msg); } catch (_e) { /* ignore */ }
    }
  }

  // ── Storage helpers ──────────────────────────────────────────────────
  function getConversationId() {
    // LibreChat URL pattern: /c/{conversationId} or /c/new
    var m = window.location.pathname.match(/\/c\/([^/]+)/);
    return m ? m[1] : 'unknown';
  }

  function getTraceKey() {
    return 'miro_trace_' + getConversationId();
  }

  function loadTrace() {
    try {
      var raw = sessionStorage.getItem(getTraceKey());
      return raw ? JSON.parse(raw) : { events: [], summary: null };
    } catch (_e) {
      return { events: [], summary: null };
    }
  }

  function saveTrace(trace) {
    try {
      sessionStorage.setItem(getTraceKey(), JSON.stringify(trace));
    } catch (_e) { /* storage full — ignore */ }
  }

  function loadKnowledge() {
    try {
      var raw = localStorage.getItem('miro_knowledge');
      return raw ? JSON.parse(raw) : [];
    } catch (_e) {
      return [];
    }
  }

  function saveKnowledge(entries) {
    try {
      // Keep last 500 entries to avoid localStorage bloat
      if (entries.length > 500) { entries = entries.slice(-500); }
      localStorage.setItem('miro_knowledge', JSON.stringify(entries));
    } catch (_e) { /* storage full — ignore */ }
  }

  // ── Process a captured trace data object ─────────────────────────────
  function processTraceData(data) {
    var trace = loadTrace();

    if (data.type === 'turn') {
      trace.events.push(data);
      saveTrace(trace);
      broadcast({ action: 'turn', data: data, conversationId: getConversationId() });
    } else if (data.type === 'summary') {
      trace.summary = data;
      // Also store the full events list from the summary
      if (data.events && data.events.length) {
        trace.events = data.events;
      }
      saveTrace(trace);
      broadcast({ action: 'summary', data: data, conversationId: getConversationId() });

      // Add to knowledge corpus — store the conversation as a research session
      var knowledge = loadKnowledge();
      knowledge.push({
        id: data.request_id || ('miro-' + Date.now()),
        conversationId: getConversationId(),
        timestamp: new Date().toISOString(),
        turns: data.turns_used,
        toolCalls: data.total_tool_calls,
        elapsed: data.elapsed_s,
        forced: !!data.forced_synthesis,
        url: window.location.href,
        // Extract tool queries as knowledge entries
        tools: (data.events || []).reduce(function (acc, evt) {
          if (evt.tools) {
            evt.tools.forEach(function (t) {
              if (t.args && (t.args.query || t.args.url)) {
                acc.push({
                  tool: t.tool,
                  query: t.args.query || t.args.url || '',
                  status: t.status,
                  chars: t.chars,
                  ms: t.ms,
                });
              }
            });
          }
          return acc;
        }, []),
      });
      saveKnowledge(knowledge);
      broadcast({ action: 'knowledge', conversationId: getConversationId() });
    }
  }

  // ── MutationObserver: capture miro-trace code blocks ─────────────────
  function scanForTraceBlocks(root) {
    if (!root || !root.querySelectorAll) return;

    var codeBlocks = root.querySelectorAll('code.language-miro-trace');
    for (var i = 0; i < codeBlocks.length; i++) {
      var code = codeBlocks[i];
      if (code.dataset.miroProcessed) continue;
      code.dataset.miroProcessed = '1';

      try {
        var jsonText = code.textContent.trim();
        var data = JSON.parse(jsonText);
        processTraceData(data);
      } catch (_e) {
        // Malformed JSON — skip
      }

      // Hide the parent <pre> (fallback for browsers without :has())
      var pre = code.closest('pre');
      if (pre) {
        pre.classList.add('miro-trace-hidden');
      }
    }
  }

  // Initial scan
  scanForTraceBlocks(document.body);

  // Watch for dynamically added content (SSE streaming)
  var traceObserver = new MutationObserver(function (mutations) {
    for (var i = 0; i < mutations.length; i++) {
      var added = mutations[i].addedNodes;
      for (var j = 0; j < added.length; j++) {
        if (added[j].nodeType === 1) {
          scanForTraceBlocks(added[j]);
        }
      }
      // Also check if the mutation target itself has trace blocks
      if (mutations[i].target && mutations[i].target.nodeType === 1) {
        scanForTraceBlocks(mutations[i].target);
      }
    }
  });

  traceObserver.observe(document.body, { childList: true, subtree: true });

  // ── Override header icon click handlers ───────────────────────────────
  //
  // Strategy: Use a single *capturing* click listener on <document> that
  // intercepts clicks on the bookmark and plus-circle buttons BEFORE
  // React's delegated handlers fire.  This never modifies the DOM, so
  // React's virtual-DOM reconciliation is not disrupted.

  document.addEventListener('click', function (e) {
    // Walk up from the click target to find the button with data-testid
    var node = e.target;
    while (node && node !== document.body) {
      var testid = node.getAttribute && node.getAttribute('data-testid');
      if (testid === 'bookmark-menu') {
        e.preventDefault();
        e.stopPropagation();
        e.stopImmediatePropagation();
        window.open('/trace.html', 'miro-trace', 'width=900,height=700');
        return;
      }
      if (testid === 'add-multi-convo-button') {
        e.preventDefault();
        e.stopPropagation();
        e.stopImmediatePropagation();
        window.open('/knowledge.html', 'miro-knowledge', 'width=1000,height=700');
        return;
      }
      node = node.parentNode;
    }
  }, true);  // true = capture phase, fires before React's bubble handlers

  // ── Tooltip override — update aria-label/title without touching DOM tree ─
  function updateTooltips() {
    var bk = document.querySelector('[data-testid="bookmark-menu"]');
    if (bk && bk.getAttribute('title') !== 'Execution Trace') {
      bk.setAttribute('title', 'Execution Trace');
      bk.setAttribute('aria-label', 'Execution Trace');
    }
    var pb = document.querySelector('[data-testid="add-multi-convo-button"]');
    if (pb && pb.getAttribute('title') !== 'Knowledge Corpus') {
      pb.setAttribute('title', 'Knowledge Corpus');
      pb.setAttribute('aria-label', 'Knowledge Corpus');
    }
  }

  updateTooltips();
  setInterval(updateTooltips, 2000);

})();
