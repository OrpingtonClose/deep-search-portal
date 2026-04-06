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

  // Re-check periodically because the header re-renders on navigation
  var lastBookmarkOverride = null;
  var lastPlusOverride = null;

  function overrideHeaderIcons() {
    // Bookmark icon — data-testid="bookmark-menu"
    var bookmarkBtn = document.querySelector('[data-testid="bookmark-menu"]');
    if (bookmarkBtn && bookmarkBtn !== lastBookmarkOverride) {
      lastBookmarkOverride = bookmarkBtn;
      // Remove existing click listeners by cloning
      var clone = bookmarkBtn.cloneNode(true);
      clone.addEventListener('click', function (e) {
        e.preventDefault();
        e.stopPropagation();
        e.stopImmediatePropagation();
        window.open('/trace.html', 'miro-trace', 'width=900,height=700');
      }, true);
      // Update tooltip
      clone.setAttribute('aria-label', 'Execution Trace');
      clone.setAttribute('title', 'Execution Trace');
      if (bookmarkBtn.parentNode) {
        bookmarkBtn.parentNode.replaceChild(clone, bookmarkBtn);
        lastBookmarkOverride = clone;
      }
    }

    // Plus-circle icon — data-testid="add-multi-convo-button"
    var plusBtn = document.querySelector('[data-testid="add-multi-convo-button"]');
    if (plusBtn && plusBtn !== lastPlusOverride) {
      lastPlusOverride = plusBtn;
      var clone2 = plusBtn.cloneNode(true);
      clone2.addEventListener('click', function (e) {
        e.preventDefault();
        e.stopPropagation();
        e.stopImmediatePropagation();
        window.open('/knowledge.html', 'miro-knowledge', 'width=1000,height=700');
      }, true);
      // Update tooltip
      clone2.setAttribute('aria-label', 'Knowledge Corpus');
      clone2.setAttribute('title', 'Knowledge Corpus');
      if (plusBtn.parentNode) {
        plusBtn.parentNode.replaceChild(clone2, plusBtn);
        lastPlusOverride = clone2;
      }
    }
  }

  // Run override immediately and on interval (header re-renders on nav)
  overrideHeaderIcons();
  setInterval(overrideHeaderIcons, 2000);

  // Also override on URL changes (SPA navigation)
  var lastHref = window.location.href;
  var navObserver = new MutationObserver(function () {
    if (window.location.href !== lastHref) {
      lastHref = window.location.href;
      // Reset per-conversation trace for new conversations
      lastBookmarkOverride = null;
      lastPlusOverride = null;
      setTimeout(overrideHeaderIcons, 500);
    }
  });
  navObserver.observe(document.body, { childList: true, subtree: true });

})();
