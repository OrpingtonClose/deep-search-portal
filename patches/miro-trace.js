/**
 * Miro Execution Trace & Knowledge Corpus — Client-Side Capture
 *
 * Injected into LibreChat's index.html BEFORE the first <script type="module">
 * tag so that our XHR patch is in place before any SSE client is created.
 *
 * Strategy:
 *   The Miro proxy emits trace data as *named* SSE events:
 *     event: miro-trace
 *     data: {"type":"turn","turn":1,...}
 *
 *   LibreChat uses sse.js (mpetazzoni/sse.js) which is an EventSource polyfill
 *   built on XMLHttpRequest.  sse.js has its OWN addEventListener/dispatchEvent
 *   (it does NOT extend EventTarget), and it DOES dispatch named SSE events to
 *   listeners registered for that event name.  However, LibreChat only registers
 *   a "message" handler, so "miro-trace" events are silently ignored.
 *
 *   We patch XMLHttpRequest.prototype to scan the raw responseText as it streams
 *   in and extract any "event: miro-trace\ndata: {...}" blocks.  This captures
 *   trace data at the XHR transport layer — before sse.js even parses the SSE
 *   stream — so it works regardless of which SSE client library is used.
 *
 * Also:
 *   3. Overrides the Bookmark icon  → opens /trace.html in a new window.
 *   4. Overrides the Plus-Circle icon → opens /knowledge.html in a new window.
 *   5. Broadcasts updates via BroadcastChannel so the viewer pages update live.
 */
(function () {
  'use strict';

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

  // ── Intercept XHR to capture trace data from SSE streams ────────────
  //
  // LibreChat uses sse.js (mpetazzoni/sse.js) — an EventSource polyfill
  // built on XMLHttpRequest.  sse.js has its OWN addEventListener (it does
  // NOT extend EventTarget), so we cannot patch EventTarget.prototype.
  //
  // The Miro proxy emits trace data as named SSE events:
  //   event: miro-trace
  //   data: {"type":"turn",...}
  //
  // We patch XMLHttpRequest.prototype.open to detect SSE streams to the
  // chat endpoint, then scan responseText as it streams in for
  // "event: miro-trace\ndata: {json}" blocks and extract the trace JSON.
  // This works at the XHR transport layer, before sse.js parses anything.

  var TRACE_EVENT_RE = /event:\s*miro-trace\ndata:\s*(.+)/g;
  var _origXhrOpen = XMLHttpRequest.prototype.open;

  XMLHttpRequest.prototype.open = function (method, url) {
    // Tag SSE chat streams so the progress handler knows to scan them.
    // The flag is checked lazily (at progress-event time, not at
    // addEventListener time) because sse.js registers its progress
    // listener BEFORE calling open().
    if (typeof url === 'string' && url.indexOf('/agents/chat/stream/') !== -1) {
      this._miroTraceStream = true;
      this._miroTraceProgress = 0;  // bytes already scanned
    }
    return _origXhrOpen.apply(this, arguments);
  };

  var _origXhrAddEL = XMLHttpRequest.prototype.addEventListener;

  XMLHttpRequest.prototype.addEventListener = function (type, listener, options) {
    if (type === 'progress') {
      // Wrap ALL progress listeners; the trace-scanning logic inside only
      // activates if _miroTraceStream was set by our patched open().
      // This is necessary because sse.js calls addEventListener('progress')
      // BEFORE calling xhr.open(), so the flag isn't set at registration time.
      var xhr = this;
      var wrappedListener = function (evt) {
        if (xhr._miroTraceStream) {
          // Scan the new portion of responseText for trace events
          try {
            var text = xhr.responseText;
            if (text && text.length > xhr._miroTraceProgress) {
              var newData = text.substring(xhr._miroTraceProgress);
              xhr._miroTraceProgress = text.length;

              // Find all "event: miro-trace\ndata: {json}" blocks
              var match;
              TRACE_EVENT_RE.lastIndex = 0;
              while ((match = TRACE_EVENT_RE.exec(newData)) !== null) {
                try {
                  var data = JSON.parse(match[1].trim());
                  processTraceData(data);
                } catch (_e) {
                  // Malformed JSON — skip
                }
              }
            }
          } catch (_e) {
            // responseText not available yet or other error — ignore
          }
        }
        // Always call the original progress listener
        return listener.call(this, evt);
      };
      return _origXhrAddEL.call(this, type, wrappedListener, options);
    }
    return _origXhrAddEL.call(this, type, listener, options);
  };

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
