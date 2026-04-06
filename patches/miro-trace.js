/**
 * Miro Execution Trace & Knowledge Corpus — Client-Side Capture
 *
 * Injected into LibreChat's index.html BEFORE the first <script type="module">
 * tag so that our EventTarget patch is in place before any SSE client is created.
 *
 * Strategy:
 *   The Miro proxy emits trace data as *named* SSE events:
 *     event: miro-trace
 *     data: {"type":"turn","turn":1,...}
 *
 *   Named events only fire on listeners registered for that specific event
 *   name — LibreChat's "message" handler never sees them.  We patch
 *   EventTarget.prototype.addEventListener to detect when any object gets a
 *   "message" listener (the hallmark of an SSE client) and piggyback our own
 *   "miro-trace" listener onto the same target.
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

  // ── Intercept EventTarget.addEventListener to capture SSE trace events ─
  //
  // LibreChat uses a custom EventSource-like class (ResumableSSE) that
  // registers "message" listeners on an EventTarget.  The Miro proxy emits
  // trace data as *named* SSE events ("event: miro-trace\ndata: {json}\n\n")
  // which only fire on listeners registered for that specific event name.
  // LibreChat's "message" handler never sees them.
  //
  // We patch addEventListener so that whenever *any* object gets a "message"
  // listener, we also attach our own "miro-trace" listener on the same
  // target.  A WeakSet tracks targets we've already attached to.

  var _origAddEventListener = EventTarget.prototype.addEventListener;
  var _trackedTargets = typeof WeakSet !== 'undefined' ? new WeakSet() : null;
  // Fallback for environments without WeakSet (unlikely but safe)
  var _trackedArray = _trackedTargets ? null : [];

  function isTracked(target) {
    if (_trackedTargets) return _trackedTargets.has(target);
    return _trackedArray.indexOf(target) !== -1;
  }

  function markTracked(target) {
    if (_trackedTargets) { _trackedTargets.add(target); }
    else { _trackedArray.push(target); }
  }

  EventTarget.prototype.addEventListener = function (type, listener, options) {
    // Call the original first
    var result = _origAddEventListener.call(this, type, listener, options);

    // If someone registers a "message" listener, piggyback a "miro-trace" listener
    if (type === 'message' && !isTracked(this)) {
      markTracked(this);
      _origAddEventListener.call(this, 'miro-trace', function (evt) {
        try {
          var data = JSON.parse(evt.data);
          processTraceData(data);
        } catch (_e) {
          // Malformed JSON or non-trace event — ignore
        }
      });
    }

    return result;
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
