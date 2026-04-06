/**
 * Miro Execution Trace & Knowledge Corpus — Client-Side Capture
 *
 * Injected into LibreChat's index.html BEFORE the first <script type="module">
 * tag.
 *
 * Strategy:
 *   Trace data CANNOT be embedded in the SSE stream between the Miro proxy
 *   and LibreChat's agents controller because:
 *     1. Fenced code blocks in content: broke message accumulation.
 *     2. SSE comments (': miro-trace …'): invisible to EventSource clients.
 *     3. Named SSE events ('event: miro-trace'): the OpenAI SDK's server-
 *        side parser doesn't support them and breaks content delivery.
 *     4. XHR/fetch interception: the browser never sees the proxy's SSE
 *        stream — the agents controller creates its own SSE stream.
 *
 *   Instead, the Miro proxy stores trace data server-side in memory and
 *   exposes it via REST endpoints:
 *     GET /miro-trace/latest  — returns the most recent trace data
 *     GET /miro-trace/{id}    — returns trace data for a specific request
 *
 *   This script polls /miro-trace/latest while a chat response is streaming
 *   and stores captured trace data in sessionStorage / localStorage for the
 *   trace and knowledge viewer pages.
 *
 * Also:
 *   - Overrides the Bookmark icon  → opens /trace.html in a new window.
 *   - Overrides the Plus-Circle icon → opens /knowledge.html in a new window.
 *   - Broadcasts updates via BroadcastChannel so the viewer pages update live.
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

  // ── Process trace data from the REST endpoint ─────────────────────────
  function processTraceResponse(data) {
    if (!data || !data.request_id) return;

    var trace = loadTrace();

    // Update events list
    if (data.events && data.events.length) {
      trace.events = data.events;
      saveTrace(trace);
      broadcast({ action: 'turn', data: data, conversationId: getConversationId() });
    }

    // Update summary if present
    if (data.summary) {
      trace.summary = data.summary;
      saveTrace(trace);
      broadcast({ action: 'summary', data: data.summary, conversationId: getConversationId() });

      // Add to knowledge corpus
      var summary = data.summary;
      var knowledge = loadKnowledge();
      // Avoid duplicate entries for the same request_id
      var isDuplicate = knowledge.some(function (k) { return k.id === data.request_id; });
      if (!isDuplicate) {
        knowledge.push({
          id: data.request_id,
          conversationId: getConversationId(),
          timestamp: new Date().toISOString(),
          turns: summary.turns_used,
          toolCalls: summary.total_tool_calls,
          elapsed: summary.elapsed_s,
          forced: !!summary.forced_synthesis,
          url: window.location.href,
          tools: (summary.events || []).reduce(function (acc, evt) {
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
  }

  // ── Poll the trace REST endpoint while streaming ──────────────────────
  //
  // We detect active streaming by watching for the stop button (which
  // LibreChat shows during generation).  While streaming, we poll
  // /miro-trace/latest every 2 seconds.  When streaming ends, we do
  // one final poll to capture the summary.

  var _pollTimer = null;
  var _lastSeenRequestId = null;
  var _isPolling = false;

  function pollTrace() {
    fetch('/miro-trace/latest', { credentials: 'same-origin' })
      .then(function (resp) { return resp.json(); })
      .then(function (data) {
        if (data && data.request_id) {
          _lastSeenRequestId = data.request_id;
          processTraceResponse(data);
        }
      })
      .catch(function (_e) {
        // Network error or proxy not available — ignore
      });
  }

  function startPolling() {
    if (_isPolling) return;
    _isPolling = true;
    pollTrace(); // immediate first poll
    _pollTimer = setInterval(pollTrace, 2000);
  }

  function stopPolling() {
    if (!_isPolling) return;
    _isPolling = false;
    if (_pollTimer) {
      clearInterval(_pollTimer);
      _pollTimer = null;
    }
    // Final poll to capture summary
    setTimeout(pollTrace, 500);
  }

  // Watch for the stop button to detect streaming state
  function checkStreamingState() {
    // LibreChat shows a stop button during generation
    var stopBtn = document.querySelector('[data-testid="stop-button"]');
    if (stopBtn) {
      startPolling();
    } else if (_isPolling) {
      stopPolling();
    }
  }

  // Check streaming state periodically
  setInterval(checkStreamingState, 1000);

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
