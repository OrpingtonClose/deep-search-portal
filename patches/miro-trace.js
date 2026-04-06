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
  // Dual-trigger approach for robustness:
  //   1. Fetch interception: wraps window.fetch to detect chat stream
  //      requests — starts polling when a chat request is made, stops
  //      when the response body is fully consumed.
  //   2. Stop button fallback: periodically checks for the stop button
  //      in the DOM as a backup detection mechanism.

  // Save original fetch BEFORE wrapping — pollTrace uses this to avoid
  // triggering the chat-detection wrapper recursively.
  var _origFetch = window.fetch;

  var _pollTimer = null;
  var _lastSeenRequestId = null;
  var _isPolling = false;

  function pollTrace() {
    _origFetch('/miro-trace/latest', { credentials: 'same-origin' })
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
    console.log('[miro-trace] Polling started');
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
    console.log('[miro-trace] Polling stopped — final polls scheduled');
    // Final polls to capture summary (the proxy writes summary after
    // finishing all turns and closing the SSE stream)
    setTimeout(pollTrace, 500);
    setTimeout(pollTrace, 3000);
  }

  // ── Trigger 1: Intercept fetch() to detect chat stream requests ──────
  //
  // When LibreChat submits a message, it calls fetch() with a URL like
  // /api/agents/chat or /api/chat.  We wrap window.fetch to detect this,
  // start polling, and monitor the response to detect when streaming ends.
  //
  // IMPORTANT: We do NOT modify the request or response — we only observe.

  window.fetch = function () {
    var url = arguments[0];
    var urlStr = typeof url === 'string' ? url : (url && url.url ? url.url : '');

    if (urlStr.indexOf('/api/agents/chat') !== -1 ||
        urlStr.indexOf('/api/chat') !== -1) {
      console.log('[miro-trace] Chat request detected:', urlStr);
      startPolling();

      // Monitor the response to detect when streaming ends
      return _origFetch.apply(this, arguments).then(function (response) {
        if (response.body && typeof response.body.getReader === 'function') {
          // Tee the stream: one leg for LibreChat, one for monitoring
          var tee = response.body.tee();
          var monitoredResponse = new Response(tee[0], {
            status: response.status,
            statusText: response.statusText,
            headers: response.headers,
          });
          // Drain the monitor leg to detect stream end
          var reader = tee[1].getReader();
          (function pump() {
            reader.read().then(function (result) {
              if (result.done) {
                console.log('[miro-trace] Chat stream ended');
                stopPolling();
                return;
              }
              pump();
            }).catch(function () {
              stopPolling();
            });
          })();
          return monitoredResponse;
        }
        // Non-streaming response
        setTimeout(stopPolling, 5000);
        return response;
      }).catch(function (err) {
        stopPolling();
        throw err;
      });
    }

    return _origFetch.apply(this, arguments);
  };

  // ── Trigger 1b: Intercept XHR to detect chat stream requests ─────────
  //
  // LibreChat uses sse.js (mpetazzoni/sse.js), which is an EventSource
  // polyfill built on XMLHttpRequest.  It uses XHR POST (not fetch) to
  // connect to the SSE stream.  We wrap XHR.open() to detect these
  // requests and start polling.

  var _origXhrOpen = XMLHttpRequest.prototype.open;
  XMLHttpRequest.prototype.open = function () {
    var url = arguments[1] || '';
    if (typeof url === 'string' &&
        (url.indexOf('/api/agents/chat') !== -1 || url.indexOf('/api/chat') !== -1)) {
      console.log('[miro-trace] XHR chat request detected:', url);
      startPolling();
      // Monitor the XHR to detect when the stream ends
      var xhr = this;
      var _origOnReadyStateChange = null;
      Object.defineProperty(this, 'onreadystatechange', {
        get: function () { return _origOnReadyStateChange; },
        set: function (fn) {
          _origOnReadyStateChange = function () {
            if (xhr.readyState === 4) {
              console.log('[miro-trace] XHR stream ended (readyState=4)');
              stopPolling();
            }
            if (fn) fn.apply(this, arguments);
          };
        },
        configurable: true,
      });
      // Also listen for loadend as a backup
      this.addEventListener('loadend', function () {
        console.log('[miro-trace] XHR loadend');
        stopPolling();
      });
    }
    return _origXhrOpen.apply(this, arguments);
  };

  // ── Trigger 2: Stop button fallback ──────────────────────────────────
  function checkStreamingState() {
    var stopBtn = document.querySelector('[data-testid="stop-button"]') ||
                  document.querySelector('button[aria-label="Stop"]') ||
                  document.querySelector('button[aria-label="stop generating"]');
    if (stopBtn) {
      startPolling();
    } else if (_isPolling) {
      stopPolling();
    }
  }

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

  console.log('[miro-trace] Script loaded successfully');

})();
