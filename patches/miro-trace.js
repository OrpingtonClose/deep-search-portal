/**
 * Miro Execution Trace & Knowledge Corpus — Client-Side Capture
 *
 * Injected into LibreChat's index.html alongside the YouTube embed script.
 *
 * 1. Monkey-patches fetch() to intercept SSE comments containing trace data
 *    (": miro-trace {json}") from the Miro proxy's SSE stream.
 * 2. Stores trace data in sessionStorage (per-conversation) and knowledge
 *    corpus in localStorage (persistent across sessions).
 * 3. Overrides the Bookmark icon  → opens /trace.html in a new window.
 * 4. Overrides the Plus-Circle icon → opens /knowledge.html in a new window.
 * 5. Broadcasts updates via BroadcastChannel so the viewer pages update live.
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

  // ── Monkey-patch fetch() to intercept SSE trace comments ─────────────
  //
  // The Miro proxy emits trace data as SSE comments:
  //   : miro-trace {"type":"turn","turn":1,...}
  //
  // We intercept the Response body stream, strip out the trace comments,
  // process the JSON payloads, and forward the remaining (clean) SSE data
  // to LibreChat's reader so message accumulation works normally.

  var TRACE_PREFIX = ': miro-trace ';
  var _origFetch = window.fetch;

  window.fetch = function () {
    var fetchArgs = Array.prototype.slice.call(arguments);
    return _origFetch.apply(this, fetchArgs).then(function (response) {
      // Only intercept SSE streams to the chat endpoint
      var url = typeof fetchArgs[0] === 'string'
        ? fetchArgs[0]
        : (fetchArgs[0] && fetchArgs[0].url ? fetchArgs[0].url : '');
      var ct = '';
      try { ct = response.headers.get('content-type') || ''; } catch (_e) { /* ignore */ }

      if (!url.includes('/chat/') || !ct.includes('text/event-stream')) {
        return response;
      }

      var body = response.body;
      if (!body) return response;

      var reader = body.getReader();
      var decoder = new TextDecoder();
      var partialLine = '';

      var newStream = new ReadableStream({
        start: function (controller) {
          function pump() {
            reader.read().then(function (result) {
              if (result.done) {
                // Flush any remaining partial data
                if (partialLine.length > 0) {
                  controller.enqueue(new TextEncoder().encode(partialLine));
                }
                controller.close();
                return;
              }

              var text = decoder.decode(result.value, { stream: true });
              // Prepend any leftover from previous chunk
              text = partialLine + text;
              partialLine = '';

              // Split into lines, keeping the delimiters
              var lines = text.split('\n');

              // The last element may be an incomplete line
              partialLine = lines.pop() || '';

              var cleanChunks = [];
              for (var i = 0; i < lines.length; i++) {
                var line = lines[i];
                if (line.indexOf(TRACE_PREFIX) === 0) {
                  // This is a trace comment — extract JSON and process it
                  var jsonStr = line.substring(TRACE_PREFIX.length).trim();
                  try {
                    var data = JSON.parse(jsonStr);
                    processTraceData(data);
                  } catch (_e) {
                    // Malformed JSON — skip
                  }
                  // Don't forward this line to LibreChat
                } else {
                  // Normal SSE line — forward it
                  cleanChunks.push(line + '\n');
                }
              }

              if (cleanChunks.length > 0) {
                controller.enqueue(new TextEncoder().encode(cleanChunks.join('')));
              }

              pump();
            }).catch(function (err) {
              controller.error(err);
            });
          }
          pump();
        },
        cancel: function () {
          reader.cancel();
        }
      });

      // Create a new Response with the filtered stream
      return new Response(newStream, {
        status: response.status,
        statusText: response.statusText,
        headers: response.headers,
      });
    });
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
