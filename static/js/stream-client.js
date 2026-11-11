// Client wrapper for SharedWorker SSE multiplexing with automatic EventSource fallback.
(function (global) {
  const KNOWN_EVENTS = [
    'agent_status',
    'question',
    'agent_stream_reset',
    'agent_reasoning_delta',
    'agent_content_delta',
    'agent_tool_delta',
    'agent_stream_end',
    'agent_message',
    'agent_usage',
    'tool_status',
    'preview_updated',
    'revision_updated',
    'screenshot_updated',
    'review_updated',
    'agent_error',
    'agent_stopped',
    'export_ready',
    'conversation_reset',
    'user_message',
    'project_created',
    'project_deleted',
    'project_renamed',
    'settings_updated',
    'stream_reset',
  ];

  /**
   * Subscribe to server events. ``project`` scopes delivery to one project
   * (plus global events); omit it to receive every event (dashboard).
   * Returns ``{ close() }``.
   */
  function connectSharedStream(options) {
    const { project, onEvent, onOpen, onError } = options || {};
    const scope = project || null;

    if (typeof SharedWorker !== 'undefined') {
      try {
        const worker = new SharedWorker('/static/js/stream-worker.js', 'cad-agent-stream');
        const port = worker.port;

        port.onmessage = function (e) {
          const msg = e.data;
          if (!msg) return;
          if (msg.type === 'open') {
            if (onOpen) onOpen();
          } else if (msg.type === 'error') {
            if (onError) onError();
          } else if (msg.type === 'event') {
            if (onEvent) onEvent(msg.eventType, msg.data);
          }
        };

        port.start();
        port.postMessage({ action: 'subscribe', project: scope });

        const unsubscribe = function () {
          try {
            port.postMessage({ action: 'unsubscribe' });
            port.close();
          } catch (_) {}
        };
        window.addEventListener('beforeunload', unsubscribe);

        return {
          close() {
            window.removeEventListener('beforeunload', unsubscribe);
            unsubscribe();
          },
        };
      } catch (err) {
        console.warn('[stream-client] SharedWorker unavailable, falling back to direct EventSource', err);
      }
    }

    // Direct EventSource fallback (server-side scoping via ``?project=``).
    let es = null;
    let reconnectTimer = null;
    let isClosed = false;

    function startDirect() {
      if (isClosed) return;
      const url = '/api/stream' + (scope ? '?project=' + encodeURIComponent(scope) : '');
      es = new EventSource(url);

      es.addEventListener('open', () => {
        if (onOpen) onOpen();
      });

      es.addEventListener('error', () => {
        if (onError) onError();
        if (es) {
          es.close();
          es = null;
        }
        if (!isClosed && !reconnectTimer) {
          reconnectTimer = setTimeout(() => {
            reconnectTimer = null;
            startDirect();
          }, 2000);
        }
      });

      for (const evtName of KNOWN_EVENTS) {
        es.addEventListener(evtName, (event) => {
          try {
            const data = JSON.parse(event.data);
            if (onEvent) onEvent(evtName, data);
          } catch (err) {
            console.error('[stream-client fallback] Failed to parse event', evtName, err);
          }
        });
      }
    }

    startDirect();

    return {
      close() {
        isClosed = true;
        if (reconnectTimer) {
          clearTimeout(reconnectTimer);
          reconnectTimer = null;
        }
        if (es) {
          es.close();
          es = null;
        }
      },
    };
  }

  global.connectSharedStream = connectSharedStream;
})(typeof window !== 'undefined' ? window : this);
