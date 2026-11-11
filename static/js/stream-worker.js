// SharedWorker for multiplexing Server-Sent Events across multiple browser tabs.
// Maintains a single unscoped EventSource connection to /api/stream and routes
// events to subscribed ports by project scope (no scope = every event).

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

const GLOBAL_EVENTS = new Set([
  'project_created',
  'project_deleted',
  'project_renamed',
  'settings_updated',
  'stream_reset',
]);

const ports = new Map(); // port -> { project } (project null = every event)
let eventSource = null;
let reconnectTimer = null;
let isConnected = false;
let idleTimer = null;

function ensureConnection() {
  if (idleTimer) {
    clearTimeout(idleTimer);
    idleTimer = null;
  }
  if (eventSource || reconnectTimer || ports.size === 0) return;

  try {
    eventSource = new EventSource('/api/stream');

    eventSource.addEventListener('open', () => {
      isConnected = true;
      for (const port of ports.keys()) {
        try {
          port.postMessage({ type: 'open' });
        } catch (_) {}
      }
    });

    eventSource.addEventListener('error', () => {
      isConnected = false;
      closeEventSource();
      for (const port of ports.keys()) {
        try {
          port.postMessage({ type: 'error' });
        } catch (_) {}
      }
      scheduleReconnect();
    });

    for (const evtName of KNOWN_EVENTS) {
      eventSource.addEventListener(evtName, (event) => {
        try {
          const data = JSON.parse(event.data);
          broadcast(evtName, data);
        } catch (err) {
          console.error('[SharedWorker] Failed to parse event', evtName, err);
        }
      });
    }
  } catch (err) {
    console.error('[SharedWorker] Failed to create EventSource', err);
    scheduleReconnect();
  }
}

function scheduleReconnect() {
  if (reconnectTimer || ports.size === 0) return;
  reconnectTimer = setTimeout(() => {
    reconnectTimer = null;
    ensureConnection();
  }, 2000);
}

function closeEventSource() {
  if (eventSource) {
    try {
      eventSource.close();
    } catch (_) {}
    eventSource = null;
  }
  isConnected = false;
}

function broadcast(eventType, data) {
  const isGlobal = GLOBAL_EVENTS.has(eventType);
  const targetProject = data && data.project;
  const oldProject = data && data.old_project;
  const affected = Array.isArray(data?.affected_projects) ? data.affected_projects : null;

  for (const [port, sub] of ports.entries()) {
    if (!isGlobal && sub.project) {
      const matchesTarget = targetProject ? (sub.project === targetProject || sub.project === oldProject) : true;
      const matchesAffected = affected ? affected.includes(sub.project) : false;
      if (!matchesTarget && !matchesAffected) {
        continue;
      }
    }
    try {
      port.postMessage({ type: 'event', eventType, data });
    } catch (_) {
      ports.delete(port);
    }
  }
}

self.onconnect = function (event) {
  const port = event.ports[0];

  port.onmessage = function (e) {
    const msg = e.data || {};
    if (msg.action === 'subscribe') {
      ports.set(port, { project: msg.project || null });
      ensureConnection();
      if (isConnected) {
        try {
          port.postMessage({ type: 'open' });
        } catch (_) {}
      }
    } else if (msg.action === 'unsubscribe') {
      ports.delete(port);
      if (ports.size === 0 && !idleTimer) {
        // Grace period so a reload/navigation reuses the open connection.
        idleTimer = setTimeout(() => {
          idleTimer = null;
          if (ports.size === 0) closeEventSource();
        }, 10000);
      }
    }
  };

  port.start();
};
