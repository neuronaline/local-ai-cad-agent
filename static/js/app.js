import { CadViewer } from './viewer.js';

const feed = document.querySelector('#chat-feed');
const chatForm = document.querySelector('#chat-form');
const message = document.querySelector('#message');
const dropzone = document.querySelector('#dropzone');
const attachments = document.querySelector('#attachments');
const attachmentLabel = document.querySelector('#attachment-label');
const stopButton = document.querySelector('#stop');
const questionArea = document.querySelector('#question-area');
const renderSection = document.querySelector('#render-section');
const renderBody = document.querySelector('#render-body');
const renderImage = document.querySelector('#render-image');
const renderToggle = document.querySelector('#render-toggle');
const activityPanel = document.querySelector('#activity-panel');
const activityTitle = document.querySelector('#activity-title');
const activityList = document.querySelector('#activity-list');
const attachmentPreview = document.querySelector('#attachment-preview');
const modelActions = document.querySelector('#model-actions');
const viewer = new CadViewer(document.querySelector('#viewer'), document.querySelector('#dimensions'));
const appConfig = JSON.parse(document.querySelector('#app-config')?.textContent || '{}');
const showInfoMessages = appConfig.showInfoMessages ?? true;
const currentProject = appConfig.projectName || '';

let selectedFiles = [];
let lastStreamedAgent = null;
let previewProject = '';
let loadedPreviewRevision = '';
let previewLoadPromise = null;
const agentStreams = new Map();
const toolMessages = new Map();
const activityItems = new Map();
const ALLOWED_TAGS = new Set([
  'p', 'br', 'b', 'strong', 'i', 'em', 'u', 's', 'del',
  'code', 'pre', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
  'ul', 'ol', 'li', 'a', 'blockquote', 'hr',
  'table', 'thead', 'tbody', 'tr', 'th', 'td',
  'span', 'div', 'details', 'summary',
]);
const ALLOWED_ATTRS = new Set(['href', 'title', 'class', 'id']);
const DROP_TAGS = new Set(['script', 'style', 'iframe', 'object', 'embed', 'link', 'meta', 'base', 'noscript', 'template']);

function isSafeHref(value) {
  const url = value.replace(/[\u0000-\u001F\u007F]/g, '').trim();
  if (/^(https?:|mailto:)/i.test(url)) return true;
  return !/^[a-z][a-z0-9+.-]*:/i.test(url);
}

function sanitizeHTML(html) {
  const doc = new DOMParser().parseFromString(html, 'text/html');
  const walker = doc.createTreeWalker(doc.body, NodeFilter.SHOW_ELEMENT);
  const elements = [];
  while (walker.nextNode()) elements.push(walker.currentNode);
  for (const el of elements) {
    if (el === doc.body) continue;
    const tag = el.tagName.toLowerCase();
    if (DROP_TAGS.has(tag)) {
      el.remove();
    } else if (!ALLOWED_TAGS.has(tag)) {
      el.replaceWith(...el.childNodes);
    } else {
      for (const attr of [...el.attributes]) {
        const name = attr.name.toLowerCase();
        if (!ALLOWED_ATTRS.has(name)) {
          el.removeAttribute(attr.name);
        } else if (name === 'href' && !isSafeHref(attr.value)) {
          el.removeAttribute('href');
        }
      }
    }
  }
  return doc.body.innerHTML;
}

marked.setOptions({
  highlight: (code, lang) => {
    if (lang && hljs.getLanguage(lang)) {
      return hljs.highlight(code, { language: lang }).value;
    }
    return hljs.highlightAuto(code).value;
  },
});

function renderAgentContent(item, text) {
  item.dataset.raw = text;
  item.querySelector('.message-content').innerHTML = sanitizeHTML(marked.parse(text || ''));
}

function addMessage(text, type = 'agent', options = {}) {
  const item = document.createElement('div');
  item.className = `message ${type}`;
  item.dataset.raw = text;
  if (type === 'agent') {
    item.innerHTML = `
      <div class="message-meta">
        <span class="agent-mark">AI</span>
        <span class="message-author">Agent</span>
        <span class="message-state">${options.streaming ? 'Responding' : ''}</span>
      </div>
      <div class="message-content"></div>
    `;
    feed.appendChild(item);
    renderAgentContent(item, text || '');
    agentStreams.set('latest', { item, text: text || '' });
    return item;
  }
  item.textContent = text || '';
  feed.appendChild(item);
  return item;
}

const activityLabels = {
  agent: 'Agent', cad: 'Building model', file: 'Updating files', screenshot: 'Visual verification',
  usage: 'Usage', terminal: 'Checking', preparing: 'Preparing', running: 'Running',
  reviewing: 'Reviewing', completed: 'Completed', error: 'Error', stopped: 'Stopped', started: 'Started',
};

function activityLabel(value) {
  return activityLabels[value] || String(value || 'Activity').replace(/[_-]+/g, ' ');
}

function clearActivity() {
  activityItems.clear();
  toolMessages.clear();
  activityList.replaceChildren();
  activityPanel.hidden = true;
}

function updateActivitySummary() {
  const items = [...activityItems.values()];
  if (!items.length) {
    activityPanel.hidden = true;
    return;
  }
  const running = items.filter(item => ['preparing', 'running', 'started', 'reviewing'].includes(item.status));
  const failed = items.filter(item => item.status === 'error');
  const completed = items.filter(item => item.status === 'completed').length;
  const current = running.at(-1);
  activityTitle.textContent = current
    ? `${activityLabel(current.tool)} · ${activityLabel(current.status)}`
    : failed.length
      ? `${failed.length} failed ${failed.length === 1 ? 'task' : 'tasks'}`
      : `${completed || items.length} ${completed === 1 ? 'task' : 'tasks'} completed`;
  activityPanel.open = Boolean(current);
  activityPanel.hidden = false;
}

function addToolMessage(data) {
  const callId = data.call_id || crypto.randomUUID();
  const status = data.status || 'running';
  const item = activityItems.get(callId) || {callId, tool: data.tool || 'agent'};
  item.tool = data.tool || item.tool;
  item.status = status;
  item.result = data.result || item.result || '';
  activityItems.set(callId, item);
  toolMessages.set(callId, item);

  let row = activityList.querySelector(`[data-call-id="${CSS.escape(callId)}"]`);
  if (!row) {
    row = document.createElement('div');
    row.className = 'activity-item';
    row.dataset.callId = callId;
    activityList.appendChild(row);
  }
  row.dataset.status = status;
  row.replaceChildren();
  const name = document.createElement('span');
  name.className = 'activity-name';
  name.textContent = activityLabel(item.tool);
  const state = document.createElement('span');
  state.className = 'activity-state';
  state.textContent = activityLabel(status);
  row.append(name, state);
  if (item.result) {
    const detail = document.createElement('span');
    detail.className = 'activity-detail';
    detail.textContent = item.result.length > 180 ? `${item.result.slice(0, 177)}…` : item.result;
    row.appendChild(detail);
  }
  updateActivitySummary();
  return row;
}

function setThinking(active) {
  stopButton.hidden = !active;
  if (active) {
    if (!document.querySelector('.thinking-indicator')) {
      const el = document.createElement('div');
      el.className = 'thinking-indicator';
      el.innerHTML = '<span></span><span></span><span></span><span>Preparing model</span>';
      feed.appendChild(el);
    }
  } else {
    const existing = document.querySelector('.thinking-indicator');
    if (existing) existing.remove();
  }
}

async function api(url, options = {}) {
  const response = await fetch(url, options);
  const isJson = (response.headers.get('content-type') || '').includes('application/json');
  const body = isJson ? await response.json() : await response.text();
  if (!response.ok) {
    const message = (isJson && body && body.error) || `Request failed (${response.status})`;
    throw new Error(message);
  }
  return body;
}

async function loadCurrentPreview(previewId) {
  if (!currentProject || previewLoadPromise) return;
  previewLoadPromise = (async () => {
    const project = currentProject;
    try {
      const url = `/api/projects/${encodeURIComponent(project)}/preview?ts=${Date.now()}`;
      await viewer.load(url);
      previewProject = project;
      const meta = await api(`/api/projects/${encodeURIComponent(project)}/preview/meta`);
      loadedPreviewRevision = meta.revision || loadedPreviewRevision;
      if (previewId) {
        try {
          await api(`/api/projects/${encodeURIComponent(project)}/preview/displayed`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({preview_id: previewId}),
          });
        } catch (err) {
          await api(`/api/projects/${encodeURIComponent(project)}/preview/failed`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({preview_id: previewId, message: err.message}),
          });
        }
      }
      await refreshRender();
      modelActions.hidden = false;
    } catch (error) {
      addMessage(`Preview failed: ${error.message}`, 'error');
    } finally {
      previewLoadPromise = null;
    }
  })();
  return previewLoadPromise;
}

async function refreshRender() {
  if (!currentProject) return;
  try {
    const res = await fetch(`/api/projects/${encodeURIComponent(currentProject)}/render?ts=${Date.now()}`);
    if (!res.ok) {
      renderSection.hidden = true;
      return;
    }
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    renderImage.src = url;
    renderSection.hidden = false;
  } catch {
    renderSection.hidden = true;
  }
}

async function syncCurrentPreview() {
  if (!currentProject || previewLoadPromise) return;
  try {
    const meta = await api(`/api/projects/${encodeURIComponent(currentProject)}/preview/meta`);
    if (
      meta.available
      && (previewProject !== currentProject || loadedPreviewRevision !== meta.revision)
    ) {
      await loadCurrentPreview();
    }
  } catch {
    // SSE is the primary path; polling is only a reconnect fallback.
  }
}

async function loadCurrentState() {
  if (!currentProject) return;
  const data = await api(`/api/projects/${encodeURIComponent(currentProject)}/state`);
  if (data.status === 'waiting_for_user') {
    const q = data.question || {};
    if (q.questions) {
      const firstQ = q.questions[0] || {};
      const preview = q.questions.length > 1
        ? `${q.title || 'Questions'} (${q.questions.length} fields)`
        : firstQ.question || '';
      if (preview) addMessage(preview);
    } else if (q.question) {
      addMessage(q.question);
    }
    showQuestion({project: currentProject, ...q});
  } else if (data.status === 'running') {
    setThinking(true);
  } else if (data.status === 'rendering' && data.preview_id) {
    setThinking(true);
    await loadCurrentPreview(data.preview_id);
  }
}

async function loadHistory(projectName) {
  try {
    const data = await api(`/api/projects/${encodeURIComponent(projectName)}/history`);
    agentStreams.clear();
    clearActivity();
    lastStreamedAgent = null;
    feed.replaceChildren();
    questionArea.replaceChildren();
    for (const evt of data.events) {
      const role = evt.role || '';
      const content = evt.content || '';
      if (role === 'user') {
        addMessage(content, 'user');
      } else if (role === 'assistant' || role === 'agent') {
        addMessage(content, 'agent');
      } else if (evt.type === 'agent_error') addMessage(evt.data?.message || content, 'error');
      else if (showInfoMessages) addInfoMessage(evt.type, evt.data);
    }
    if (!data.events.length) {
      const empty = document.createElement('div');
      empty.className = 'empty-state';
      empty.textContent = `Project "${projectName}" selected. Describe a part to begin.`;
      feed.appendChild(empty);
    }
  } catch (error) {
    addMessage(error.message, 'error');
  }
}

function addInfoMessage(type, data = {}) {
  if (type === 'agent_status') {
    addToolMessage({
      call_id: `status-${data.timestamp || crypto.randomUUID()}`,
      tool: 'agent',
      status: data.status || 'info',
      result: data.message,
    });
  } else if (type === 'tool_status') {
    addToolMessage(data);
  } else if (type === 'agent_usage') {
    const cache = Number(data.cached_tokens || 0);
    addToolMessage({
      call_id: `usage-${crypto.randomUUID()}`,
      tool: 'usage',
      status: 'completed',
      result: `Prompt ${data.prompt_tokens ?? '—'} · Completion ${data.completion_tokens ?? '—'} · Cached ${cache}`,
    });
  } else if (type === 'agent_stopped') {
    addToolMessage({
      call_id: `stopped-${crypto.randomUUID()}`,
      tool: 'agent',
      status: 'stopped',
      result: 'Agent task stopped.',
    });
  }
}

message.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    chatForm.dispatchEvent(new Event('submit'));
  }
});

chatForm.addEventListener('submit', async event => {
  event.preventDefault();
  if (!currentProject) return addMessage('Create a project first.', 'error');
  const text = message.value.trim();
  if (!text) return;
  const btn = chatForm.querySelector('button[type="submit"]');
  btn.disabled = true;
  message.disabled = true;
  try {
    clearActivity();
    addMessage(text, 'user');
    setThinking(true);
    message.value = '';
    const idempotencyKey = crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random().toString(36).slice(2)}`;
    const body = new FormData();
    body.append('project', currentProject);
    body.append('message', text);
    body.append('idempotency_key', idempotencyKey);
    selectedFiles.forEach(file => body.append('attachments', file));
    const response = await api('/api/chat', {method: 'POST', body});
    if (response.duplicate) {
      setThinking(false);
      return;
    }
    if (response.attachments?.length) {
      addToolMessage({
        call_id: `attachments-${crypto.randomUUID()}`,
        tool: 'Images',
        status: 'completed',
        result: `${response.attachments.length} reference image(s) uploaded.`,
      });
    }
    clearAttachments();
  } catch (error) {
    addMessage(error.message, 'error');
    setThinking(false);
  } finally {
    btn.disabled = false;
    message.disabled = false;
    message.focus();
  }
});

stopButton.addEventListener('click', async () => {
  if (!currentProject) return;
  stopButton.disabled = true;
  try {
    await api('/api/stop', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({project: currentProject}),
    });
  } catch (error) {
    addMessage(error.message, 'error');
  } finally {
    stopButton.disabled = false;
  }
});

attachments.addEventListener('change', () => {
  selectedFiles = Array.from(attachments.files || []);
  renderAttachmentPreview();
});

['dragenter', 'dragover'].forEach(eventName => {
  dropzone.addEventListener(eventName, event => {
    event.preventDefault();
    dropzone.classList.add('is-dragging');
  });
});
['dragleave', 'drop'].forEach(eventName => {
  dropzone.addEventListener(eventName, event => {
    event.preventDefault();
    dropzone.classList.remove('is-dragging');
  });
});
dropzone.addEventListener('drop', event => {
  const files = Array.from(event.dataTransfer?.files || []).filter(file => file.type.startsWith('image/'));
  if (!files.length) return;
  selectedFiles = [...selectedFiles, ...files];
  renderAttachmentPreview();
});

function clearAttachments() {
  selectedFiles = [];
  attachments.value = '';
  if (attachmentLabel) attachmentLabel.textContent = 'Attach';
  attachmentPreview.replaceChildren();
}

function renderAttachmentPreview() {
  attachmentPreview.replaceChildren();
  if (attachmentLabel) {
    attachmentLabel.textContent = selectedFiles.length ? `Images (${selectedFiles.length})` : 'Attach';
  }
  selectedFiles.forEach((file, index) => {
    const item = document.createElement('div');
    item.className = 'attachment-item';
    const image = document.createElement('img');
    const url = URL.createObjectURL(file);
    image.src = url;
    image.alt = file.name;
    image.addEventListener('load', () => URL.revokeObjectURL(url), {once: true});
    const name = document.createElement('span');
    name.textContent = file.name;
    const remove = document.createElement('button');
    remove.type = 'button';
    remove.className = 'quiet icon-only';
    remove.textContent = '×';
    remove.title = `Remove ${file.name}`;
    remove.addEventListener('click', () => {
      selectedFiles.splice(index, 1);
      renderAttachmentPreview();
    });
    item.append(image, name, remove);
    attachmentPreview.appendChild(item);
  });
}

renderToggle.addEventListener('click', () => {
  const expanded = renderToggle.getAttribute('aria-expanded') === 'true';
  renderToggle.setAttribute('aria-expanded', String(!expanded));
  renderToggle.textContent = expanded ? 'Show' : 'Hide';
  renderBody.hidden = expanded;
});

// Viewer toolbar
document.querySelector('#toggle-wireframe')?.addEventListener('click', event => {
  event.currentTarget.setAttribute('aria-pressed', String(viewer.toggleWireframe()));
});
document.querySelector('#toggle-grid')?.addEventListener('click', event => {
  event.currentTarget.setAttribute('aria-pressed', String(viewer.toggleGrid()));
});
document.querySelector('#reset-view')?.addEventListener('click', () => viewer.fit());
document.querySelectorAll('[data-view]').forEach(button => {
  button.addEventListener('click', () => {
    viewer.setView(button.dataset.view);
    document.querySelectorAll('[data-view]').forEach(item => item.classList.toggle('active', item === button));
  });
});

document.querySelector('#approve-design')?.addEventListener('click', () => {
  message.value = 'Finalize the design. Run the final verification and prepare the outputs.';
  chatForm.dispatchEvent(new Event('submit'));
});
document.querySelector('#continue-editing')?.addEventListener('click', () => message.focus());

document.querySelectorAll('[data-mobile-view]').forEach(button => {
  button.addEventListener('click', () => {
    document.body.dataset.mobileView = button.dataset.mobileView;
    document.querySelectorAll('[data-mobile-view]').forEach(item => {
      item.setAttribute('aria-pressed', String(item === button));
    });
  });
});

const resizer = document.querySelector('#panel-resizer');
resizer?.addEventListener('pointerdown', event => {
  event.preventDefault();
  resizer.setPointerCapture(event.pointerId);
  document.body.classList.add('is-resizing');
  const onMove = moveEvent => {
    const width = Math.min(Math.max(moveEvent.clientX, 360), window.innerWidth - 400);
    document.documentElement.style.setProperty('--chat-width', `${width}px`);
  };
  const onEnd = () => {
    document.body.classList.remove('is-resizing');
    resizer.removeEventListener('pointermove', onMove);
    resizer.removeEventListener('pointerup', onEnd);
    resizer.removeEventListener('pointercancel', onEnd);
  };
  resizer.addEventListener('pointermove', onMove);
  resizer.addEventListener('pointerup', onEnd);
  resizer.addEventListener('pointercancel', onEnd);
});

// History drawer
const historyDrawer = document.querySelector('#history-drawer');
const historyContent = document.querySelector('#history-content');
const modelContent = document.querySelector('#model-content');
const historyTab = document.querySelector('#history-tab');
const modelTab = document.querySelector('#model-tab');

function setDrawerTab(active) {
  const isHistory = active === 'history';
  historyTab.setAttribute('aria-pressed', String(isHistory));
  modelTab.setAttribute('aria-pressed', String(!isHistory));
  historyContent.hidden = !isHistory;
  modelContent.hidden = isHistory;
}

document.querySelector('#history-btn')?.addEventListener('click', () => {
  historyDrawer.hidden = false;
  setDrawerTab('history');
  loadHistory(currentProject);
  loadModelPane();
});
document.querySelector('#history-close')?.addEventListener('click', () => {
  historyDrawer.hidden = true;
});
historyTab?.addEventListener('click', () => setDrawerTab('history'));
modelTab?.addEventListener('click', () => setDrawerTab('model'));

async function loadModelPane() {
  if (!currentProject || !modelContent) return;
  modelContent.replaceChildren();
  try {
    const data = await api(`/api/projects/${encodeURIComponent(currentProject)}/revisions?limit=25`);
    const list = document.createElement('ul');
    list.className = 'revision-list';
    for (const rev of data.revisions || []) {
      const item = document.createElement('li');
      item.className = 'revision-item';
      const header = document.createElement('div');
      header.className = 'revision-header';
      header.innerHTML = `
        <span class="revision-id">${rev.id.slice(0, 8)}</span>
        <span class="revision-status">${rev.build_status || 'not_run'}</span>
      `;
      item.appendChild(header);
      const meta = document.createElement('div');
      meta.className = 'revision-meta';
      meta.textContent = new Date(rev.created_at).toLocaleString();
      item.appendChild(meta);
      const actions = document.createElement('div');
      actions.className = 'revision-actions';
      const restore = document.createElement('button');
      restore.type = 'button';
      restore.className = 'quiet';
      restore.textContent = rev.is_active ? 'Active' : 'Restore';
      restore.disabled = rev.is_active;
      restore.addEventListener('click', async () => {
        try {
          await api(`/api/projects/${encodeURIComponent(currentProject)}/revisions/${rev.id}/restore`, {method: 'POST'});
          historyDrawer.hidden = true;
        } catch (error) {
          addMessage(error.message, 'error');
        }
      });
      actions.appendChild(restore);
      item.appendChild(actions);
      list.appendChild(item);
    }
    modelContent.appendChild(list);
  } catch (error) {
    addMessage(error.message, 'error');
  }
}

// Question rendering (delegated to the existing logic in question_tool).
function showQuestion(question) {
  questionArea.replaceChildren();
  const form = document.createElement('form');
  form.className = 'question-form';
  const fields = [];
  if (question.questions?.length) {
    question.questions.forEach((q, index) => {
      const id = q.id || `q-${index}`;
      fields.push({ id, label: q.question || id, options: q.options || [], type: q.type || 'text' });
    });
  } else if (question.question) {
    fields.push({ id: 'answer', label: question.question, options: question.options || [], type: 'text' });
  }
  for (const field of fields) {
    const label = document.createElement('label');
    label.textContent = field.label;
    let input;
    if (field.options?.length) {
      input = document.createElement('select');
      input.required = true;
      for (const opt of field.options) {
        const optionEl = document.createElement('option');
        optionEl.value = opt.value || opt;
        optionEl.textContent = opt.label || opt;
        input.appendChild(optionEl);
      }
    } else if (field.type === 'multi_line') {
      input = document.createElement('textarea');
      input.rows = 2;
    } else {
      input = document.createElement('input');
      input.type = 'text';
    }
    input.name = field.id;
    label.appendChild(input);
    form.appendChild(label);
  }
  const submit = document.createElement('button');
  submit.type = 'submit';
  submit.textContent = 'Send';
  form.appendChild(submit);
  form.addEventListener('submit', async event => {
    event.preventDefault();
    const answers = {};
    for (const field of fields) {
      const el = form.elements.namedItem(field.id);
      if (el) answers[field.id] = el.value;
    }
    submit.disabled = true;
    try {
      await api('/api/questions/answer', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({project: currentProject, answers}),
      });
      questionArea.replaceChildren();
    } catch (error) {
      addMessage(error.message, 'error');
      submit.disabled = false;
    }
  });
  questionArea.appendChild(form);
}

// SSE event stream
let eventSource = null;
function connectStream() {
  if (eventSource) eventSource.close();
  const status = document.querySelector('#connection-status');
  eventSource = new EventSource('/api/stream');
  eventSource.addEventListener('error', () => {
    if (status) status.classList.remove('connected');
    setTimeout(connectStream, 2000);
  });
  eventSource.addEventListener('open', () => {
    if (status) status.classList.add('connected');
  });
  const handlers = {
    agent_status: data => {
      if (data.project !== currentProject) return;
      addToolMessage({
        call_id: `status-${data.timestamp || data.status || crypto.randomUUID()}`,
        tool: 'agent',
        status: data.status || 'running',
        result: data.message,
      });
      if (data.status === 'started' || data.status === 'reviewing') {
        setThinking(true);
      } else if (['stopped', 'failed', 'completed'].includes(data.status)) {
        setThinking(false);
      }
    },
    agent_stream_start: data => {
      if (data.project !== currentProject) return;
      const item = addMessage('', 'agent', {streaming: true});
      agentStreams.set(data.message_id, { item, text: '' });
    },
    agent_message_delta: data => {
      if (data.project !== currentProject) return;
      const entry = agentStreams.get(data.message_id);
      if (!entry) return;
      const delta = (data.content || data.text || '');
      entry.text += delta;
      renderAgentContent(entry.item, entry.text);
    },
    agent_reasoning_delta: data => { /* hidden by policy */ },
    agent_tool_call_delta: data => { /* not surfaced */ },
    agent_stream_end: data => {
      if (data.project !== currentProject) return;
      const entry = agentStreams.get(data.message_id);
      if (entry) {
        entry.item.querySelector('.message-state').textContent = '';
        lastStreamedAgent = entry;
      }
    },
    agent_message: data => {
      if (data.project !== currentProject) return;
      addMessage(data.message || '', 'agent');
    },
    tool_status: data => {
      if (data.project !== currentProject) return;
      addToolMessage(data);
    },
    preview_updated: data => {
      if (data.project !== currentProject) return;
      loadCurrentPreview(data.preview_id);
    },
    revision_updated: data => {
      if (data.project !== currentProject) return;
      syncCurrentPreview();
    },
  };
  for (const [eventName, handler] of Object.entries(handlers)) {
    eventSource.addEventListener(eventName, event => {
      try {
        handler(JSON.parse(event.data));
      } catch (error) {
        console.error('Failed to parse event', eventName, error);
      }
    });
  }
  eventSource.addEventListener('agent_error', event => {
    try {
      const data = JSON.parse(event.data);
      if (data.project !== currentProject) return;
      addMessage(data.message || 'Agent error.', 'error');
    } catch {}
  });
  eventSource.addEventListener('agent_stopped', event => {
    try {
      const data = JSON.parse(event.data);
      if (data.project !== currentProject) return;
      setThinking(false);
    } catch {}
  });
}

(async function init() {
  if (!currentProject) return;
  await loadCurrentState();
  await syncCurrentPreview();
  connectStream();
})();
