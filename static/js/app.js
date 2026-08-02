import { CadViewer } from './viewer.js';

const feed = document.querySelector('#chat-feed');
const chatForm = document.querySelector('#chat-form');
const message = document.querySelector('#message');
const dropzone = document.querySelector('#dropzone');
const attachments = document.querySelector('#attachments');
const attachmentLabel = document.querySelector('#attachment-label');
const stopButton = document.querySelector('#stop');
const questionArea = document.querySelector('#question-area');
const viewer = new CadViewer(document.querySelector('#viewer'), document.querySelector('#dimensions'));
const showInfoMessages = window.APP_CONFIG?.showInfoMessages ?? true;
const currentProject = window.APP_CONFIG?.projectName || '';

let selectedFiles = [];
let lastStreamedAgent = null;
let previewProject = '';
let loadedPreviewRevision = '';
let previewLoadPromise = null;
const agentStreams = new Map();
const streamedTools = new Map();
const toolMessages = new Map();
function sanitizeHTML(html) {
  const div = document.createElement('div');
  div.innerHTML = html;
  div.querySelectorAll('script').forEach(el => el.remove());
  div.querySelectorAll('*').forEach(el => {
    [...el.attributes].forEach(attr => {
      if (attr.name.startsWith('on')) el.removeAttribute(attr.name);
    });
  });
  return div.innerHTML;
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
    if (options.messageId) {
      item.dataset.messageId = options.messageId;
      agentStreams.set(options.messageId, item);
    }
    if (options.streaming) item.classList.add('streaming');
    renderAgentContent(item, text);
  } else if (type === 'error') {
    item.textContent = text;
  } else {
    item.textContent = text;
  }
  feed.querySelector('.empty-state')?.remove();
  feed.append(item);
  feed.scrollTop = feed.scrollHeight;
  return item;
}

function formatPayload(value) {
  if (value === undefined || value === null || value === '') return '';
  if (typeof value === 'string') {
    try {
      return JSON.stringify(JSON.parse(value), null, 2);
    } catch {
      return value;
    }
  }
  return JSON.stringify(value, null, 2);
}

function updateToolMessage(item, data) {
  if (data.tool) item.querySelector('.tool-name').textContent = data.tool;
  const status = data.status || 'preparing';
  item.dataset.status = status;
  item.querySelector('.tool-status').textContent = status;
  item.querySelector('.tool-pulse').setAttribute('aria-label', status);
  const argumentsBlock = item.querySelector('.tool-arguments');
  const resultBlock = item.querySelector('.tool-result');
  const argumentsText = Object.hasOwn(data, 'arguments')
    ? formatPayload(data.arguments)
    : argumentsBlock.textContent;
  const resultText = Object.hasOwn(data, 'result')
    ? formatPayload(data.result)
    : resultBlock.textContent;
  if (Object.hasOwn(data, 'arguments')) argumentsBlock.textContent = argumentsText;
  if (Object.hasOwn(data, 'result')) resultBlock.textContent = resultText;
  argumentsBlock.closest('.tool-section').hidden = !argumentsText;
  resultBlock.closest('.tool-section').hidden = !resultText;
  item.open = status === 'running' || status === 'preparing' || status === 'error';
  item.dataset.raw = [
    `${data.tool || 'tool'}: ${status}`,
    argumentsText && `Arguments:\n${argumentsText}`,
    resultText && `Result:\n${resultText}`,
  ].filter(Boolean).join('\n\n');
}

function addToolMessage(data = {}) {
  const callId = data.call_id || data.callId;
  let item = callId ? toolMessages.get(callId) : null;
  if (!item) {
    item = document.createElement('details');
    item.className = 'message tool';
    item.innerHTML = `
      <summary>
        <span class="tool-pulse" aria-label="preparing"></span>
        <span class="tool-name">tool</span>
        <span class="tool-status">preparing</span>
        <span class="tool-chevron">⌄</span>
      </summary>
      <div class="tool-detail">
        <div class="tool-section"><span>Arguments</span><pre class="tool-arguments"></pre></div>
        <div class="tool-section"><span>Result</span><pre class="tool-result"></pre></div>
      </div>
    `;
    feed.querySelector('.empty-state')?.remove();
    feed.append(item);
  }
  if (callId) {
    item.dataset.callId = callId;
    toolMessages.set(callId, item);
  }
  updateToolMessage(item, data);
  feed.scrollTop = feed.scrollHeight;
  return item;
}

function setThinking(visible) {
  stopButton.hidden = !visible;
  feed.querySelector('.thinking-indicator')?.remove();
  if (!visible) return;
  const indicator = document.createElement('div');
  indicator.className = 'thinking-indicator';
  indicator.setAttribute('aria-label', 'AI is thinking');
  indicator.innerHTML = '<span></span><span></span><span></span>';
  feed.querySelector('.empty-state')?.remove();
  feed.append(indicator);
  feed.scrollTop = feed.scrollHeight;
}

async function api(path, options) {
  const response = await fetch(path, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || 'Request failed');
  return data;
}

async function loadCurrentPreview(previewId = '') {
  if (!currentProject) {
    viewer.clear('Select or create a project to begin.');
    return;
  }
  const project = currentProject;
  if (previewProject !== project) {
    previewProject = project;
    loadedPreviewRevision = '';
  }
  try {
    const meta = await api(`/api/projects/${encodeURIComponent(project)}/preview/meta`);
    if (!meta.available) {
      if (!viewer.hasModel()) viewer.clear();
      return;
    }
    if (loadedPreviewRevision !== meta.revision || !viewer.hasModel()) {
      if (previewLoadPromise) await previewLoadPromise;
      if (loadedPreviewRevision !== meta.revision || !viewer.hasModel()) {
        previewLoadPromise = viewer.load(
          `/api/projects/${encodeURIComponent(project)}/preview?v=${encodeURIComponent(meta.revision)}`,
        );
        await previewLoadPromise;
        if (project !== currentProject) return;
        loadedPreviewRevision = meta.revision;
      }
    }
    if (!previewId || project !== currentProject) return;
    await api(`/api/projects/${encodeURIComponent(project)}/preview/displayed`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({preview_id: previewId}),
    });
  } catch (error) {
    if (!previewId || project !== currentProject) return;
    try {
      await api(`/api/projects/${encodeURIComponent(project)}/preview/failed`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({preview_id: previewId, message: error.message}),
      });
    } catch (reportError) {
      addMessage(reportError.message, 'error');
    }
  } finally {
    previewLoadPromise = null;
  }
}

async function syncCurrentPreview() {
  if (!currentProject || previewLoadPromise) return;
  const project = currentProject;
  try {
    const meta = await api(`/api/projects/${encodeURIComponent(project)}/preview/meta`);
    if (
      project === currentProject
      && meta.available
      && (previewProject !== project || loadedPreviewRevision !== meta.revision)
    ) {
      await loadCurrentPreview();
    }
  } catch {
    // SSE remains the primary path; polling is only a reconnect fallback.
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
    streamedTools.clear();
    toolMessages.clear();
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
      else if (evt.type === 'finalized') addFinalizedCard(evt.data);
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
  }
  if (type === 'tool_status') {
    addToolMessage(data);
  }
  if (type === 'agent_usage') {
    const cache = Number(data.cached_tokens || 0);
    addToolMessage({
      call_id: `usage-${crypto.randomUUID()}`,
      tool: 'usage',
      status: 'completed',
      result: `Prompt ${data.prompt_tokens ?? '—'} · Completion ${data.completion_tokens ?? '—'} · Cached ${cache}`,
    });
  }
  if (type === 'agent_stopped') {
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
    addMessage(text, 'user');
    setThinking(true);
    message.value = '';
    const body = new FormData();
    body.append('project', currentProject);
    body.append('message', text);
    selectedFiles.forEach(file => body.append('attachments', file));
    const response = await api('/api/chat', {method: 'POST', body});
    if (response.attachments.length) {
      addMessage(`${response.attachments.length} reference image(s) uploaded.`, 'tool');
    }
    selectedFiles = [];
    attachments.value = '';
    attachmentLabel.textContent = 'Attach';
  } catch (error) {
    const isConflict = error.message.includes('already running') || error.message.includes('pending question');
    if (isConflict) {
      setThinking(false);
      addMessage(error.message, 'error');
    } else {
      addMessage('Connection issue — retrying…', 'tool');
      await new Promise(resolve => setTimeout(resolve, 1000));
      try {
        const retryBody = new FormData();
        retryBody.append('project', currentProject);
        retryBody.append('message', text);
        selectedFiles.forEach(file => retryBody.append('attachments', file));
        const retryResponse = await api('/api/chat', {method: 'POST', body: retryBody});
        if (retryResponse.attachments.length) {
          addMessage(`${retryResponse.attachments.length} reference image(s) uploaded.`, 'tool');
        }
        selectedFiles = [];
        attachments.value = '';
        attachmentLabel.textContent = 'Attach';
      } catch (retryError) {
        setThinking(false);
        addMessage(retryError.message, 'error');
      }
    }
  } finally {
    btn.disabled = false;
    message.disabled = false;
  }
});

document.querySelector('#stop').addEventListener('click', () => {
  api('/api/stop', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({project: currentProject}),
  }).catch(error => addMessage(error.message, 'error'));
});

const finalizeBtn = document.querySelector('#finalize');

function setFinalizing(active) {
  finalizeBtn.disabled = active;
  finalizeBtn.textContent = active ? 'Finalizing…' : 'Finalize';
}

finalizeBtn.addEventListener('click', async () => {
  if (!currentProject) return addMessage('Create a project first.', 'error');
  setFinalizing(true);
  setThinking(true);
  try {
    await api(`/api/projects/${encodeURIComponent(currentProject)}/finalize`, {method: 'POST'});
    setThinking(false);
    setFinalizing(false);
  } catch (error) {
    setThinking(false);
    setFinalizing(false);
    addMessage(error.message, 'error');
  }
});

document.querySelector('#toggle-wireframe').addEventListener('click', event => {
  event.currentTarget.textContent = viewer.toggleWireframe() ? 'Solid' : 'Wireframe';
});
document.querySelector('#toggle-grid').addEventListener('click', event => {
  event.currentTarget.textContent = viewer.toggleGrid() ? 'Grid On' : 'Grid Off';
});
document.querySelector('#reset-view').addEventListener('click', () => viewer.fit());

['dragenter', 'dragover'].forEach(type => {
  dropzone.addEventListener(type, event => {
    event.preventDefault();
    dropzone.classList.add('dragging');
  });
});
['dragleave', 'drop'].forEach(type => {
  dropzone.addEventListener(type, event => {
    event.preventDefault();
    dropzone.classList.remove('dragging');
  });
});
dropzone.addEventListener('drop', event => {
  selectedFiles = [...event.dataTransfer.files].filter(file => file.type.startsWith('image/'));
  attachmentLabel.textContent = selectedFiles.length ? `${selectedFiles.length} attached` : 'Attach';
});
attachments.addEventListener('change', () => {
  selectedFiles = [...attachments.files];
  attachmentLabel.textContent = selectedFiles.length ? `${selectedFiles.length} attached` : 'Attach';
});

function addFinalizedCard(data) {
  const metrics = data.metrics || {};
  const dims = metrics.dimensions_mm || {};
  const dimensionText = dims.x !== undefined
    ? `${dims.x} × ${dims.y} × ${dims.z} mm`
    : '—';

  const item = document.createElement('div');
  item.className = 'message finalized-card';
  item.innerHTML = `
    <div class="finalized-header">
      <span class="finalized-check">✓</span>
      <span class="finalized-title">Finalization Complete</span>
    </div>
    <div class="finalized-body">
      <img class="finalized-render" src="/api/projects/${encodeURIComponent(currentProject)}/render?v=${Date.now()}"
           alt="CAD render" loading="lazy"
           onerror="this.style.display='none'">
      <div class="finalized-metrics">
        <div class="metric"><span>Solids</span><strong>${metrics.solid_count ?? '—'}</strong></div>
        <div class="metric"><span>Volume</span><strong>${metrics.volume_mm3 != null ? `${metrics.volume_mm3} mm³` : '—'}</strong></div>
        <div class="metric"><span>Dimensions</span><strong>${dimensionText}</strong></div>
        <div class="metric"><span>Valid</span><strong>${metrics.is_valid ? 'Yes' : 'No'}</strong></div>
      </div>
      <div class="finalized-links">
        <a class="button-link" href="/api/projects/${encodeURIComponent(currentProject)}/output/model.step" download>STEP</a>
        <a class="button-link" href="/api/projects/${encodeURIComponent(currentProject)}/output/model.stl" download>STL</a>
        <a class="button-link" href="/api/projects/${encodeURIComponent(currentProject)}/output/report" target="_blank">Report</a>
      </div>
    </div>
  `;

  if (data.report_text) {
    const reportSection = document.createElement('div');
    reportSection.className = 'finalized-report';
    reportSection.innerHTML = sanitizeHTML(marked.parse(data.report_text));
    item.querySelector('.finalized-body').append(reportSection);
  }

  feed.querySelector('.empty-state')?.remove();
  feed.append(item);
  feed.scrollTop = feed.scrollHeight;
}

function showQuestion(data) {
  const isMulti = Array.isArray(data.questions) && data.questions.length > 0;
  const questionList = isMulti
    ? data.questions
    : [{id: 'q1', question: data.question, input_type: data.input_type || 'text', options: data.options || [], required: true}];
  const title = isMulti ? (data.title || '') : '';

  const answerForm = document.createElement('form');
  answerForm.className = 'question-form';

  if (title) {
    const heading = document.createElement('div');
    heading.className = 'question-form-title';
    heading.textContent = title;
    answerForm.append(heading);
  }

  const fields = {};

  questionList.forEach(q => {
    const wrap = document.createElement('div');
    wrap.className = 'question-field';

    const label = document.createElement('label');
    label.textContent = q.question;
    if (q.required === false) label.textContent += ' (optional)';
    wrap.append(label);

    const inputType = q.input_type || 'text';

    if (inputType === 'select') {
      const select = document.createElement('select');
      select.required = q.required !== false;
      (q.options || []).forEach(opt => select.add(new Option(opt, opt)));
      wrap.append(select);
      fields[q.id] = select;
    } else if (inputType === 'multiselect') {
      const group = document.createElement('div');
      group.className = 'multiselect-group';
      (q.options || []).forEach(opt => {
        const optLabel = document.createElement('label');
        optLabel.className = 'multiselect-option';
        const cb = document.createElement('input');
        cb.type = 'checkbox';
        cb.value = opt;
        optLabel.append(cb, ' ' + opt);
        group.append(optLabel);
      });
      wrap.append(group);
      fields[q.id] = group;
    } else if (inputType === 'number') {
      const row = document.createElement('div');
      row.className = 'number-row';
      const input = document.createElement('input');
      input.type = 'number';
      input.step = 'any';
      input.required = q.required !== false;
      input.placeholder = 'Numeric value';
      row.append(input);
      const unit = document.createElement('select');
      unit.setAttribute('aria-label', 'Unit');
      ['mm', 'in'].forEach(u => unit.add(new Option(u, u)));
      row.append(unit);
      wrap.append(row);
      fields[q.id] = {input, unit};
    } else {
      const input = document.createElement('input');
      input.type = 'text';
      input.required = q.required !== false;
      input.placeholder = 'Your answer';
      wrap.append(input);
      fields[q.id] = input;
    }

    answerForm.append(wrap);
  });

  const btn = document.createElement('button');
  btn.textContent = 'Reply';
  btn.className = 'question-submit';
  answerForm.append(btn);

  answerForm.addEventListener('submit', async event => {
    event.preventDefault();
    const answers = {};
    let displayText = '';

    questionList.forEach(q => {
      const field = fields[q.id];
      if (!field) return;
      const it = q.input_type || 'text';
      let val = '';
      if (it === 'multiselect') {
        val = [...field.querySelectorAll('input:checked')].map(cb => cb.value).join(', ');
      } else if (it === 'number') {
        const num = field.input.value.trim();
        val = num ? `${num} ${field.unit.value}` : '';
      } else {
        val = field.value.trim();
      }
      answers[q.id] = val;
    });

    const answered = questionList.filter(q => answers[q.id]);
    if (!answered.length) return;
    displayText = answered.map(q => `${q.question}: ${answers[q.id]}`).join('\n');
    addMessage(displayText, 'user');

    try {
      await api('/api/questions/answer', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          project: data.project,
          answers: isMulti ? answers : undefined,
          answer: isMulti ? undefined : displayText,
        }),
      });
      answerForm.remove();
    } catch (error) {
      addMessage(error.message, 'error');
    }
  });

  questionArea.replaceChildren(answerForm);
}

function onProjectEvent(type, callback) {
  events.addEventListener(type, event => {
    const data = JSON.parse(event.data);
    if (data.project === currentProject) callback(data);
  });
}

const events = new EventSource('/api/stream');
events.addEventListener('stream_reset', () => {
  initProject().catch(error => addMessage(error.message, 'error'));
});
onProjectEvent('agent_stream_start', data => {
  setThinking(false);
  lastStreamedAgent = null;
  addMessage('', 'agent', {messageId: data.message_id, streaming: true});
});
onProjectEvent('agent_content_delta', data => {
  let item = agentStreams.get(data.message_id);
  if (!item) {
    item = addMessage('', 'agent', {messageId: data.message_id, streaming: true});
  }
  renderAgentContent(item, `${item.dataset.raw || ''}${data.delta || ''}`);
  feed.scrollTop = feed.scrollHeight;
});
onProjectEvent('agent_reasoning_delta', data => {
  let item = agentStreams.get(data.message_id);
  if (!item) {
    item = addMessage('', 'agent', {messageId: data.message_id, streaming: true});
  }
  let panel = item.querySelector('.reasoning-panel');
  if (!panel) {
    panel = document.createElement('details');
    panel.className = 'reasoning-panel';
    panel.open = true;
    panel.innerHTML = `
      <summary><span>Reasoning</span><span class="reasoning-state">Live</span></summary>
      <div class="reasoning-content"></div>
    `;
    item.querySelector('.message-content').before(panel);
  }
  const content = panel.querySelector('.reasoning-content');
  content.textContent += data.delta || '';
  content.scrollTop = content.scrollHeight;
  feed.scrollTop = feed.scrollHeight;
});
onProjectEvent('agent_tool_call_delta', data => {
  const key = `${data.message_id}:${data.index}`;
  let state = streamedTools.get(key);
  if (!state) {
    state = {name: '', arguments: '', item: addToolMessage({status: 'preparing'})};
    streamedTools.set(key, state);
  }
  state.name += data.name_delta || '';
  state.arguments += data.arguments_delta || '';
  if (data.id) {
    state.item.dataset.callId = data.id;
    toolMessages.set(data.id, state.item);
  }
  updateToolMessage(state.item, {
    tool: state.name || 'tool',
    status: 'preparing',
    arguments: state.arguments,
  });
  feed.scrollTop = feed.scrollHeight;
});
onProjectEvent('agent_stream_end', data => {
  const item = agentStreams.get(data.message_id);
  if (!item) return;
  if (data.message && !item.dataset.raw) renderAgentContent(item, data.message);
  const reasoningPanel = item.querySelector('.reasoning-panel');
  if (!item.dataset.raw && !reasoningPanel) {
    item.remove();
  } else {
    item.classList.remove('streaming');
    item.querySelector('.message-state').textContent = 'Complete';
    if (reasoningPanel) {
      reasoningPanel.querySelector('.reasoning-state').textContent = 'Complete';
    }
    if (item.dataset.raw) lastStreamedAgent = {item, text: item.dataset.raw};
  }
  agentStreams.delete(data.message_id);
});
onProjectEvent('agent_message', data => {
  setThinking(false);
  if (lastStreamedAgent?.item.isConnected && lastStreamedAgent.text === data.message) return;
  addMessage(data.message);
});
onProjectEvent('agent_error', data => {
  setThinking(false);
  setFinalizing(false);
  addMessage(data.message, 'error');
});
onProjectEvent('agent_status', data => {
  if (showInfoMessages) addInfoMessage('agent_status', data);
});
onProjectEvent('tool_status', data => {
  if (showInfoMessages) addInfoMessage('tool_status', data);
});
onProjectEvent('agent_usage', data => {
  if (showInfoMessages) addInfoMessage('agent_usage', data);
});
onProjectEvent('question', data => {
  setThinking(false);
  if (data.questions) {
    const firstQ = data.questions[0] || {};
    const preview = data.questions.length > 1
      ? `${data.title || 'Questions'} (${data.questions.length} fields)`
      : firstQ.question || '';
    if (preview) addMessage(preview);
  } else if (data.question) {
    addMessage(data.question);
  }
  showQuestion(data);
});
onProjectEvent('preview_updated', data => {
  loadCurrentPreview(data.preview_id).catch(error => addMessage(error.message, 'error'));
});
onProjectEvent('screenshot_request', data => {
  const path = `/api/projects/${encodeURIComponent(currentProject)}/screenshot`;
  const reply = payload => api(path, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({request_id: data.request_id, ...payload}),
  });
  try {
    const b64 = viewer.captureScreenshot(data.view || 'current', data.proximity);
    reply({image: b64}).catch(error => {
      reply({error: error.message}).catch(() => {});
      addMessage(error.message, 'error');
    });
  } catch (error) {
    reply({error: error.message}).catch(() => {});
    addMessage(`Screenshot failed: ${error.message}`, 'error');
  }
});
onProjectEvent('finalized', data => {
  setThinking(false);
  setFinalizing(false);
  addFinalizedCard(data);
  loadCurrentPreview().catch(() => {});
});
events.addEventListener('agent_stopped', event => {
  const data = JSON.parse(event.data);
  if (data.project === currentProject) {
    setThinking(false);
    questionArea.replaceChildren();
    if (showInfoMessages) addInfoMessage('agent_stopped', data);
  }
});

events.addEventListener('open', () => {
  const status = document.querySelector('#connection-status');
  status.className = 'connection-status connected';
  status.title = 'Connected';
});
events.addEventListener('error', () => {
  const status = document.querySelector('#connection-status');
  status.className = 'connection-status disconnected';
  status.title = 'Disconnected — reconnecting…';
});

async function initProject() {
  await loadHistory(currentProject);
  loadCurrentPreview();
  await loadCurrentState();
}

initProject().catch(error => addMessage(error.message, 'error'));
setInterval(syncCurrentPreview, 1500);
