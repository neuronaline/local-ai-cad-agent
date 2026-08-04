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
let decisionAttempt = null; // {run_id, attempt_id, revision_id} awaiting a decision
const decisionBar = document.querySelector('#decision-bar');
const acceptDesignBtn = document.querySelector('#accept-design');
const reportIssueBtn = document.querySelector('#report-issue');
const continueEditingBtn = document.querySelector('#continue-editing');
const issueModal = document.querySelector('#issue-modal');
const issueForm = document.querySelector('#issue-form');
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
    if (!previewId || project !== currentProject) {
      refreshDecisionBar();
      return;
    }
    await api(`/api/projects/${encodeURIComponent(project)}/preview/displayed`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({preview_id: previewId}),
    });
    refreshDecisionBar();
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

function hideDecisionBar() {
  decisionAttempt = null;
  decisionBar.hidden = true;
}

async function refreshDecisionBar() {
  if (!currentProject) return hideDecisionBar();
  const project = currentProject;
  try {
    const meta = await api(`/api/projects/${encodeURIComponent(project)}/preview/meta`);
    if (!meta.available || meta.revision !== loadedPreviewRevision || !viewer.hasModel()) {
      return hideDecisionBar();
    }
    const listed = await api(`/api/projects/${encodeURIComponent(project)}/quality/runs?limit=1`);
    const run = listed.runs && listed.runs[0];
    if (!run) return hideDecisionBar();
    const detail = await api(`/api/projects/${encodeURIComponent(project)}/quality/runs/${run.run_id}`);
    const attempt = (detail.attempts || []).find(a => a.status === 'succeeded');
    if (!attempt || attempt.source_sha256 !== meta.model_sha256) return hideDecisionBar();
    if ((detail.decisions || []).some(d => d.attempt_id === attempt.attempt_id)) {
      return hideDecisionBar();
    }
    decisionAttempt = {run_id: run.run_id, attempt_id: attempt.attempt_id, revision_id: attempt.revision_id};
    decisionBar.hidden = false;
  } catch {
    hideDecisionBar();
  }
}

function openIssueModal() {
  if (!decisionAttempt) return;
  issueForm.reset();
  issueModal.hidden = false;
  document.querySelector('#issue-message').focus();
}

function closeIssueModal() {
  issueModal.hidden = true;
  reportIssueBtn.focus();
}

acceptDesignBtn.addEventListener('click', async () => {
  if (!decisionAttempt) return;
  const attemptId = decisionAttempt.attempt_id;
  try {
    await api(`/api/projects/${encodeURIComponent(currentProject)}/quality/attempts/${attemptId}/decision`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({decision: 'accepted'}),
    });
    hideDecisionBar();
    addMessage('Design accepted.');
  } catch (error) {
    addMessage(error.message, 'error');
  }
});

reportIssueBtn.addEventListener('click', openIssueModal);

continueEditingBtn.addEventListener('click', hideDecisionBar);

document.querySelector('#issue-cancel').addEventListener('click', () => {
  closeIssueModal();
});

issueModal.addEventListener('click', event => {
  if (event.target === issueModal) closeIssueModal();
});

issueForm.addEventListener('submit', async event => {
  event.preventDefault();
  if (!decisionAttempt) return;
  const submitButton = issueForm.querySelector('button[type="submit"]');
  submitButton.disabled = true;
  try {
    const payload = {
      category: document.querySelector('#issue-category').value,
      severity: document.querySelector('#issue-severity').value,
      message: document.querySelector('#issue-message').value,
      camera: viewer.getCameraState(),
    };
    try {
      if (viewer.hasModel()) payload.screenshot = viewer.captureScreenshot('current');
    } catch {
      // Camera capture is best-effort; the report still records camera state.
    }
    await api(`/api/projects/${encodeURIComponent(currentProject)}/quality/attempts/${decisionAttempt.attempt_id}/issues`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload),
    });
    hideDecisionBar();
    closeIssueModal();
    addMessage('Issue reported — ask the agent to fix it.');
  } catch (error) {
    addMessage(error.message, 'error');
  } finally {
    submitButton.disabled = false;
  }
});

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
    if (response.attachments?.length) {
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
        if (retryResponse.attachments?.length) {
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
onProjectEvent('design_accepted', data => {
  hideDecisionBar();
  addMessage('Design accepted.', 'user', {});
});
onProjectEvent('design_rejected', data => {
  hideDecisionBar();
  addMessage('Design rejected — revision requires correction.', 'user', {});
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

// ── History Drawer ──

const historyBtn = document.querySelector('#history-btn');
const historyDrawer = document.querySelector('#history-drawer');
const historyClose = document.querySelector('#history-close');
const historyContent = document.querySelector('#history-content');

function openHistory() {
  historyDrawer.hidden = false;
  loadRevisions();
}

function closeHistory() {
  historyDrawer.hidden = true;
}

historyBtn.addEventListener('click', openHistory);
historyClose.addEventListener('click', closeHistory);

document.addEventListener('keydown', event => {
  if (event.key !== 'Escape') return;
  if (!issueModal.hidden) closeIssueModal();
  else if (!historyDrawer.hidden) closeHistory();
});

function formatRevisionTime(iso) {
  if (!iso) return '';
  try {
    const d = new Date(iso);
    return d.toLocaleString(undefined, {month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit'});
  } catch {
    return iso;
  }
}

function formatOrigin(origin) {
  if (!origin) return '';
  const kind = origin.kind || '';
  const labels = {agent_edit: 'Edit', restore: 'Restore', import: 'Import', recovery: 'Recovery'};
  let label = labels[kind] || kind;
  if (origin.operation && kind === 'agent_edit') label += ` (${origin.operation})`;
  return label;
}

async function loadRevisions(before = '') {
  if (!before) historyContent.innerHTML = '<div class="history-empty">Loading…</div>';
  try {
    const query = new URLSearchParams({limit: '50'});
    if (before) query.set('before', before);
    const data = await api(`/api/projects/${encodeURIComponent(currentProject)}/revisions?${query}`);
    if (!data.revisions || !data.revisions.length) {
      if (before) return;
      historyContent.innerHTML = `
        <div class="history-empty">No revisions yet. Edit model.py to create one.</div>
        <div class="history-info">
          The agent may edit only part of model.py; each preview rebuilds the
          complete script in a fresh sandbox.
        </div>`;
      loadConstraintPanel();
      return;
    }
    const constraintSection = historyContent.querySelector('.constraint-section');
    constraintSection?.remove();
    historyContent.querySelector('.revision-more')?.remove();
    if (!before) {
      historyContent.replaceChildren();
      const info = document.createElement('div');
      info.className = 'history-info';
      info.textContent = 'The agent may edit only part of model.py; each preview rebuilds the complete script in a fresh sandbox.';
      historyContent.append(info);
    }
    for (const rev of data.revisions) {
      historyContent.append(createRevisionCard(rev));
    }
    if (data.next_before) {
      const more = document.createElement('button');
      more.className = 'quiet revision-more';
      more.textContent = 'Load older revisions';
      more.addEventListener('click', () => loadRevisions(data.next_before));
      historyContent.append(more);
    }
    if (constraintSection) historyContent.append(constraintSection);
    else loadConstraintPanel();
  } catch (error) {
    const item = document.createElement('div');
    item.className = 'history-error';
    item.textContent = error.message;
    historyContent.replaceChildren(item);
  }
}

function createRevisionCard(rev) {
  const card = document.createElement('div');
  card.className = 'revision-item';
  if (rev.is_active) card.classList.add('active');
  if (rev.is_last_known_good) card.classList.add('lkg');

  const badges = document.createElement('div');
  badges.className = 'revision-badges';
  if (rev.is_active) badges.append(makeBadge('active', 'Active'));
  if (rev.is_last_known_good) badges.append(makeBadge('lkg', 'Last Good'));
  badges.append(makeBadge(rev.build_status, rev.build_status.replace('_', ' ')));
  if (rev.acceptance) {
    badges.append(makeBadge(
      `acceptance-${rev.acceptance.decision}`,
      rev.acceptance.decision.replace(/_/g, ' '),
    ));
  }

  const time = document.createElement('span');
  time.className = 'revision-time';
  time.textContent = formatRevisionTime(rev.created_at);

  const top = document.createElement('div');
  top.className = 'revision-top';
  top.append(badges, time);

  const origin = document.createElement('div');
  origin.className = 'revision-origin';
  origin.textContent = formatOrigin(rev.origin);

  card.append(top, origin);

  if (rev.metrics) {
    const m = rev.metrics;
    const dims = m.dimensions_mm || {};
    const metrics = document.createElement('div');
    metrics.className = 'revision-metrics';
    metrics.textContent = [
      m.solid_count != null ? `${m.solid_count} solid(s)` : '',
      m.volume_mm3 != null ? `${m.volume_mm3} mm³` : '',
      dims.x ? `${dims.x}×${dims.y}×${dims.z} mm` : '',
    ].filter(Boolean).join(' · ');
    card.append(metrics);
  }

  if (rev.error) {
    const err = document.createElement('div');
    err.className = 'revision-metrics';
    err.style.color = '#ff9eaa';
    err.textContent = rev.error.slice(0, 200);
    card.append(err);
  }

  if (rev.open_issues > 0) {
    const issues = document.createElement('div');
    issues.className = 'revision-metrics';
    issues.style.color = '#ffb4a8';
    issues.textContent = `${rev.open_issues} open issue${rev.open_issues === 1 ? '' : 's'}`;
    card.append(issues);
  }

  const issueDetails = document.createElement('div');
  issueDetails.className = 'revision-issues';
  issueDetails.hidden = true;
  card.append(issueDetails);

  // Diff (lazy-loaded on click).
  const diffContainer = document.createElement('div');
  diffContainer.className = 'revision-diff';
  diffContainer.hidden = true;
  card.append(diffContainer);

  const actions = document.createElement('div');
  actions.className = 'revision-actions';

  const diffBtn = document.createElement('button');
  diffBtn.className = 'quiet';
  diffBtn.textContent = 'Diff';
  diffBtn.addEventListener('click', async () => {
    if (!diffContainer.hidden) {
      diffContainer.hidden = true;
      return;
    }
    diffContainer.hidden = false;
    diffContainer.textContent = 'Loading diff…';
    try {
      const data = await api(`/api/projects/${encodeURIComponent(currentProject)}/revisions/${rev.id}/diff`);
      diffContainer.textContent = data.diff || '(no changes)';
      diffContainer.classList.toggle('truncated', data.truncated);
    } catch (error) {
      diffContainer.textContent = error.message;
    }
  });
  actions.append(diffBtn);

  if (rev.open_issues > 0) {
    const issuesBtn = document.createElement('button');
    issuesBtn.className = 'quiet';
    issuesBtn.textContent = 'Issues';
    issuesBtn.addEventListener('click', async () => {
      if (!issueDetails.hidden) {
        issueDetails.hidden = true;
        return;
      }
      issueDetails.hidden = false;
      issueDetails.textContent = 'Loading issues…';
      try {
        const query = new URLSearchParams({open: 'true', revision_id: rev.id});
        const data = await api(`/api/projects/${encodeURIComponent(currentProject)}/quality/issues?${query}`);
        issueDetails.replaceChildren();
        for (const issue of data.issues || []) {
          const row = document.createElement('div');
          row.className = 'revision-metrics';
          const resolveBtn = document.createElement('button');
          resolveBtn.className = 'quiet';
          resolveBtn.textContent = 'Resolve';
          resolveBtn.addEventListener('click', async () => {
            resolveBtn.disabled = true;
            try {
              await api(`/api/projects/${encodeURIComponent(currentProject)}/quality/issues/${issue.issue_id}/resolve`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({confirmed_by: 'user'}),
              });
              loadRevisions();
            } catch (error) {
              resolveBtn.disabled = false;
              addMessage(error.message, 'error');
            }
          });
          row.textContent = `${issue.severity}: ${issue.message || issue.category} `;
          row.append(resolveBtn);
          issueDetails.append(row);
        }
        if (!issueDetails.childElementCount) issueDetails.textContent = 'No open issues.';
      } catch (error) {
        issueDetails.textContent = error.message;
      }
    });
    actions.append(issuesBtn);
  }

  if (!rev.is_active) {
    const restoreBtn = document.createElement('button');
    restoreBtn.className = 'quiet';
    restoreBtn.textContent = 'Restore';
    if (rev.build_status === 'failed' || rev.build_status === 'not_run') {
      restoreBtn.title = 'This revision has no successful build; restore at your own risk.';
    }
    restoreBtn.addEventListener('click', () => restoreRevision(rev.id, rev.build_status));
    actions.append(restoreBtn);
  }

  if (rev.is_last_known_good && !rev.is_active) {
    const lkgBtn = document.createElement('button');
    lkgBtn.textContent = 'Restore Last Good';
    lkgBtn.addEventListener('click', () => restoreRevision(rev.id, 'succeeded'));
    actions.prepend(lkgBtn);
  }

  card.append(actions);
  return card;
}

function makeBadge(status, label) {
  const badge = document.createElement('span');
  badge.className = `rev-badge ${status}`;
  badge.textContent = label;
  return badge;
}

// ── Constraint Panel ──

async function loadConstraintPanel() {
  const section = document.createElement('div');
  section.className = 'constraint-section';
  section.innerHTML = `
    <h3 class="constraint-heading">Protected</h3>
    <p class="constraint-note">Pins protect named source definitions, not arbitrary mesh faces.</p>
    <div class="constraint-loading">Loading…</div>`;
  historyContent.append(section);

  try {
    const data = await api(`/api/projects/${encodeURIComponent(currentProject)}/constraints`);
    const constraints = data.constraints || [];
    const targets = data.targets || {parameters: [], features: []};
    const container = section.querySelector('.constraint-loading');
    container.replaceChildren();
    container.className = 'constraint-list';

    if (!constraints.length) {
      const empty = document.createElement('div');
      empty.className = 'constraint-empty';
      empty.textContent = 'No parameters or features are pinned.';
      container.append(empty);
    }

    for (const c of constraints) {
      const row = document.createElement('div');
      row.className = 'constraint-row';
      row.dataset.constraintId = c.id;

      const badge = document.createElement('span');
      badge.className = `rev-badge ${c.kind === 'parameter' ? 'active' : 'lkg'}`;
      badge.textContent = c.kind === 'parameter' ? 'param' : 'feature';
      row.append(badge);

      const name = document.createElement('span');
      name.className = 'constraint-name';
      name.textContent = c.name;
      row.append(name);

      const unpin = document.createElement('button');
      unpin.className = 'quiet';
      unpin.textContent = 'Unpin';
      unpin.style.fontSize = '.68rem';
      unpin.style.padding = '.2rem .4rem';
      unpin.addEventListener('click', () => unpinConstraint(c.id, c.name));
      row.append(unpin);

      container.append(row);
    }

    const available = [
      ...(targets.parameters || []).filter(target => !target.pinned).map(target => ({
        ...target,
        kind: 'parameter',
        detail: target.value,
      })),
      ...(targets.features || []).filter(target => !target.pinned).map(target => ({
        ...target,
        kind: 'source_feature',
        detail: `lines ${target.start_line}–${target.end_line}`,
      })),
    ];

    if (available.length) {
      const heading = document.createElement('h4');
      heading.className = 'constraint-subheading';
      heading.textContent = 'Available to protect';
      container.append(heading);
    }

    for (const target of available) {
      const row = document.createElement('div');
      row.className = 'constraint-row available';

      const badge = document.createElement('span');
      badge.className = `rev-badge ${target.kind === 'parameter' ? 'active' : 'lkg'}`;
      badge.textContent = target.kind === 'parameter' ? 'param' : 'feature';

      const label = document.createElement('span');
      label.className = 'constraint-name';
      label.textContent = target.name;
      if (target.detail) label.title = target.detail;

      const pin = document.createElement('button');
      pin.className = 'quiet';
      pin.textContent = 'Pin';
      pin.addEventListener('click', () => pinConstraint(target.kind, target.name));

      row.append(badge, label, pin);
      container.append(row);
    }

    if (!available.length && !constraints.length) {
      const hint = document.createElement('div');
      hint.className = 'constraint-empty';
      hint.textContent = 'Add typed parameters or named cad-feature regions to model.py to make them protectable.';
      container.append(hint);
    }
  } catch (error) {
    const container = section.querySelector('.constraint-loading, .constraint-list');
    if (container) {
      container.className = 'history-error';
      container.textContent = error.message;
    }
  }
}

async function reloadConstraintPanel() {
  historyContent.querySelector('.constraint-section')?.remove();
  await loadConstraintPanel();
}

async function pinConstraint(kind, name) {
  try {
    await api(`/api/projects/${encodeURIComponent(currentProject)}/constraints`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({kind, name}),
    });
    await reloadConstraintPanel();
  } catch (error) {
    addMessage(`Pin failed: ${error.message}`, 'error');
  }
}

async function unpinConstraint(id, name) {
  if (!confirm(`Remove pin from '${name}'? The agent will be able to modify it again.`)) return;
  try {
    await api(`/api/projects/${encodeURIComponent(currentProject)}/constraints/${id}`, {
      method: 'DELETE',
      headers: {'Content-Type': 'application/json'},
    });
    // Reload the constraint panel.
    await reloadConstraintPanel();
  } catch (error) {
    addMessage(`Unpin failed: ${error.message}`, 'error');
  }
}

// ── Restore ──

function setRestoring(active, drawerButtons) {
  drawerButtons.forEach(button => { button.disabled = active; });
  chatForm.querySelector('button[type="submit"]').disabled = active;
  message.disabled = active;
  attachments.disabled = active;
  finalizeBtn.disabled = active;
  historyBtn.disabled = active;
  setThinking(active);
  if (active) stopButton.hidden = true;
}

async function restoreRevision(revisionId, buildStatus) {
  const warning = buildStatus === 'failed' || buildStatus === 'not_run'
    ? '\n\nThis revision has no successful build. Restore anyway?'
    : '';
  if (!confirm(`Restore this revision? It will rebuild in the sandbox.${warning}`)) return;

  // Disable conflicting actions during restore.
  const buttons = historyContent.querySelectorAll('button');
  setRestoring(true, buttons);

  try {
    const result = await api(`/api/projects/${encodeURIComponent(currentProject)}/revisions/${revisionId}/restore`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
    });
    if (result.build_status === 'failed') {
      addMessage(`Source restored, but CAD rebuild failed. The displayed preview may be from an older model: ${result.error}`, 'error');
    } else {
      addMessage(`Restored revision ${result.revision_id.slice(0, 8)} and rebuilt successfully.`, 'tool');
    }
    closeHistory();
  } catch (error) {
    addMessage(`Restore failed: ${error.message}`, 'error');
  } finally {
    setRestoring(false, buttons);
  }
}

function refreshHistoryIfOpen() {
  if (!historyDrawer.hidden) loadRevisions();
}

onProjectEvent('revision_updated', () => {
  hideDecisionBar();
  refreshHistoryIfOpen();
});
onProjectEvent('constraint_added', refreshHistoryIfOpen);
onProjectEvent('constraint_removed', refreshHistoryIfOpen);

// Example prompt clicks — place text into input without sending.
feed.addEventListener('click', event => {
  const btn = event.target.closest('.example-prompt');
  if (!btn) return;
  message.value = btn.dataset.prompt || '';
  message.focus();
});
