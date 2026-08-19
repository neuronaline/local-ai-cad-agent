import { CadViewer } from './viewer.js';

const feed = document.querySelector('#chat-feed');
const chatForm = document.querySelector('#chat-form');
const message = document.querySelector('#message');
const dropzone = document.querySelector('#dropzone');
const attachments = document.querySelector('#attachments');
const attachmentLabel = document.querySelector('#attachment-label');
const sendToggle = document.querySelector('#send-toggle');
const sendIcon = sendToggle?.querySelector('[data-mode="send"]');
const stopIcon = sendToggle?.querySelector('[data-mode="stop"]');
const resetButton = document.querySelector('#reset-context');
const questionArea = document.querySelector('#question-area');
const renderSection = document.querySelector('#render-section');
const renderBody = document.querySelector('#render-body');
const renderImage = document.querySelector('#render-image');
const renderToggle = document.querySelector('#render-toggle');
const reviewSection = document.querySelector('#review-section');
const reviewStatus = document.querySelector('#review-status');
const reviewSummary = document.querySelector('#review-summary');
const reviewGallery = document.querySelector('#review-gallery');
const activityPanel = document.querySelector('#activity-panel');
const activityTitle = document.querySelector('#activity-title');
const activityList = document.querySelector('#activity-list');
const usagePill = document.querySelector('#usage-pill');
const attachmentPreview = document.querySelector('#attachment-preview');
const modelActions = document.querySelector('#model-actions');
const appConfig = JSON.parse(document.querySelector('#app-config')?.textContent || '{}');
const showInfoMessages = appConfig.showInfoMessages ?? true;
const currentProject = appConfig.projectName || '';
const viewer = new CadViewer(document.querySelector('#viewer'), document.querySelector('#dimensions'), appConfig);

let selectedFiles = [];
let previewProject = '';
let loadedPreviewRevision = '';
let previewLoadPromise = null;
let renderImageUrl = '';
let renderLoadSequence = 0;
let reviewLoadSequence = 0;
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
  const target = options.target || feed;
  const item = document.createElement('div');
  item.className = `message ${type}`;
  item.dataset.raw = text;
  if (options.messageId) item.dataset.messageId = options.messageId;
  if (type === 'agent') {
    item.innerHTML = `
      <div class="message-meta">
        <span class="agent-mark">AI</span>
        <span class="message-author">Agent</span>
        <span class="message-state">${options.streaming ? 'Responding' : ''}</span>
      </div>
      <div class="message-content"></div>
    `;
    target.appendChild(item);
    renderAgentContent(item, text || '');
    return item;
  }
  if (type === 'user') {
    item.classList.add('user-message');
    if (Array.isArray(options.images) && options.images.length) {
      const strip = document.createElement('div');
      strip.className = 'message-attachments';
      for (const img of options.images) {
        const el = document.createElement('img');
        el.className = 'attachment-thumb';
        el.src = img.src;
        el.alt = img.alt || 'attachment';
        el.loading = 'lazy';
        strip.appendChild(el);
      }
      item.appendChild(strip);
    }
    const textNode = document.createElement('div');
    textNode.className = 'message-text';
    textNode.textContent = text || '';
    item.appendChild(textNode);
    target.appendChild(item);
    return item;
  }
  // ``error`` and other bare-message types keep the simple text rendering path.
  item.textContent = text || '';
  target.appendChild(item);
  return item;
}

const activityLabels = {
  // Tool schema names published by the backend (agent/tools/tool_schemas.py).
  cad_build_and_verify: 'Building model',
  write_file: 'Updating model',
  edit_file: 'Updating model',
  read_file: 'Reading model',
  question: 'Requesting input',
  // Phase labels surfaced via ``tool_status`` from the new split tools. The
  // agent no longer auto-reviews; both steps are opt-in, so we give each one
  // a distinct label that matches what the backend publishes.
  cad_screenshot: 'Re-rasterising views',
  cad_review: 'Reviewing build',
  rendering_subset: 'Re-rasterising subset',
  screenshot_auto: 'Rendering views for review',
  // Status / pseudo-tool labels (kept for SSE events and info rows).
  agent: 'Agent',
  preparing: 'Preparing',
  running: 'Running',
  rendering: 'Rendering',
  rendering_views: 'Rendering review views',
  reviewing: 'Reviewing',
  started: 'Started',
  completed: 'Completed',
  error: 'Error',
  stopped: 'Stopped',
  // Review verdicts surfaced by ``review_updated``. ``activityLabel`` falls
  // back to a generic sanitised string, but listing them explicitly keeps the
  // pill readable (``Pass`` / ``Fail`` / ``Inconclusive``).
  pass: 'Pass',
  fail: 'Fail',
  inconclusive: 'Inconclusive',
  limit_reached: 'Limit reached',
};

function activityLabel(value) {
  return activityLabels[value] || String(value || 'Activity').replace(/[_-]+/g, ' ');
}

function activityDetail(tool, status, value) {
  if (!value) return '';
  if (!['completed', 'error'].includes(status)) return String(value);
  let payload = value;
  if (typeof value === 'string') {
    try { payload = JSON.parse(value); } catch { return value; }
  }
  if (!payload || typeof payload !== 'object') return String(payload || '');
  if (payload.ok === false) return String(payload.error?.message || 'The step failed.');
  const data = payload.ok === true ? payload.data : payload;
  if (typeof data === 'string') return data;
  // ``cad_build_and_verify`` returns a compact ``summary`` string alongside
  // the structured payload. Prefer it over the legacy fixed message so the
  // UI accurately reflects whether a render/review was produced or skipped
  // (render=false iterations return a metrics-only summary).
  if (tool === 'cad_build_and_verify' && data && typeof data.summary === 'string' && data.summary.trim()) {
    return data.summary.trim();
  }
  if (tool === 'cad_build_and_verify') return 'Model built and review artifacts created.';
  if (data && typeof data === 'object') {
    const output = data.stdout || data.stderr || data.message || data.summary;
    if (typeof output === 'string' && output.trim()) return output.trim();
  }
  return status === 'error' ? 'The step failed.' : 'Completed.';
}

function clearActivity() {
  activityItems.clear();
  activityList.replaceChildren();
  activityPanel.hidden = true;
  resetUsagePill();
}

// Token usage pill lives next to the activity-panel summary. Each
// ``agent_usage`` SSE event overwrites the pill in place; the activity list
// itself never sees these events, so tool rows stay readable.
let lastUsage = null;

function formatTokenCount(value) {
  const n = Number(value);
  if (!Number.isFinite(n) || n < 0) return '—';
  if (n >= 10000) return `${Math.round(n / 1000)}k`;
  return String(n);
}

function updateUsagePill(data = {}) {
  const prompt = Number(data.prompt_tokens);
  const completion = Number(data.completion_tokens);
  const cached = Number(data.cached_tokens || 0);
  if (!usagePill) return;
  // Keep the most recent meaningful reading; the backend publishes several
  // usage events per turn (one per LLM call), and we want the latest one.
  if (Number.isFinite(prompt)) {
    lastUsage = { prompt, completion: Number.isFinite(completion) ? completion : null, cached };
  } else if (!lastUsage) {
    return;
  }
  const snapshot = lastUsage;
  const promptLabel = formatTokenCount(snapshot.prompt);
  const completionLabel = formatTokenCount(snapshot.completion);
  const cachedLabel = formatTokenCount(snapshot.cached);
  const cacheRatio = snapshot.prompt > 0 ? snapshot.cached / snapshot.prompt : 0;
  let cacheState = 'miss';
  if (cacheRatio >= 0.9) cacheState = 'hit';
  else if (cacheRatio > 0) cacheState = 'partial';
  usagePill.dataset.cache = cacheState;
  usagePill.textContent =
    `↑${promptLabel} ↓${completionLabel} · cache ${cachedLabel}`;
  usagePill.hidden = false;
}

function resetUsagePill() {
  lastUsage = null;
  if (!usagePill) return;
  usagePill.textContent = '';
  usagePill.hidden = true;
  delete usagePill.dataset.cache;
}

function updateActivitySummary() {
  const items = [...activityItems.values()];
  if (!items.length) {
    activityPanel.hidden = true;
    return;
  }
  const running = items.filter(item => ['preparing', 'running', 'started', 'reviewing', 'rendering_views', 'rendering_subset', 'screenshot_auto'].includes(item.status));
  const failed = items.filter(item => item.status === 'error');
  const current = running.at(-1);
  activityTitle.textContent = current
    ? `${activityLabel(current.tool)} · ${activityLabel(current.status)}`
    : failed.length
      ? `${failed.length} failed ${failed.length === 1 ? 'task' : 'tasks'}`
      : `${items.length} ${items.length === 1 ? 'step' : 'steps'} completed`;
  activityPanel.open = Boolean(current);
  activityPanel.hidden = false;
}

function markActivityRecovered() {
  // Final completion should clear any in-flight activity rows but must NOT
  // hide failed rows — the failure was real and the user needs to see it.
  for (const item of activityItems.values()) {
    const row = activityList.querySelector(`[data-call-id="${CSS.escape(item.callId)}"]`);
    if (['preparing', 'running', 'started', 'reviewing', 'rendering', 'rendering_views', 'rendering_subset', 'screenshot_auto'].includes(item.status)) {
      item.status = 'completed';
      if (row) {
        row.dataset.status = 'completed';
        row.querySelector('.activity-state').textContent = 'Completed';
      }
    }
  }
  updateActivitySummary();
}

function addToolMessage(data) {
  const callId = data.call_id || crypto.randomUUID();
  const status = data.status || 'running';
  const item = activityItems.get(callId) || {callId, tool: data.tool || 'agent'};
  item.tool = data.tool || item.tool;
  item.status = status;
  item.result = activityDetail(item.tool, status, data.result) || item.result || '';
  activityItems.set(callId, item);

  let row = activityList.querySelector(`[data-call-id="${CSS.escape(callId)}"]`);
  if (!row) {
    row = document.createElement('div');
    row.className = 'activity-item';
    row.dataset.callId = callId;
    activityList.appendChild(row);
  }
  row.dataset.status = status;
  row.hidden = false;
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
  if (sendToggle) {
    sendToggle.dataset.mode = active ? 'stop' : 'send';
    sendToggle.title = active ? 'Stop' : 'Send';
    sendToggle.setAttribute('aria-label', active ? 'Stop' : 'Send');
    sendToggle.disabled = false;
    if (sendIcon) sendIcon.hidden = active;
    if (stopIcon) stopIcon.hidden = !active;
  }
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
      const meta = await api(`/api/projects/${encodeURIComponent(project)}/preview/meta`);
      if (!meta.displayable) {
        hideUnapprovedPreview(meta.review_status);
        return;
      }
      const url = `/api/projects/${encodeURIComponent(project)}/preview?ts=${Date.now()}`;
      await viewer.load(url);
      previewProject = project;
      loadedPreviewRevision = meta.revision || loadedPreviewRevision;
      // The agent turn has already completed by the time we render; failures
      // here are surfaced as a chat-side error and never block the canonical
      // ``agent_message`` from being persisted.
      await refreshRender();
      modelActions.hidden = false;
      loadReviewGallery();
    } catch (error) {
      addMessage(`Preview failed: ${error.message}`, 'error');
    } finally {
      previewLoadPromise = null;
    }
  })();
  return previewLoadPromise;
}

function hideUnapprovedPreview(reviewStatus = 'pending') {
  previewProject = '';
  loadedPreviewRevision = '';
  modelActions.hidden = true;
  viewer.clear(
    reviewStatus === 'fail'
      ? 'Preview was rejected by review'
      : 'Preview is awaiting review',
  );
}

async function refreshRender() {
  if (!currentProject) return;
  const sequence = ++renderLoadSequence;
  try {
    const res = await fetch(`/api/projects/${encodeURIComponent(currentProject)}/render?ts=${Date.now()}`);
    if (sequence !== renderLoadSequence) return;
    if (!res.ok) {
      clearRenderImage();
      renderSection.hidden = true;
      return;
    }
    const blob = await res.blob();
    if (sequence !== renderLoadSequence) return;
    const url = URL.createObjectURL(blob);
    const previousUrl = renderImageUrl;
    renderImageUrl = url;
    renderImage.src = url;
    if (previousUrl) URL.revokeObjectURL(previousUrl);
    renderSection.hidden = false;
  } catch {
    if (sequence !== renderLoadSequence) return;
    clearRenderImage();
    renderSection.hidden = true;
  }
}

function clearRenderImage() {
  if (renderImageUrl) URL.revokeObjectURL(renderImageUrl);
  renderImageUrl = '';
  renderImage.removeAttribute('src');
}

async function loadReviewGallery() {
  if (!currentProject) return;
  const sequence = ++reviewLoadSequence;
  try {
    const manifest = await api(`/api/projects/${encodeURIComponent(currentProject)}/review/manifest?ts=${Date.now()}`);
    if (sequence !== reviewLoadSequence) return;
    if (!manifest || !Array.isArray(manifest.views)) {
      reviewSection.hidden = true;
      return;
    }
    reviewSection.hidden = false;
    const result = manifest.result && typeof manifest.result === 'object' ? manifest.result : null;
    const status = result && typeof result.status === 'string' ? result.status : 'pending';
    const summary = result && typeof result.summary === 'string' ? result.summary : '';
    const findings = result && Array.isArray(result.findings) ? result.findings : [];
    reviewStatus.textContent = summary || statusLabel(status);
    reviewStatus.dataset.state = status;
    reviewSummary.replaceChildren();
    if (findings.length) {
      reviewSummary.hidden = false;
      for (const finding of findings) {
        const line = document.createElement('div');
        line.className = 'finding';
        const severity = finding && typeof finding.severity === 'string' ? finding.severity : 'minor';
        const message = finding && typeof finding.message === 'string' ? finding.message : '';
        line.dataset.severity = severity;
        line.textContent = message;
        reviewSummary.appendChild(line);
      }
    } else {
      reviewSummary.hidden = true;
    }
    reviewGallery.replaceChildren();
    for (const view of manifest.views) {
      if (!view || typeof view.view_id !== 'string') continue;
      const figure = document.createElement('figure');
      const img = document.createElement('img');
      img.src = `/api/projects/${encodeURIComponent(currentProject)}/review/view/${encodeURIComponent(view.view_id)}?ts=${Date.now()}`;
      img.alt = view.label || view.view_id;
      img.loading = 'lazy';
      const caption = document.createElement('figcaption');
      caption.textContent = view.label || view.view_id;
      figure.append(img, caption);
      reviewGallery.appendChild(figure);
    }
  } catch {
    if (sequence !== reviewLoadSequence) return;
    reviewSection.hidden = true;
  }
}

function statusLabel(status) {
  if (status === 'pass') return 'Pass';
  if (status === 'fail') return 'Fail';
  if (status === 'inconclusive') return 'Inconclusive';
  return 'Pending';
}

async function syncCurrentPreview() {
  if (!currentProject || previewLoadPromise) return;
  try {
    const meta = await api(`/api/projects/${encodeURIComponent(currentProject)}/preview/meta`);
    if (
      meta.available
      && (previewProject !== currentProject || loadedPreviewRevision !== meta.revision)
    ) {
      if (meta.displayable) await loadCurrentPreview();
      else hideUnapprovedPreview(meta.review_status);
      loadReviewGallery();
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
  }
}

// Extract readable text from a History entry's `content` field. The
// backend persists image attachments as structured parts (text + image_url),
// and the History endpoint redacts the image data, leaving a
// `[Reference image N]` text placeholder for each part.
function normalizeHistoryContent(content) {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const lines = [];
  let imageIndex = 0;
  for (const part of content) {
    if (!part || typeof part !== 'object') continue;
    if (part.type === 'text' && typeof part.text === 'string') {
      lines.push(part.text);
    } else if (part.type === 'image_url') {
      imageIndex += 1;
      lines.push(`[Reference image ${imageIndex}]`);
    }
  }
  return lines.join('\n');
}

async function loadHistory(projectName, options = {}) {
  const target = options.target === 'drawer' ? historyContent : feed;
  const intoDrawer = options.target === 'drawer';
  try {
    const data = await api(`/api/projects/${encodeURIComponent(projectName)}/history`);
    if (intoDrawer) {
      // Drawer mode must NEVER mutate the main chat feed.
      target.replaceChildren();
    } else {
      clearActivity();
      target.replaceChildren();
      questionArea.replaceChildren();
    }
    let renderedAny = false;
    for (const evt of data.events) {
      const role = evt.role || '';
      const raw = evt.content;
      const text = role === 'user' ? normalizeHistoryContent(raw) : (raw || '');
      if (role === 'user') {
        addMessage(text, 'user', {target});
        renderedAny = true;
      } else if (role === 'assistant' || role === 'agent') {
        if (evt.tool_calls || !String(text).trim()) continue;
        addMessage(text, 'agent', {target});
        renderedAny = true;
      } else if (evt.type === 'agent_error') {
        addMessage(evt.data?.message || text, 'error', {target});
        renderedAny = true;
      } else if (!intoDrawer && showInfoMessages) {
        addInfoMessage(evt.type, evt.data);
      }
    }
    if (!renderedAny && !intoDrawer) {
      // Project with no conversation → restore the empty state from the
      // template (preserves the project name and example prompts).
      const emptyTpl = document.querySelector('#chat-empty');
      if (emptyTpl) target.appendChild(emptyTpl.content.cloneNode(true));
      else {
        const empty = document.createElement('div');
        empty.className = 'empty-state';
        empty.textContent = `Project "${projectName}" selected. Describe a part to begin.`;
        target.appendChild(empty);
      }
    }
  } catch (error) {
    addMessage(error.message, 'error', {target});
  }
}

function addInfoMessage(type, data = {}) {
  if (type === 'agent_status') {
    addToolMessage({
      call_id: 'agent-run',
      tool: 'agent',
      status: data.status || 'info',
      result: data.message,
    });
  } else if (type === 'tool_status') {
    addToolMessage(data);
  } else if (type === 'agent_usage') {
    updateUsagePill(data);
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
  if (sendToggle) sendToggle.disabled = true;
  message.disabled = true;
  // Snapshot attached files and their preview URLs so we can render the
  // images inline in the chat feed before sending them to the server.
  const imagePayload = selectedFiles.map((file) => ({
    src: URL.createObjectURL(file),
    alt: file.name,
  }));
  try {
    clearActivity();
    addMessage(text, 'user', {images: imagePayload});
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
    if (sendToggle) sendToggle.disabled = false;
    message.disabled = false;
    message.focus();
  }
});

sendToggle?.addEventListener('click', async () => {
  // Single button toggles between Send and Stop based on agent state.
  if (sendToggle.dataset.mode === 'stop') {
    if (!currentProject) return;
    sendToggle.disabled = true;
    try {
      await api('/api/stop', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({project: currentProject}),
      });
    } catch (error) {
      addMessage(error.message, 'error');
    } finally {
      sendToggle.disabled = false;
    }
    return;
  }
  chatForm.dispatchEvent(new Event('submit'));
});

resetButton.addEventListener('click', async () => {
  if (!currentProject) return;
  const proceed = window.confirm(
    'Reset the AI\u2019s conversation memory for this project?\n\nThe model, preview, and revisions are kept. The next message starts a fresh context.'
  );
  if (!proceed) return;
  resetButton.disabled = true;
  try {
    await api(`/api/projects/${encodeURIComponent(currentProject)}/reset`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
    });
  } catch (error) {
    addMessage(error.message, 'error');
  } finally {
    resetButton.disabled = false;
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
  loadHistory(currentProject, {target: 'drawer'});
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
      fields.push({
        id,
        label: q.question || id,
        options: q.options || [],
        type: q.input_type || q.type || 'text',
        required: q.required !== false,
      });
    });
  } else if (question.question) {
    fields.push({ id: 'answer', label: question.question, options: question.options || [], type: 'text' });
  }
  for (const field of fields) {
    const label = document.createElement('label');
    label.textContent = field.label;
    let input;
    if (field.options?.length && field.type !== 'multiselect') {
      input = document.createElement('select');
      input.required = true;
      for (const opt of field.options) {
        const optionEl = document.createElement('option');
        optionEl.value = opt.value || opt;
        optionEl.textContent = opt.label || opt;
        input.appendChild(optionEl);
      }
    } else if (field.type === 'number') {
      input = document.createElement('input');
      input.type = 'number';
    } else if (field.type === 'multiselect') {
      input = document.createElement('select');
      input.multiple = true;
      for (const opt of field.options) {
        const optionEl = document.createElement('option');
        optionEl.value = opt;
        optionEl.textContent = opt;
        input.appendChild(optionEl);
      }
    } else if (field.type === 'textarea') {
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
      if (el) {
        answers[field.id] = field.type === 'multiselect'
          ? Array.from(el.selectedOptions).map(option => option.value)
          : el.value;
      }
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
// Cards currently being filled from streaming deltas. Keyed by message_id so
// multiple in-flight turns (rare in practice, but the API supports them) do
// not collide. Cards are removed once agent_message finalizes them.
const streamingMessages = new Map();
// The streaming card that has not yet been finalized; used by agent_message
// to update the canonical final text without duplicating the card.
let pendingFinalCard = null;

function startStreamingCard(messageId) {
  if (!messageId || streamingMessages.has(messageId)) return streamingMessages.get(messageId);
  const item = document.createElement('div');
  item.className = 'message agent';
  item.dataset.raw = '';
  item.dataset.messageId = messageId;
  item.innerHTML = `
    <div class="message-meta">
      <span class="agent-mark">AI</span>
      <span class="message-author">Agent</span>
      <span class="message-state">Responding</span>
    </div>
    <div class="message-content"></div>
  `;
  feed.appendChild(item);
  streamingMessages.set(messageId, item);
  pendingFinalCard = item;
  return item;
}

function appendStreamingDelta(messageId, delta) {
  if (!messageId || !delta) return;
  let card = streamingMessages.get(messageId);
  if (!card) card = startStreamingCard(messageId);
  card.dataset.raw = (card.dataset.raw || '') + delta;
  renderAgentContent(card, card.dataset.raw);
}

function finalizeStreamingCard(messageId, finalText) {
  if (messageId) {
    const card = streamingMessages.get(messageId);
    if (card) {
      renderAgentContent(card, finalText || '');
      card.dataset.raw = finalText || '';
      const state = card.querySelector('.message-state');
      if (state) state.textContent = '';
      streamingMessages.delete(messageId);
      // Keep pendingFinalCard pointing at this card so the subsequent
      // agent_message event can replace its content with the canonical
      // final message without creating a duplicate card.
      return card;
    }
  }
  // No streaming card — happens when content arrived before stream_start
  // (or the user reloaded mid-run). Create a new card with the canonical text.
  const item = addMessage(finalText || '', 'agent');
  pendingFinalCard = item;
  return item;
}

function discardEmptyStreamingCard(messageId) {
  if (!messageId) return;
  const card = streamingMessages.get(messageId);
  if (!card) return;
  streamingMessages.delete(messageId);
  if (pendingFinalCard === card) pendingFinalCard = null;
  if (!card.dataset.raw) card.remove();
}

async function syncAfterStreamReset() {
  // Re-fetch the canonical state so events missed during the gap are visible.
  // Re-rendering the main feed replaces persisted content rather than
  // appending, which prevents duplicate messages on reconnect.
  if (!currentProject) return;
  await loadHistory(currentProject);
  await loadCurrentState();
  await syncCurrentPreview();
}

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
        call_id: 'agent-run',
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
    question: data => {
      if (data.project !== currentProject) return;
      showQuestion(data);
    },
    agent_content_delta: data => {
      if (data.project !== currentProject) return;
      // Backend publishes agent_content_delta for content chunks. The delta
      // payload may carry the chunk under `content` or `delta`; accept either.
      const chunk = data.content ?? data.delta ?? '';
      if (chunk) appendStreamingDelta(data.message_id, chunk);
    },
    agent_stream_end: data => {
      if (data.project !== currentProject) return;
      const messageId = data.message_id;
      const finalText = data.message || '';
      if (!finalText.trim()) {
        // Tool-only turn — drop any empty placeholder card so the chat does
        // not show empty assistant messages.
        discardEmptyStreamingCard(messageId);
        return;
      }
      finalizeStreamingCard(messageId, finalText);
    },
    agent_message: data => {
      if (data.project !== currentProject) return;
      const finalText = data.message || '';
      if (!finalText.trim()) {
        // No user-visible content (e.g. transient terminal notification).
        setThinking(false);
        return;
      }
      // Prefer the active streaming card so the final message replaces the
      // streamed text instead of appending a duplicate card. If there is no
      // streaming card (e.g. events were missed over a reset, or the
      // backend published a non-streamed final answer), fall back to a new
      // card.
      if (pendingFinalCard) {
        renderAgentContent(pendingFinalCard, finalText);
        pendingFinalCard.dataset.raw = finalText;
        const state = pendingFinalCard.querySelector('.message-state');
        if (state) state.textContent = '';
        for (const [key, value] of streamingMessages) {
          if (value === pendingFinalCard) streamingMessages.delete(key);
        }
        pendingFinalCard = null;
      } else {
        addMessage(finalText, 'agent');
      }
      markActivityRecovered();
      // Defensive: every terminal turn publishes agent_message, so the
      // thinking indicator must clear here even if the matching
      // agent_status event was missed (or never published for this path).
      setThinking(false);
    },
    tool_status: data => {
      if (data.project !== currentProject) return;
      addToolMessage(data);
    },
    preview_updated: data => {
      if (data.project !== currentProject) return;
      if (data.preview_id) loadCurrentPreview(data.preview_id);
      else hideUnapprovedPreview('pending');
    },
    revision_updated: data => {
      if (data.project !== currentProject) return;
      syncCurrentPreview();
    },
    // Phase-complete events from the split cad_screenshot / cad_review tools.
    // The intermediate ``tool_status`` events already drive the running rows;
    // these terminal events refresh the activity pill with the final verdict
    // (e.g. ``cad_review`` → "Reviewing build · Pass") and re-sync the preview
    // when the screenshot tool produced a new artifact cache tier.
    screenshot_updated: data => {
      if (data.project !== currentProject) return;
      if (data.cache_hit) {
        activityTitle.textContent = 'Re-rasterising views · Cache hit';
        activityPanel.hidden = false;
      }
    },
    review_updated: data => {
      if (data.project !== currentProject) return;
      const verdict = activityLabel(data.status || 'inconclusive');
      const summary = typeof data.summary === 'string' && data.summary.trim()
        ? ` — ${data.summary.trim().slice(0, 160)}`
        : '';
      activityTitle.textContent = `Reviewing build · ${verdict}${summary}`;
      activityPanel.hidden = false;
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
      // Errors are terminal; ensure the thinking indicator clears.
      setThinking(false);
    } catch {}
  });
  eventSource.addEventListener('agent_stopped', event => {
    try {
      const data = JSON.parse(event.data);
      if (data.project !== currentProject) return;
      setThinking(false);
    } catch {}
  });
  eventSource.addEventListener('stream_reset', () => {
    // The backend disconnected an overflowed subscriber and signalled that
    // some events were dropped. Re-fetch the canonical state so the UI
    // converges to the persisted conversation, project state, and preview.
    syncAfterStreamReset().catch(err => console.error('stream_reset sync failed', err));
  });

  eventSource.addEventListener('conversation_reset', event => {
    // The backend cleared this project's conversation.jsonl. The model and
    // preview are unchanged, but the chat feed (and the history drawer if
    // open) must drop everything so the next turn starts on an empty canvas.
    try {
      const data = JSON.parse(event.data);
      if (!data || data.project !== currentProject) return;
      applyConversationReset();
    } catch {}
  });
}

function applyConversationReset() {
  // Wipe the main feed, dismiss any pending question, and clear activity so
  // the UI matches the freshly truncated conversation.jsonl. The drawer is
  // refreshed lazily the next time it is opened.
  clearActivity();
  feed.replaceChildren();
  questionArea.replaceChildren();
  const emptyTpl = document.querySelector('#chat-empty');
  if (emptyTpl) feed.appendChild(emptyTpl.content.cloneNode(true));
  else {
    const empty = document.createElement('div');
    empty.className = 'empty-state';
    empty.textContent = `Project "${currentProject}" selected. Describe a part to begin.`;
    feed.appendChild(empty);
  }
  historyContent.replaceChildren();
  setThinking(false);
}

// Populate the chat input with the text of an example-prompt button. Uses
// event delegation so it works whether the empty-state is rendered inline
// (initial page load) or re-cloned into the feed by loadHistory().
document.addEventListener('click', event => {
  const target = event.target.closest('.example-prompt');
  if (!target) return;
  message.value = target.dataset.prompt || message.value;
  message.focus();
});

(async function init() {
  if (!currentProject) return;
  await loadCurrentState();
  // Load the persisted conversation into the main feed so reopening a
  // project immediately shows its history. The empty-state element is
  // removed automatically once any displayable message is rendered.
  await loadHistory(currentProject);
  await syncCurrentPreview();
  connectStream();
})();
