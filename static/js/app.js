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
const downloadModelBtn = document.querySelector('#download-model');
const exportFormatSelect = document.querySelector('#export-format');
const appConfig = JSON.parse(document.querySelector('#app-config')?.textContent || '{}');
const showInfoMessages = appConfig.showInfoMessages ?? true;
const currentProject = appConfig.projectName || '';
const viewer = new CadViewer(document.querySelector('#viewer'), document.querySelector('#dimensions'), appConfig);

try {
  const savedWidth = parseInt(localStorage.getItem('cad_chat_width'), 10);
  if (Number.isFinite(savedWidth) && savedWidth >= 360 && savedWidth <= window.innerWidth - 400) {
    document.documentElement.style.setProperty('--chat-width', `${savedWidth}px`);
  }
} catch {}

function generateUUID() {
  return typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function'
    ? crypto.randomUUID()
    : `${Date.now()}-${Math.random().toString(36).slice(2, 11)}`;
}

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
      if (tag === 'a' && el.getAttribute('href')?.startsWith('http')) {
        el.setAttribute('target', '_blank');
        el.setAttribute('rel', 'noopener noreferrer');
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

function stripToolCallTags(text, { trim = true } = {}) {
  if (!text || typeof text !== 'string') return '';
  let cleaned = text;
  if (cleaned.includes('<function=')) {
    cleaned = cleaned.replace(/<function=[a-zA-Z0-9_-]+>.*?(?:<\/function>|(?=<function=)|$)/gs, '');
  }
  if (cleaned.includes('<tool_call')) {
    cleaned = cleaned.replace(/<tool_call[^>]*>.*?(?:<\/tool_call>|$)/gs, '');
  }
  if (cleaned.includes('[TOOL_CALLS]')) {
    cleaned = cleaned.replace(/\[TOOL_CALLS\].*?(?:\[\/TOOL_CALLS\]|$)/gs, '');
  }
  cleaned = cleaned.replace(/<\/?(?:function|parameter|tool_call)[^>]*>/g, '');
  return trim ? cleaned.trim() : cleaned;
}

function renderAgentContent(item, text) {
  const clean = stripToolCallTags(text || '', { trim: false });
  const contentEl = item.querySelector('.message-content');
  if (!contentEl) return;
  if (clean.trim()) {
    contentEl.innerHTML = sanitizeHTML(marked.parse(clean));
  } else if (item.dataset.state && item.dataset.state !== 'done' && !(item.dataset.reasoning && item.dataset.state === 'thinking')) {
    if (!contentEl.querySelector('.message-skeleton')) {
      contentEl.innerHTML = `
        <div class="message-skeleton" aria-label="Waiting for response">
          <div class="skeleton-line" style="width: 88%;"></div>
          <div class="skeleton-line" style="width: 74%;"></div>
          <div class="skeleton-line" style="width: 52%;"></div>
        </div>
      `;
    }
  } else {
    contentEl.innerHTML = '';
  }
}

function stripEncryptedReasoning(text) {
  if (typeof text !== 'string' || !text) return '';
  const s = text.trim();
  if (s.length >= 50 && /^[A-Za-z0-9+/=]+$/.test(s)) return '';
  return text.replace(/(?:\r?\n|\s)+(?:[A-Za-z0-9+/=]{40,}(?:\r?\n|\s)*)+$/g, '').trim();
}

function createThoughtDisclosure(reasoningText = '', isOpen = false, isLive = false) {
  const details = document.createElement('details');
  details.className = 'thought-disclosure';
  if (isOpen) details.open = true;

  const summary = document.createElement('summary');
  summary.className = 'thought-summary';
  summary.innerHTML = `
    <span class="thought-title">Thought Process</span>
    <span class="thought-chevron">▼</span>
  `;

  const content = document.createElement('div');
  content.className = 'thought-content';
  content.textContent = reasoningText;

  details.appendChild(summary);
  details.appendChild(content);
  return details;
}

function ensureThoughtDisclosure(card, reasoning, isOpen = false, isLive = false) {
  if (reasoning === null || reasoning === undefined) return null;
  const cleanReasoning = stripEncryptedReasoning(String(reasoning));
  if (!cleanReasoning) return null;
  if (!isLive && card.dataset.raw && card.dataset.raw.trim() === cleanReasoning) {
    const existing = card.querySelector('.thought-disclosure');
    if (existing) existing.remove();
    return null;
  }
  let disclosure = card.querySelector('.thought-disclosure');
  if (!disclosure) {
    disclosure = createThoughtDisclosure(cleanReasoning, isOpen, isLive);
    const contentEl = card.querySelector('.message-content');
    card.insertBefore(disclosure, contentEl);
  } else {
    if (isOpen || isLive) {
      disclosure.open = true;
    }
    const content = disclosure.querySelector('.thought-content');
    if (content && content.textContent !== cleanReasoning) {
      content.textContent = cleanReasoning;
    }
  }
  return disclosure;
}

function addMessage(text, type = 'agent', options = {}) {
  const target = options.target || feed;
  const empty = target.querySelector('.empty-state');
  if (empty) empty.remove();
  const item = document.createElement('div');
  item.className = `message ${type}`;
  item.dataset.raw = text;
  if (options.messageId) item.dataset.messageId = options.messageId;
  if (type === 'agent') {
    const isStreaming = Boolean(options.streaming);
    if (isStreaming) {
      item.classList.add('streaming', 'thinking');
    }
    item.innerHTML = `
      <div class="message-meta">
        <span class="agent-mark">AI</span>
        <span class="message-author">Agent</span>
        <span class="message-state ${isStreaming ? 'state-thinking' : ''}">${isStreaming ? 'Thinking' : ''}</span>
      </div>
      <div class="message-content"></div>
    `;
    ensureThoughtDisclosure(item, options.reasoning, false, false);
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
        if (img.src && img.src.startsWith('blob:')) {
          const revoke = () => URL.revokeObjectURL(img.src);
          el.addEventListener('load', revoke, { once: true });
          el.addEventListener('error', revoke, { once: true });
        }
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
  cad_build: 'Building model',
  cad_build_and_verify: 'Building model',
  get_view_images: 'Retrieving images',
  write_file: 'Updating model',
  edit_file: 'Updating model',
  read_file: 'Reading model',
  question: 'Requesting input',
  // Phase labels surfaced via ``tool_status`` from the new split tools. The
  // agent no longer auto-reviews; both steps are opt-in, so we give each one
  // a distinct label that matches what the backend publishes.
  cad_screenshot: 'Re-rasterising views',
  cad_review: 'Reviewing build',
  rendering_subset: 'Re-rasterising a few views',
  screenshot_auto: 'Rendering views for review',
  // Status / pseudo-tool labels (kept for SSE events and info rows). Plain
  // English so the activity pill stays readable while a run is in progress.
  agent: 'Agent',
  preparing: 'Preparing',
  running: 'In progress',
  rendering: 'Rendering',
  rendering_views: 'Rendering review views',
  verifying: 'Verifying geometry',
  reviewing: 'Reviewing',
  started: 'Starting',
  completed: 'Completed',
  error: 'Error',
  stopped: 'Stopped',
  // Review verdicts surfaced by ``review_updated``. ``activityLabel`` falls
  // back to a generic sanitised string, but listing them explicitly keeps the
  // pill readable. The verdict is also promoted onto the matching row's
  // state label via ``refreshLatestToolRow`` so the drawer reflects the
  // outcome without expanding the title pill.
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
  // ``cad_build`` / ``cad_build_and_verify`` returns a compact ``summary`` string alongside
  // the structured payload. Prefer it over the legacy fixed message so the
  // UI accurately reflects whether a render/review was produced or skipped
  // (render=false iterations return a metrics-only summary).
  if ((tool === 'cad_build' || tool === 'cad_build_and_verify') && data && typeof data.summary === 'string' && data.summary.trim()) {
    return data.summary.trim();
  }
  if (tool === 'cad_build' || tool === 'cad_build_and_verify') return 'Model built and review artifacts created.';
  if (tool === 'get_view_images' && data && Array.isArray(data.images)) {
    return `${data.images.length} view(s) retrieved.`;
  }
  if (data && typeof data === 'object') {
    const output = data.stdout || data.stderr || data.message || data.summary;
    if (typeof output === 'string' && output.trim()) return output.trim();
  }
  return status === 'error' ? 'The step failed.' : 'Completed.';
}

let userManuallyToggled = false;
activityPanel?.querySelector('summary')?.addEventListener('click', () => {
  userManuallyToggled = true;
});

function clearActivity() {
  userManuallyToggled = false;
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
  if (!userManuallyToggled) {
    activityPanel.open = Boolean(current);
  }
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
  const callId = data.call_id || generateUUID();
  const status = data.status || 'running';
  if (callId !== 'agent-run' && activityItems.has('agent-run')) {
    const placeholder = activityList.querySelector('[data-call-id="agent-run"]');
    placeholder?.remove();
    activityItems.delete('agent-run');
  }

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
  // ``screenshot_updated`` / ``review_updated`` events arrive between
  // ``tool_status:running`` and ``tool_status:completed`` and set a cosmetic
  // verdict label (``Cache hit``, ``Pass`` …) via ``refreshLatestToolRow``.
  // That label must survive the dispatcher's ``tool_status:completed``
  // rebuild — otherwise the activity row and the title pill flash the
  // verdict and then snap back to ``Completed`` / ``X step(s) completed``
  // before the user can read them. Honour ``dataset.decorativeState`` only
  // on the completion transition; intermediate rebuilds drop a stale value
  // so the loop's state machine stays the source of truth.
  const cosmeticState = status === 'completed' ? row.dataset.decorativeState : null;
  if (!cosmeticState) {
    delete row.dataset.decorativeState;
  }
  row.replaceChildren();
  const name = document.createElement('span');
  name.className = 'activity-name';
  name.textContent = activityLabel(item.tool);
  const state = document.createElement('span');
  state.className = 'activity-state';
  state.textContent = cosmeticState || activityLabel(status);
  row.append(name, state);
  if (item.result) {
    const detail = document.createElement('span');
    detail.className = 'activity-detail';
    detail.textContent = item.result.length > 180 ? `${item.result.slice(0, 177)}…` : item.result;
    row.appendChild(detail);
  }
  // Suppress the title pill reset when a cosmetic label is showing so the
  // verdict text (``Re-rasterising views · Cache hit`` /
  // ``Reviewing build · Pass``) stays visible until the next tool call
  // takes over the panel.
  if (!cosmeticState) {
    updateActivitySummary();
  }
  return row;
}

function refreshLatestToolRow(tool, updates = {}) {
  // Find the most recent row for a tool name and update its visible state.
  // ``screenshot_updated`` / ``review_updated`` events carry verdict/cache
  // info that arrives after the matching ``tool_status:completed`` has
  // already finalised the row, so the dispatcher-side activity machine is
  // no longer running. The cosmetic label is stashed on the row's dataset
  // so ``addToolMessage`` can preserve it across the completion rebuild
  // that the dispatcher will publish immediately afterwards.
  const rows = activityList.querySelectorAll('.activity-item');
  for (let index = rows.length - 1; index >= 0; index -= 1) {
    const row = rows[index];
    const item = activityItems.get(row.dataset.callId);
    if (!item || item.tool !== tool) continue;
    if (updates.state) {
      // Cosmetic label only — the underlying ``item.status`` stays as the
      // agent loop set it; only the visible label changes.
      const state = row.querySelector('.activity-state');
      if (state) state.textContent = updates.state;
      row.dataset.decorativeState = updates.state;
    }
    if (updates.status) {
      item.status = updates.status;
      row.dataset.status = updates.status;
      const state = row.querySelector('.activity-state');
      if (state) state.textContent = activityLabel(updates.status);
      // A full ``updates.status`` supersedes the cosmetic override; the
      // loop is taking over again, so clear the stash.
      delete row.dataset.decorativeState;
      updateActivitySummary();
    }
    if (Object.prototype.hasOwnProperty.call(updates, 'result')) {
      item.result = updates.result;
      let detail = row.querySelector('.activity-detail');
      if (!detail) {
        detail = document.createElement('span');
        detail.className = 'activity-detail';
        row.appendChild(detail);
      }
      detail.textContent = updates.result
        ? (updates.result.length > 180 ? `${updates.result.slice(0, 177)}…` : updates.result)
        : '';
      if (!updates.result) detail.remove();
    }
    return row;
  }
  return null;
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
    if (feed && !getActiveCard()) {
      getOrCreateRunCard(null, { optimistic: true });
    }
  } else {
    document.querySelectorAll('.thinking-indicator').forEach(el => el.remove());
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
      if (downloadModelBtn) downloadModelBtn.disabled = false;
      loadReviewGallery();
    } catch (error) {
      if (error.message?.includes('superseded')) return;
      if (downloadModelBtn) downloadModelBtn.disabled = true;
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
  if (downloadModelBtn) downloadModelBtn.disabled = true;
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
    setThinking(false);
    const q = data.question || {};
    if (!questionArea.querySelector('.question-form')) {
      showQuestion({project: currentProject, ...q});
    }
  } else {
    if (questionArea.querySelector('.question-form')) {
      clearQuestionArea();
    }
    if (data.status === 'running') {
      setThinking(true);
    } else if (data.status === 'idle') {
      setThinking(false);
      const agentRunItem = activityItems.get('agent-run');
      if (agentRunItem && (agentRunItem.status === 'started' || agentRunItem.status === 'running')) {
        addToolMessage({
          call_id: 'agent-run',
          tool: 'agent',
          status: 'stopped',
          result: 'Agent is idle.',
        });
      }
    }
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
      clearQuestionArea();
    }
    let renderedAny = false;
    let accumulatedReasoning = '';
    for (const evt of data.events) {
      const role = evt.role || '';
      const raw = evt.content;
      const text = normalizeHistoryContent(raw);
      if (role === 'user') {
        accumulatedReasoning = '';
        addMessage(text, 'user', {target});
        renderedAny = true;
      } else if (role === 'assistant' || role === 'agent') {
        let reasoning = evt.reasoning || '';
        let displayText = text;
        if (!reasoning && Array.isArray(evt.reasoning_details)) {
          reasoning = evt.reasoning_details
            .map(d => {
              if (!d || typeof d !== 'object' || d.type === 'reasoning.encrypted') return '';
              const val = d.text || d.summary;
              return typeof val === 'string' ? val : '';
            })
            .join('');
        }
        reasoning = stripEncryptedReasoning(reasoning);
        if (!reasoning && typeof displayText === 'string' && displayText.includes('<think>')) {
          const blocks = [];
          displayText = displayText.replace(/<think>([\s\S]*?)<\/think>/g, (_, b) => {
            if (b.trim()) blocks.push(b.trim());
            return '';
          }).trim();
          if (displayText.includes('<think>')) {
            const startIdx = displayText.indexOf('<think>');
            const unclosed = displayText.slice(startIdx + 7).trim();
            if (unclosed) blocks.push(unclosed);
            displayText = displayText.slice(0, startIdx).trim();
          }
          if (displayText.includes('</think>')) {
            displayText = displayText.replace(/<\/think>/g, '').trim();
          }
          if (blocks.length > 0) {
            reasoning = blocks.join('\n\n');
          }
        }
        if (reasoning) {
          accumulatedReasoning = accumulatedReasoning
            ? `${accumulatedReasoning}\n\n${reasoning}`
            : reasoning;
        }
        displayText = stripToolCallTags(displayText);
        if (!String(displayText).trim()) {
          // Tool-only intermediate turn: preserve accumulated reasoning for final response
          continue;
        }
        const cleanReasoning = (String(displayText).trim() === accumulatedReasoning.trim()) ? '' : accumulatedReasoning;
        addMessage(displayText, 'agent', {target, reasoning: cleanReasoning});
        accumulatedReasoning = '';
        renderedAny = true;
      } else if (evt.type === 'agent_error') {
        addMessage(evt.data?.message || text, 'error', {target});
        renderedAny = true;
      } else if (!intoDrawer && showInfoMessages) {
        addInfoMessage(evt.type, evt.data);
      }
    }
    if (accumulatedReasoning) {
      addMessage('', 'agent', {target, reasoning: accumulatedReasoning});
      renderedAny = true;
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
      call_id: 'agent-run',
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
    finalizeAllActiveCards();
    addMessage(text, 'user', {images: imagePayload});
    getOrCreateRunCard(null, { optimistic: true });
    setThinking(true);
    scrollFeedToBottom(true);
    message.value = '';
    const idempotencyKey = generateUUID();
    const body = new FormData();
    body.append('project', currentProject);
    body.append('message', text);
    body.append('idempotency_key', idempotencyKey);
    selectedFiles.forEach(file => body.append('attachments', file));
    const response = await api('/api/chat', {method: 'POST', body});
    if (response.duplicate) {
      if (optimisticCard) {
        optimisticCard.remove();
        optimisticCard = null;
      }
      setThinking(false);
      return;
    }
    if (response.attachments?.length) {
      addToolMessage({
        call_id: `attachments-${generateUUID()}`,
        tool: 'Images',
        status: 'completed',
        result: `${response.attachments.length} reference image(s) uploaded.`,
      });
    }
    clearAttachments();
  } catch (error) {
    if (optimisticCard) {
      optimisticCard.remove();
      optimisticCard = null;
    }
    addMessage(error.message, 'error');
    setThinking(false);
  } finally {
    if (!isQuestionPending) {
      if (sendToggle) sendToggle.disabled = false;
      message.disabled = false;
      message.focus();
    }
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
      setThinking(false);
      const activeCard = getActiveCard();
      if (activeCard) {
        setCardState(activeCard, 'stopped', 'Stopped');
      }
      finalizeAllActiveCards();
      addToolMessage({
        call_id: 'agent-run',
        tool: 'agent',
        status: 'stopped',
        result: 'Agent task stopped.',
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

const ALLOWED_IMAGE_MIMES = new Set(['image/png', 'image/jpeg', 'image/webp']);

function handleFiles(files) {
  const valid = files.filter(f => ALLOWED_IMAGE_MIMES.has(f.type?.toLowerCase()));
  if (files.length > valid.length) {
    addMessage('Only PNG, JPEG, and WEBP images are supported.', 'error');
  }
  if (!valid.length) return;
  if (selectedFiles.length + valid.length > 5) {
    addMessage('You can attach at most 5 reference images.', 'error');
    return;
  }
  selectedFiles = [...selectedFiles, ...valid];
  renderAttachmentPreview();
}

attachments.addEventListener('change', () => {
  handleFiles(Array.from(attachments.files || []));
  attachments.value = '';
});

['dragenter', 'dragover'].forEach(eventName => {
  dropzone.addEventListener(eventName, event => {
    event.preventDefault();
    dropzone.classList.add('is-dragging');
  });
});
dropzone.addEventListener('dragleave', event => {
  event.preventDefault();
  dropzone.classList.remove('is-dragging');
});
dropzone.addEventListener('drop', event => {
  event.preventDefault();
  dropzone.classList.remove('is-dragging');
  const files = Array.from(event.dataTransfer?.files || []);
  if (!files.length) return;
  handleFiles(files);
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
downloadModelBtn?.addEventListener('click', async () => {
  if (!currentProject || downloadModelBtn.disabled) return;
  const format = exportFormatSelect?.value || 'stl';
  const downloadUrl = `/api/projects/${encodeURIComponent(currentProject)}/export?format=${encodeURIComponent(format)}`;
  downloadModelBtn.disabled = true;
  try {
    const res = await fetch(downloadUrl);
    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: 'Export failed' }));
      addMessage(`Export failed: ${err.error || res.statusText}`, 'error');
      return;
    }
    const blob = await res.blob();
    const blobUrl = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = blobUrl;
    a.download = `${currentProject}.${format}`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(blobUrl), 10000);
  } catch (error) {
    addMessage(`Export failed: ${error.message}`, 'error');
  } finally {
    downloadModelBtn.disabled = Boolean(modelActions?.hidden);
  }
});
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
  let currentWidth = null;
  const onMove = moveEvent => {
    const width = Math.min(Math.max(moveEvent.clientX, 360), window.innerWidth - 400);
    currentWidth = width;
    document.documentElement.style.setProperty('--chat-width', `${width}px`);
  };
  const onEnd = () => {
    document.body.classList.remove('is-resizing');
    if (currentWidth) {
      try { localStorage.setItem('cad_chat_width', `${currentWidth}px`); } catch {}
    }
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
        restore.disabled = true;
        try {
          const res = await api(`/api/projects/${encodeURIComponent(currentProject)}/revisions/${rev.id}/restore`, {method: 'POST'});
          if (res && res.ok === false) {
            addMessage(`Restore partial failure: ${res.error}`, 'error');
          }
          historyDrawer.hidden = true;
          loadModelPane();
        } catch (error) {
          addMessage(error.message, 'error');
        } finally {
          restore.disabled = rev.is_active;
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
let isQuestionPending = false;

function setComposerBlockedForQuestion(blocked) {
  isQuestionPending = blocked;
  message.disabled = blocked;
  if (sendToggle) sendToggle.disabled = blocked;
  message.placeholder = blocked ? 'Please answer the pending question above…' : 'Describe the part, dimensions, fit, and required features…';
}

function clearQuestionArea() {
  questionArea.replaceChildren();
  setComposerBlockedForQuestion(false);
}

function showQuestion(question) {
  clearQuestionArea();
  setComposerBlockedForQuestion(true);
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
    fields.push({ id: 'answer', label: question.question, options: question.options || [], type: question.input_type || 'text' });
  }
  for (const field of fields) {
    const label = document.createElement('label');
    label.textContent = field.label;
    let input;
    if (field.type === 'select' || (!field.type && field.options?.length)) {
      input = document.createElement('select');
      if (field.required) input.required = true;
      for (const opt of field.options || []) {
        const optionEl = document.createElement('option');
        optionEl.value = opt.value !== undefined ? opt.value : opt;
        optionEl.textContent = opt.label !== undefined ? opt.label : opt;
        input.appendChild(optionEl);
      }
    } else if (field.type === 'number') {
      input = document.createElement('input');
      input.type = 'number';
    } else if (field.type === 'multiselect') {
      input = document.createElement('select');
      input.multiple = true;
      for (const opt of field.options || []) {
        const optionEl = document.createElement('option');
        optionEl.value = opt.value !== undefined ? opt.value : opt;
        optionEl.textContent = opt.label !== undefined ? opt.label : opt;
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
    const answerLines = ['User answers:'];
    for (const field of fields) {
      const el = form.elements.namedItem(field.id);
      if (el) {
        const val = field.type === 'multiselect'
          ? Array.from(el.selectedOptions).map(option => option.value)
          : el.value;
        answers[field.id] = val;
        if (val !== undefined && val !== '') {
          const displayVal = Array.isArray(val) ? val.join(', ') : val;
          answerLines.push(`- ${field.label}: ${displayVal}`);
        }
      }
    }
    submit.disabled = true;
    try {
      await api('/api/questions/answer', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({project: currentProject, answers}),
      });
      clearQuestionArea();
      finalizeAllActiveCards();
      addMessage(answerLines.join('\n'), 'user');
      getOrCreateRunCard(null, { optimistic: true });
      setThinking(true);
    } catch (error) {
      addMessage(error.message, 'error');
      submit.disabled = false;
    }
  });
  questionArea.appendChild(form);
}

// SSE event stream & Run-Card State Machine
let eventSource = null;
const runCards = new Map(); // run_id -> message card element
let optimisticCard = null;  // Card created on user submit, adopted when run_id is known

function scrollFeedToBottom(force = false) {
  if (!feed) return;
  const isNearBottom = feed.scrollHeight - feed.scrollTop - feed.clientHeight <= 60;
  if (force || isNearBottom) {
    feed.scrollTop = feed.scrollHeight;
  }
}

function getActiveCard(runId) {
  if (runId && runCards.has(runId)) {
    const card = runCards.get(runId);
    if (card && card.dataset.state !== 'done' && card.dataset.state !== 'error' && card.dataset.state !== 'stopped') {
      return card;
    }
  }
  if (optimisticCard && optimisticCard.dataset.state !== 'done' && optimisticCard.dataset.state !== 'error' && optimisticCard.dataset.state !== 'stopped') {
    if (runId) {
      optimisticCard.dataset.runId = runId;
      runCards.set(runId, optimisticCard);
      const card = optimisticCard;
      optimisticCard = null;
      return card;
    }
    return optimisticCard;
  }
  for (const c of runCards.values()) {
    if (c.dataset.state !== 'done' && c.dataset.state !== 'error' && c.dataset.state !== 'stopped') {
      if (runId && !c.dataset.runId) {
        c.dataset.runId = runId;
        runCards.set(runId, c);
      }
      return c;
    }
  }
  if (feed) {
    const streaming = feed.querySelectorAll('.message.agent.streaming');
    for (let i = streaming.length - 1; i >= 0; i--) {
      const c = streaming[i];
      if (c.dataset.state !== 'done' && c.dataset.state !== 'error' && c.dataset.state !== 'stopped') {
        if (runId) {
          c.dataset.runId = runId;
          runCards.set(runId, c);
        }
        return c;
      }
    }
  }
  return null;
}

function finalizeAllActiveCards() {
  for (const card of runCards.values()) {
    if (card && card.dataset.state !== 'done' && card.dataset.state !== 'error' && card.dataset.state !== 'stopped') {
      setCardState(card, 'done');
    }
    if (card && !card.dataset.raw && !card.dataset.reasoning) {
      card.remove();
    }
  }
  runCards.clear();
  if (optimisticCard) {
    if (optimisticCard.dataset.state !== 'done' && optimisticCard.dataset.state !== 'error' && optimisticCard.dataset.state !== 'stopped') {
      setCardState(optimisticCard, 'done');
    }
    if (!optimisticCard.dataset.raw && !optimisticCard.dataset.reasoning) {
      optimisticCard.remove();
    }
    optimisticCard = null;
  }
  if (feed) {
    feed.querySelectorAll('.message.agent.streaming').forEach(card => {
      if (card.dataset.state !== 'done' && card.dataset.state !== 'error' && card.dataset.state !== 'stopped') {
        setCardState(card, 'done');
      }
      if (!card.dataset.raw && !card.dataset.reasoning) {
        card.remove();
      }
    });
  }
  document.querySelectorAll('.thinking-indicator').forEach(el => el.remove());
}

function getOrCreateRunCard(runId, { target = feed, optimistic = false } = {}) {
  const existing = getActiveCard(runId);
  if (existing) {
    return existing;
  }

  // Finalize any prior active cards before mounting a new card into feed
  finalizeAllActiveCards();

  const empty = target.querySelector('.empty-state');
  if (empty) empty.remove();
  const indicator = target.querySelector('.thinking-indicator');
  if (indicator) indicator.remove();

  const item = document.createElement('div');
  item.className = 'message agent streaming thinking';
  item.dataset.state = 'thinking';
  item.dataset.raw = '';
  item.dataset.reasoning = '';
  if (runId) item.dataset.runId = runId;

  item.innerHTML = `
    <div class="message-meta">
      <span class="agent-mark">AI</span>
      <span class="message-author">Agent</span>
      <span class="message-state state-thinking">Thinking</span>
    </div>
    <div class="message-content">
      <div class="message-skeleton" aria-label="Waiting for response">
        <div class="skeleton-line" style="width: 88%;"></div>
        <div class="skeleton-line" style="width: 74%;"></div>
        <div class="skeleton-line" style="width: 52%;"></div>
      </div>
    </div>
  `;
  target.appendChild(item);
  if (runId) {
    runCards.set(runId, item);
  } else if (optimistic) {
    optimisticCard = item;
  }
  scrollFeedToBottom(true);
  return item;
}

function setCardState(card, state, labelText = null) {
  if (!card) return;
  card.dataset.state = state;
  const stateEl = card.querySelector('.message-state');
  if (!stateEl) return;

  if (state === 'done') {
    const disclosure = card.querySelector('.thought-disclosure');
    if (disclosure) {
      disclosure.open = false;
    }
  }

  if (state === 'thinking') {
    card.classList.add('streaming', 'thinking');
    card.classList.remove('writing', 'working');
    stateEl.className = 'message-state state-thinking';
    stateEl.textContent = labelText || 'Thinking';
    if (card.dataset.reasoning) {
      const skeleton = card.querySelector('.message-skeleton');
      if (skeleton) skeleton.remove();
      ensureThoughtDisclosure(card, card.dataset.reasoning, true, true);
    }
  } else if (state === 'writing') {
    card.classList.add('streaming', 'writing');
    card.classList.remove('thinking', 'working');
    stateEl.className = 'message-state state-writing';
    stateEl.textContent = labelText || 'Writing';
    const skeleton = card.querySelector('.message-skeleton');
    if (skeleton && card.dataset.raw && card.dataset.raw.trim()) skeleton.remove();
  } else if (state === 'working') {
    card.classList.add('streaming', 'working');
    card.classList.remove('thinking', 'writing');
    stateEl.className = 'message-state state-working';
    stateEl.textContent = labelText || 'Working';
  } else if (state === 'done') {
    card.classList.remove('streaming', 'thinking', 'writing', 'working');
    stateEl.textContent = '';
    stateEl.className = 'message-state';
    const skeleton = card.querySelector('.message-skeleton');
    if (skeleton) skeleton.remove();
  } else if (state === 'error') {
    card.classList.remove('streaming', 'thinking', 'writing', 'working');
    stateEl.textContent = labelText || 'Failed';
    stateEl.className = 'message-state state-error';
    const skeleton = card.querySelector('.message-skeleton');
    if (skeleton) skeleton.remove();
  } else if (state === 'stopped') {
    card.classList.remove('streaming', 'thinking', 'writing', 'working');
    stateEl.textContent = labelText || 'Stopped';
    stateEl.className = 'message-state state-stopped';
    const skeleton = card.querySelector('.message-skeleton');
    if (skeleton) skeleton.remove();
  }
}

let renderRafId = null;
let pendingRenderCard = null;

function applyCardRender(card) {
  if (!card) return;
  const rawClean = stripToolCallTags(card.dataset.raw || '', { trim: false });
  renderAgentContent(card, rawClean);
  if (card.dataset.reasoning) {
    const isThinking = card.dataset.state === 'thinking';
    const disclosure = ensureThoughtDisclosure(
      card,
      card.dataset.reasoning,
      isThinking,
      isThinking
    );
    const thoughtContent = disclosure?.querySelector('.thought-content');
    if (thoughtContent && isThinking) {
      thoughtContent.scrollTop = thoughtContent.scrollHeight;
    }
  }
  scrollFeedToBottom();
}

function flushStreamingRender() {
  if (renderRafId) {
    cancelAnimationFrame(renderRafId);
    renderRafId = null;
  }
  if (pendingRenderCard) {
    applyCardRender(pendingRenderCard);
    pendingRenderCard = null;
  }
}

function scheduleStreamingRender(card) {
  pendingRenderCard = card;
  if (renderRafId) return;
  renderRafId = requestAnimationFrame(() => {
    renderRafId = null;
    if (pendingRenderCard) {
      applyCardRender(pendingRenderCard);
      pendingRenderCard = null;
    }
  });
}

function appendStreamingReasoningDelta(runId, delta, messageId = null) {
  if (!delta) return;
  // Ignore raw encrypted ciphertext blocks (>= 50 continuous base64 chars)
  if (/^[A-Za-z0-9+/=]{50,}\s*$/.test(delta.trim())) return;
  const card = getOrCreateRunCard(runId);
  if (card.dataset.state !== 'thinking' && card.dataset.state !== 'writing') {
    setCardState(card, 'thinking');
  }
  if (messageId && card.dataset.lastReasoningTurnId && card.dataset.lastReasoningTurnId !== messageId && card.dataset.reasoning) {
    if (!card.dataset.reasoning.endsWith('\n\n')) {
      card.dataset.reasoning += '\n\n';
    }
  }
  if (messageId) card.dataset.lastReasoningTurnId = messageId;
  card.dataset.reasoning = (card.dataset.reasoning || '') + delta;
  scheduleStreamingRender(card);
}

function appendStreamingDelta(runId, delta, messageId = null) {
  if (!delta) return;
  const card = getOrCreateRunCard(runId);
  if (card.dataset.state !== 'writing') {
    setCardState(card, 'writing');
  }
  if (messageId && card.dataset.lastContentTurnId && card.dataset.lastContentTurnId !== messageId && card.dataset.raw) {
    if (!card.dataset.raw.endsWith('\n\n')) {
      card.dataset.raw += '\n\n';
    }
  }
  if (messageId) card.dataset.lastContentTurnId = messageId;
  card.dataset.raw = (card.dataset.raw || '') + delta;
  scheduleStreamingRender(card);
}

function handleStreamingToolDelta(runId, data) {
  const card = getActiveCard(runId) || getOrCreateRunCard(runId);
  const toolName = data?.name || data?.tool || (data?.function && data.function.name) || '';
  const label = toolName ? activityLabel(toolName) : 'Calling tool';
  setCardState(card, 'working', label);
}

function finalizeRunCard(runId, finalText, reasoning) {
  flushStreamingRender();
  let card = getActiveCard(runId);

  const text = stripToolCallTags(finalText || card?.dataset.raw || '').trim();
  const finalReasoning = stripEncryptedReasoning(card?.dataset.reasoning || reasoning || '').trim();
  const cleanReasoning = (text && text === finalReasoning) ? '' : finalReasoning;

  if (card) {
    if (text) {
      renderAgentContent(card, text);
      card.dataset.raw = text;
    } else {
      const contentEl = card.querySelector('.message-content');
      if (contentEl) contentEl.innerHTML = '';
      card.dataset.raw = '';
    }
    setCardState(card, 'done');
    if (cleanReasoning) {
      ensureThoughtDisclosure(card, cleanReasoning, false, false);
    } else {
      const existing = card.querySelector('.thought-disclosure');
      if (existing) existing.remove();
    }
    if (runId) runCards.delete(runId);
    if (optimisticCard === card) optimisticCard = null;

    // Safety sweep: ensure any other dangling cards are also set to done and pruned
    for (const [id, dangling] of runCards.entries()) {
      if (dangling.dataset.state !== 'done') setCardState(dangling, 'done');
      if (!dangling.dataset.raw && !dangling.dataset.reasoning) dangling.remove();
      runCards.delete(id);
    }
    optimisticCard = null;

    scrollFeedToBottom();
    return card;
  }

  const item = addMessage(finalText || '', 'agent', { reasoning: cleanReasoning });
  scrollFeedToBottom();
  return item;
}

async function syncAfterStreamReset() {
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
      if (!(data.status === 'completed' && activityItems.size > 0 && !activityItems.has('agent-run'))) {
        addToolMessage({
          call_id: 'agent-run',
          tool: 'agent',
          status: data.status || 'running',
          result: data.message,
        });
      }
      const runId = data.run_id;
      const card = getActiveCard(runId);

      if (data.status === 'started') {
        setThinking(true);
        if (card && card.dataset.state !== 'done' && card.dataset.state !== 'error' && card.dataset.state !== 'stopped') {
          const label = data.message ? data.message.replace(/\.+$/, '') : 'Planning CAD task';
          setCardState(card, 'thinking', label);
        }
      } else if (data.status === 'reviewing' || data.status === 'verifying') {
        setThinking(true);
        if (card && card.dataset.state !== 'done' && card.dataset.state !== 'error' && card.dataset.state !== 'stopped') {
          const label = data.status === 'verifying' ? 'Verifying model' : 'Reviewing build';
          setCardState(card, 'working', label);
        }
      } else if (data.status === 'running') {
        setThinking(true);
        if (card && card.dataset.state !== 'done' && card.dataset.state !== 'error' && card.dataset.state !== 'stopped') {
          const label = data.message || 'Working';
          setCardState(card, 'working', label);
        }
      } else if (['stopped', 'failed', 'completed', 'waiting_for_user'].includes(data.status)) {
        setThinking(false);
        if (data.status === 'stopped' && card) {
          setCardState(card, 'stopped', 'Stopped');
        } else if (data.status === 'failed' && card) {
          setCardState(card, 'error', 'Failed');
        }
        finalizeAllActiveCards();
      }
    },
    question: data => {
      if (data.project !== currentProject) return;
      setThinking(false);
      showQuestion(data);
      finalizeAllActiveCards();
    },
    agent_reasoning_delta: data => {
      if (data.project !== currentProject) return;
      const chunk = data.content ?? data.delta ?? '';
      const runId = data.run_id || data.message_id;
      if (chunk) appendStreamingReasoningDelta(runId, chunk, data.message_id);
    },
    agent_content_delta: data => {
      if (data.project !== currentProject) return;
      const chunk = data.content ?? data.delta ?? '';
      const runId = data.run_id || data.message_id;
      if (chunk) appendStreamingDelta(runId, chunk, data.message_id);
    },
    agent_tool_delta: data => {
      if (data.project !== currentProject) return;
      const runId = data.run_id || data.message_id;
      handleStreamingToolDelta(runId, data);
    },
    agent_stream_end: data => {
      if (data.project !== currentProject) return;
      flushStreamingRender();
      const runId = data.run_id || data.message_id;
      const card = getActiveCard(runId);
      if (card) {
        const combined = stripEncryptedReasoning(card.dataset.reasoning || data.reasoning || '');
        if (combined) {
          card.dataset.reasoning = combined;
          ensureThoughtDisclosure(card, combined, false, false);
        }
        const text = stripToolCallTags(data.message || card.dataset.raw || '').trim();
        if (text) {
          card.dataset.raw = text;
          renderAgentContent(card, text);
        }
        if (data.has_tools) {
          setCardState(card, 'working', 'Running tools');
        }
      }
    },
    agent_message: data => {
      if (data.project !== currentProject) return;
      const runId = data.run_id || data.message_id;
      finalizeRunCard(runId, data.message || '', stripEncryptedReasoning(data.reasoning || ''));
      markActivityRecovered();
      setThinking(false);
      scrollFeedToBottom();
    },
    tool_status: data => {
      if (data.project !== currentProject) return;
      addToolMessage(data);
      const runId = data.run_id;
      const card = getActiveCard(runId);
      if (card && card.dataset.state !== 'done' && card.dataset.state !== 'error' && card.dataset.state !== 'stopped') {
        if (data.status === 'completed') {
          setCardState(card, 'thinking', 'Thinking');
        } else if (data.status === 'error') {
          setCardState(card, 'thinking', 'Thinking');
        } else if (data.status === 'rendering_views' || data.status === 'rendering_subset') {
          const label = data.result || 'Rendering review views';
          setCardState(card, 'working', label);
        } else if (data.status === 'running' || data.status === 'preparing') {
          let label = activityLabel(data.tool);
          const path = data.arguments?.path;
          if (path && (data.tool === 'read_file' || data.tool === 'write_file' || data.tool === 'edit_file')) {
            const verb = data.tool === 'read_file' ? 'Reading' : (data.tool === 'write_file' ? 'Writing' : 'Editing');
            label = `${verb} ${path}`;
          } else if (data.result && typeof data.result === 'string' && data.result.length <= 40 && !data.result.startsWith('{')) {
            label = data.result;
          }
          setCardState(card, 'working', label);
        } else {
          const label = (typeof data.result === 'string' && data.result.length <= 40 && !data.result.startsWith('{'))
            ? data.result
            : activityLabel(data.status || data.tool);
          setCardState(card, 'working', label);
        }
      }
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
      const cacheLabel = data.cache_hit ? 'Cache hit' : 'Re-rendered';
      // Mirror the cache outcome onto the activity row's state label so the
      // drawer shows "Re-rendered" / "Cache hit" instead of the generic
      // "Completed" the dispatcher set on tool_status. The row's existing
      // detail text (e.g. "Rendered 3/8 view(s) at standard quality") is
      // preserved so the user keeps the per-step summary that the dispatcher
      // already produced.
      activityTitle.textContent = `Re-rasterising views · ${cacheLabel}`;
      activityPanel.hidden = false;
      refreshLatestToolRow('cad_screenshot', {state: cacheLabel});
    },
    review_updated: data => {
      if (data.project !== currentProject) return;
      const verdictKey = data.status || 'inconclusive';
      const verdict = activityLabel(verdictKey);
      const summary = typeof data.summary === 'string' && data.summary.trim()
        ? ` — ${data.summary.trim().slice(0, 160)}`
        : '';
      activityTitle.textContent = `Reviewing build · ${verdict}${summary}`;
      activityPanel.hidden = false;
      // Promote the verdict onto the row's state label; the dispatcher's
      // tool_status:completed already filled the row's detail with the full
      // review summary, so we leave it untouched.
      refreshLatestToolRow('cad_review', {state: verdict});
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
      const runId = data.run_id || data.message_id;
      const card = getActiveCard(runId);
      if (card) {
        if (!card.dataset.raw && !card.dataset.reasoning) {
          card.remove();
        } else {
          setCardState(card, 'error', 'Failed');
        }
        if (runId) runCards.delete(runId);
        if (optimisticCard === card) optimisticCard = null;
      }
      finalizeAllActiveCards();
      addMessage(data.message || 'Agent error.', 'error');
      // Errors are terminal; ensure the thinking indicator clears.
      setThinking(false);
      addToolMessage({
        call_id: 'agent-run',
        tool: 'agent',
        status: 'failed',
        result: data.message || 'Agent error.',
      });
    } catch {}
  });
  eventSource.addEventListener('agent_stopped', event => {
    try {
      const data = JSON.parse(event.data);
      if (data.project !== currentProject) return;
      const runId = data.run_id || data.message_id;
      const card = getActiveCard(runId);
      if (card) {
        if (!card.dataset.raw && !card.dataset.reasoning) {
          card.remove();
        } else {
          setCardState(card, 'stopped', 'Stopped');
        }
        if (runId) runCards.delete(runId);
        if (optimisticCard === card) optimisticCard = null;
      }
      finalizeAllActiveCards();
      setThinking(false);
      addToolMessage({
        call_id: 'agent-run',
        tool: 'agent',
        status: 'stopped',
        result: 'Agent task stopped.',
      });
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
  if (renderRafId) {
    cancelAnimationFrame(renderRafId);
    renderRafId = null;
    pendingRenderCard = null;
  }
  optimisticCard = null;
  runCards.clear();
  // Wipe the main feed, dismiss any pending question, and clear activity so
  // the UI matches the freshly truncated conversation.jsonl. The drawer is
  // refreshed lazily the next time it is opened.
  clearActivity();
  feed.replaceChildren();
  clearQuestionArea();
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
  // Load the persisted conversation into the main feed so reopening a
  // project immediately shows its history. The empty-state element is
  // removed automatically once any displayable message is rendered.
  await loadHistory(currentProject);
  await loadCurrentState();
  await syncCurrentPreview();
  connectStream();
})();
