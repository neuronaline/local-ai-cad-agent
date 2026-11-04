import { CadViewer } from './viewer.js';
import {
  api,
  escapeHTML,
  checkPreflight,
  flashButtonFeedback,
  copyTextToClipboard,
  triggerBlobDownload,
  generateUUID,
} from './utils.js';
import {
  sanitizeHTML,
  stripToolCallTags,
  stripEncryptedReasoning,
  normalizeAgentTurn,
  normalizeHistoryContent,
  extractHistoryImages,
} from './sanitizer.js';
import {
  initEditor,
  loadModelCode,
  isCodeDirty,
  setViewerMode,
  jumpToEditorLine,
  triggerBuild,
} from './editor.js';
import {
  initRevisions,
  loadRevisions,
  isRevisionsDrawerOpen,
  closeRevisionsDrawer,
} from './revisions.js';

// ── DOM References & Config ──
const feed = document.querySelector('#chat-feed');
const chatForm = document.querySelector('#chat-form');
const message = document.querySelector('#message');
const dropzone = document.querySelector('#dropzone');
const attachments = document.querySelector('#attachments');
const attachmentLabel = document.querySelector('#attachment-label');
const sendToggle = document.querySelector('#send-toggle');
const sendIcon = sendToggle?.querySelector('[data-mode="send"]');
const stopIcon = sendToggle?.querySelector('[data-mode="stop"]');
const slashPopup = document.querySelector('#slash-popup');
const slashPopupList = document.querySelector('#slash-popup-list');
const questionArea = document.querySelector('#question-area');
const usageBar = document.querySelector('#usage-bar');
const usagePillTurn = document.querySelector('#usage-pill-turn');
const usageTurnMetrics = document.querySelector('#usage-turn-metrics');
const usagePillTotal = document.querySelector('#usage-pill-total');
const usageTotalMetrics = document.querySelector('#usage-total-metrics');
const attachmentPreview = document.querySelector('#attachment-preview');
const downloadModelBtn = document.querySelector('#download-model');
const exportFormatSelect = document.querySelector('#export-format');
const screenshotBtn = document.querySelector('#screenshot-btn');
const screenshotMenuBtn = document.querySelector('#screenshot-menu-btn');
const screenshotMenu = document.querySelector('#screenshot-menu');
const screenshotSingleBtn = document.querySelector('#screenshot-single-btn');
const screenshotAllBtn = document.querySelector('#screenshot-all-btn');
const reasoningEffortSelect = document.querySelector('#reasoning-effort');
const modelSelect = document.querySelector('#model-select');
const scrollBottomBtn = document.querySelector('#scroll-bottom-btn');
const scrollBottomBadge = document.querySelector('#scroll-bottom-badge');
const balancePill = document.querySelector('#balance-pill');
const balanceAmount = document.querySelector('#balance-amount');
const verificationPill = document.querySelector('#verification-pill');
const verificationIcon = document.querySelector('#verification-icon');
const verificationLabel = document.querySelector('#verification-label');
const telemetryPopover = document.querySelector('#telemetry-popover');
const telemetryBody = document.querySelector('#telemetry-body');
const telemetryClose = document.querySelector('#telemetry-close');
const hardwarePickerBtn = document.querySelector('#hardware-picker-btn');
const hardwarePickerLabel = document.querySelector('#hardware-picker-label');
const hardwareModal = document.querySelector('#hardware-modal');
const hardwareForm = document.querySelector('#hardware-form');
const hwPrinter = document.querySelector('#hw-printer');
const hwFilament = document.querySelector('#hw-filament');
const hwNozzle = document.querySelector('#hw-nozzle');
const hwAdditional = document.querySelector('#hw-additional');
const hardwareModalClose = document.querySelector('#hardware-modal-close');
const hardwareModalCancel = document.querySelector('#hardware-modal-cancel');
const imagePreviewModal = document.querySelector('#image-preview-modal');
const imageModalImg = document.querySelector('#image-modal-img');
const imageModalTitle = document.querySelector('#image-modal-title');
const imageModalClose = document.querySelector('#image-modal-close');

const appConfig = JSON.parse(document.querySelector('#app-config')?.textContent || '{}');
const showInfoMessages = appConfig.showInfoMessages ?? true;
const currentProject = appConfig.projectName || '';
const viewer = new CadViewer(document.querySelector('#viewer'), document.querySelector('#dimensions'), appConfig);

const VALID_EFFORTS = new Set(['', 'none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max']);
const availableModels = Array.isArray(appConfig.availableModels) ? appConfig.availableModels : [];
const validModelKeys = new Set(availableModels.map(m => m.key));

// ── Models & Reasoning Effort ──
function getSupportedEffortsForModel(modelKey) {
  const [provider = '', model = ''] = String(modelKey || '').toLowerCase().split(':');
  if (provider === 'mimo' || provider === 'xiaomi') {
    return new Set(['', 'none']);
  }
  if (provider === 'gemini' || provider === 'vertex') {
    const efforts = new Set(['', 'none', 'low', 'medium', 'high']);
    if (model.includes('2.5') || model.includes('gemini-2')) {
      efforts.add('minimal');
      efforts.add('xhigh');
      efforts.add('max');
    } else if (model.includes('flash') && !model.includes('gemini-4')) {
      efforts.add('minimal');
    }
    return efforts;
  }
  if (provider === 'openai') {
    return new Set(['', 'none', 'low', 'medium', 'high']);
  }
  return VALID_EFFORTS;
}

function syncEffortOptionsForModel(modelKey) {
  if (!reasoningEffortSelect) return;
  const [provider = ''] = String(modelKey || '').toLowerCase().split(':');
  const supported = getSupportedEffortsForModel(modelKey);
  const isMimo = provider === 'mimo' || provider === 'xiaomi';

  for (const opt of reasoningEffortSelect.options) {
    const allowed = supported.has(opt.value);
    opt.hidden = !allowed;
    opt.disabled = !allowed;
    if (opt.value === '') {
      opt.textContent = isMimo ? 'Enabled (Default)' : 'Default';
    } else if (opt.value === 'none') {
      opt.textContent = isMimo ? 'Disabled' : 'None';
    }
  }

  const picker = reasoningEffortSelect.closest('.effort-picker');
  const tooltip = isMimo ? 'MiMo Deep Thinking (enabled / disabled)' : 'Model reasoning effort';
  if (picker) {
    picker.title = tooltip;
  }
  reasoningEffortSelect.title = tooltip;
  reasoningEffortSelect.setAttribute('aria-label', tooltip);

  if (!supported.has(reasoningEffortSelect.value)) {
    reasoningEffortSelect.value = '';
  }
}

function getActiveModelKey() {
  const val = modelSelect?.value;
  return val && validModelKeys.has(val) ? val : (appConfig.activeModelKey || '');
}

function getActiveReasoningEffort() {
  const val = reasoningEffortSelect?.value ?? '';
  return VALID_EFFORTS.has(val) ? val : '';
}

function setActiveReasoningEffort(effort, syncBackend = true, persistLocal = true) {
  const normalized = effort || '';
  if (!VALID_EFFORTS.has(normalized) || !reasoningEffortSelect) return;
  const supported = getSupportedEffortsForModel(getActiveModelKey());
  const effective = supported.has(normalized) ? normalized : '';
  reasoningEffortSelect.value = effective;
  if (persistLocal) {
    try {
      if (effective) {
        localStorage.setItem('cad_user_reasoning_effort', effective);
      } else {
        localStorage.removeItem('cad_user_reasoning_effort');
      }
    } catch {}
  }
  if (syncBackend) {
    api('/api/settings/reasoning-effort', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({effort: effective}),
    }).catch(() => {});
  }
}

function setActiveModelKey(key, adoptModelEffort = false) {
  if (!key || !validModelKeys.has(key) || !modelSelect) return Promise.resolve();
  modelSelect.value = key;
  syncEffortOptionsForModel(key);
  try {
    localStorage.setItem('cad_selected_model', key);
  } catch {}
  return api('/api/settings/models', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({model_key: key}),
  }).then(res => {
    if (adoptModelEffort && res && VALID_EFFORTS.has(res.reasoning_effort ?? '')) {
      const supported = getSupportedEffortsForModel(key);
      const nextEffort = supported.has(res.reasoning_effort ?? '') ? (res.reasoning_effort ?? '') : '';
      setActiveReasoningEffort(nextEffort, false, false);
      try {
        localStorage.removeItem('cad_user_reasoning_effort');
      } catch {}
    }
    return res;
  }).catch(() => null);
}

function initModelAndEffort() {
  try {
    localStorage.removeItem('cad_reasoning_effort');
    const savedModel = localStorage.getItem('cad_selected_model');
    const savedEffort = localStorage.getItem('cad_user_reasoning_effort');
    const activeModel = (savedModel && validModelKeys.has(savedModel)) ? savedModel : getActiveModelKey();

    if (modelSelect && activeModel) {
      modelSelect.value = activeModel;
    }
    syncEffortOptionsForModel(activeModel);

    if (savedEffort !== null && VALID_EFFORTS.has(savedEffort)) {
      setActiveReasoningEffort(savedEffort, false, false);
    }

    if (savedModel && savedModel !== appConfig.activeModelKey && validModelKeys.has(savedModel)) {
      setActiveModelKey(savedModel, false);
    }
  } catch {}
}

modelSelect?.addEventListener('change', async () => {
  await setActiveModelKey(modelSelect.value, true);
});

reasoningEffortSelect?.addEventListener('change', () => {
  setActiveReasoningEffort(reasoningEffortSelect.value, true, true);
});

// ── Layout Resizer ──
function updateChatWidth(width) {
  const minWidth = 360;
  const maxWidth = Math.max(minWidth, window.innerWidth - 400);
  const clamped = Math.min(Math.max(width, minWidth), maxWidth);
  document.documentElement.style.setProperty('--chat-width', `${clamped}px`);
  const resizerEl = document.querySelector('#panel-resizer');
  if (resizerEl) {
    resizerEl.setAttribute('aria-valuenow', String(clamped));
    resizerEl.setAttribute('aria-valuemin', String(minWidth));
    resizerEl.setAttribute('aria-valuemax', String(maxWidth));
  }
  return clamped;
}

try {
  const savedWidth = parseInt(localStorage.getItem('cad_chat_width'), 10);
  updateChatWidth(Number.isFinite(savedWidth) ? savedWidth : 400);
} catch {}

const resizer = document.querySelector('#panel-resizer');
resizer?.addEventListener('pointerdown', event => {
  event.preventDefault();
  resizer.setPointerCapture(event.pointerId);
  document.body.classList.add('is-resizing');
  let currentWidth = null;
  const onMove = moveEvent => {
    currentWidth = updateChatWidth(moveEvent.clientX);
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

resizer?.addEventListener('keydown', event => {
  const currentChatWidth = parseInt(getComputedStyle(document.documentElement).getPropertyValue('--chat-width'), 10) || 400;
  let newWidth = currentChatWidth;
  if (event.key === 'ArrowRight') newWidth += 20;
  else if (event.key === 'ArrowLeft') newWidth -= 20;
  else if (event.key === 'Home') newWidth = 360;
  else if (event.key === 'End') newWidth = window.innerWidth - 400;
  else return;
  event.preventDefault();
  const applied = updateChatWidth(newWidth);
  try { localStorage.setItem('cad_chat_width', `${applied}px`); } catch {}
});

// ── Icons ──
const COPY_ICON_SVG = `
  <svg class="icon" viewBox="0 0 24 24" aria-hidden="true" focusable="false">
    <rect x="9" y="9" width="13" height="13" rx="2" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>
    <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>
  </svg>
`;

const RETRY_ICON_SVG = `
  <svg class="icon" viewBox="0 0 24 24" aria-hidden="true" focusable="false">
    <path d="M21 12a9 9 0 1 1-2.636-6.364M21 3v6h-6" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
  </svg>
`;

// ── Chat Rendering ──
function renderAgentContent(item, text) {
  const clean = stripToolCallTags(text || '', { trim: false });
  const contentEl = item.querySelector('.message-content');
  if (!contentEl) return;
  if (clean.trim()) {
    contentEl.innerHTML = sanitizeHTML(marked.parse(clean));
  } else if (item.dataset.state && item.dataset.state !== 'done' && !(item._rawReasoning && item.dataset.state === 'thinking')) {
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

function createThoughtDisclosure(reasoningText = '', isOpen = false) {
  const details = document.createElement('details');
  details.className = 'thought-disclosure';
  if (isOpen) details.open = true;

  const summary = document.createElement('summary');
  summary.className = 'thought-summary';
  summary.innerHTML = `
    <span class="thought-title">Thought Process</span>
    <button type="button" class="thought-copy-btn" title="Copy thought process" aria-label="Copy thought process">
      ${COPY_ICON_SVG}
      <span class="copy-label">Copy</span>
    </button>
    <span class="thought-chevron">⌄</span>
  `;

  const content = document.createElement('div');
  content.className = 'thought-content';
  content.dataset.raw = reasoningText;
  content.innerHTML = reasoningText && reasoningText.trim() ? sanitizeHTML(marked.parse(reasoningText)) : '';

  details.appendChild(summary);
  details.appendChild(content);
  return details;
}

function ensureThoughtDisclosure(card, reasoning, isOpen = false, isLive = false) {
  if (reasoning === null || reasoning === undefined) return null;
  const cleanReasoning = stripEncryptedReasoning(String(reasoning));
  if (!cleanReasoning) return null;
  if (!isLive && card._rawText && card._rawText.trim() === cleanReasoning) {
    card.querySelector('.thought-disclosure')?.remove();
    return null;
  }
  let disclosure = card.querySelector('.thought-disclosure');
  if (!disclosure) {
    disclosure = createThoughtDisclosure(cleanReasoning, isOpen);
    const contentEl = card.querySelector('.message-content');
    card.insertBefore(disclosure, contentEl);
  } else {
    if (isOpen || isLive) disclosure.open = true;
    const content = disclosure.querySelector('.thought-content');
    if (content && content.dataset.raw !== cleanReasoning) {
      content.dataset.raw = cleanReasoning;
      content.innerHTML = cleanReasoning.trim() ? sanitizeHTML(marked.parse(cleanReasoning)) : '';
    }
  }
  return disclosure;
}

function createAgentCardElement({ runId = null, streaming = false, showSkeleton = false } = {}) {
  const item = document.createElement('div');
  item.className = `message agent${streaming ? ' streaming thinking' : ''}`;
  item.dataset.state = streaming ? 'thinking' : 'done';
  item._rawText = '';
  item._rawReasoning = '';
  if (runId) item.dataset.runId = runId;
  item.innerHTML = `
    <div class="message-meta">
      <span class="agent-mark">AI</span>
      <span class="message-author">Agent</span>
      <span class="message-state ${streaming ? 'state-thinking' : ''}">${streaming ? 'Thinking' : ''}</span>
      <button type="button" class="message-copy-btn" title="Copy message" aria-label="Copy message">
        ${COPY_ICON_SVG}
        <span class="copy-label">Copy</span>
      </button>
    </div>
    <div class="message-content">${showSkeleton ? `
      <div class="message-skeleton" aria-label="Waiting for response">
        <div class="skeleton-line" style="width: 88%;"></div>
        <div class="skeleton-line" style="width: 74%;"></div>
        <div class="skeleton-line" style="width: 52%;"></div>
      </div>` : ''}</div>
  `;
  return item;
}

function addMessage(text, type = 'agent', options = {}) {
  const target = options.target || feed;
  target.querySelector('.empty-state')?.remove();

  if (type === 'agent') {
    const isStreaming = Boolean(options.streaming);
    const item = createAgentCardElement({ streaming: isStreaming, showSkeleton: isStreaming && !text });
    item._rawText = text || '';
    item._rawReasoning = options.reasoning || '';
    if (options.messageId) item.dataset.messageId = options.messageId;
    ensureThoughtDisclosure(item, options.reasoning, false, false);
    target.appendChild(item);
    renderAgentContent(item, text || '');
    return item;
  }

  const item = document.createElement('div');
  item.className = `message ${type}`;
  item._rawText = text;
  if (options.messageId) item.dataset.messageId = options.messageId;

  if (type === 'user') {
    item.classList.add('user-message');
    const meta = document.createElement('div');
    meta.className = 'message-meta user-meta';
    meta.innerHTML = `
      <span class="user-mark">U</span>
      <span class="message-author">You</span>
      <div class="user-actions">
        <button type="button" class="message-retry-btn" title="Retry prompt" aria-label="Retry prompt">
          ${RETRY_ICON_SVG}
          <span class="retry-label">Retry</span>
        </button>
        <button type="button" class="message-copy-btn" title="Copy message" aria-label="Copy message">
          ${COPY_ICON_SVG}
          <span class="copy-label">Copy</span>
        </button>
      </div>
    `;
    item.appendChild(meta);

    if (Array.isArray(options.images) && options.images.length) {
      const strip = document.createElement('div');
      strip.className = 'message-attachments';
      for (const img of options.images) {
        const el = document.createElement('img');
        el.className = 'attachment-thumb';
        el.src = img.src;
        el.alt = img.alt || 'attachment';
        el.title = 'Click to preview';
        el.loading = 'lazy';
        strip.appendChild(el);
      }
      item.appendChild(strip);
    }
    const textNode = document.createElement('div');
    textNode.className = 'message-text';
    textNode.innerHTML = text && text.trim() ? sanitizeHTML(marked.parse(text)) : escapeHTML(text || '');
    item.appendChild(textNode);
    target.appendChild(item);
    return item;
  }

  if (type === 'error') {
    const errorBody = document.createElement('div');
    errorBody.className = 'error-body';
    const textNode = document.createElement('div');
    textNode.className = 'error-text';
    textNode.textContent = text || '';
    errorBody.appendChild(textNode);

    if (options.retryable !== false) {
      const actions = document.createElement('div');
      actions.className = 'error-actions';
      const retryBtn = document.createElement('button');
      retryBtn.type = 'button';
      retryBtn.className = 'error-retry-btn';
      retryBtn.title = 'Retry this request';
      retryBtn.setAttribute('aria-label', 'Retry this request');
      retryBtn.innerHTML = `${RETRY_ICON_SVG}<span>Retry</span>`;
      retryBtn.addEventListener('click', (e) => {
        e.preventDefault();
        e.stopPropagation();
        handleRetry(item, options.retryContext, retryBtn);
      });
      actions.appendChild(retryBtn);
      errorBody.appendChild(actions);
    }
    item.appendChild(errorBody);
    target.appendChild(item);
    return item;
  }

  item.textContent = text || '';
  target.appendChild(item);
  return item;
}

// ── Persistent Inline Tool Call Bubbles & Dual Token Usage ──
const activityLabels = {
  cad_build: 'Building model',
  cad_build_and_verify: 'Building model',
  get_view_images: 'Retrieving images',
  write_file: 'Updating model',
  edit_file: 'Updating model',
  read_file: 'Reading model',
  python: 'Running calculation',
  calculate: 'Running calculation',
  run_python: 'Running calculation',
  question: 'Requesting input',
  Images: 'Reference images',
  cad_screenshot: 'Re-rasterising views',
  cad_review: 'Reviewing build',
  rendering_subset: 'Re-rasterising a few views',
  screenshot_auto: 'Rendering views for review',
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
  failed: 'Failed',
  stopped: 'Stopped',
  pass: 'Pass',
  fail: 'Fail',
  inconclusive: 'Inconclusive',
  limit_reached: 'Limit reached',
};

function activityLabel(value) {
  return activityLabels[value] || String(value || 'Activity').replace(/[_-]+/g, ' ');
}

function parseResultPayload(value) {
  if (!value) return null;
  if (typeof value === 'object') return value;
  if (typeof value === 'string') {
    try { return JSON.parse(value); } catch { return null; }
  }
  return null;
}

function activityDetail(tool, status, value) {
  if (!value) return '';
  if (!['completed', 'error', 'failed'].includes(status)) return String(value);
  const payload = parseResultPayload(value);
  if (!payload || typeof payload !== 'object') return String(value || '');
  if (payload.ok === false) return String(payload.error?.message || 'The step failed.');
  const data = payload.ok === true ? payload.data : payload;
  if (typeof data === 'string') return data;
  if ((tool === 'cad_build' || tool === 'cad_build_and_verify') && data && typeof data.summary === 'string' && data.summary.trim()) {
    return data.summary.trim();
  }
  if (tool === 'cad_build' || tool === 'cad_build_and_verify') return 'Model built and review artifacts created.';
  if (tool === 'get_view_images' && data && Array.isArray(data.images)) {
    return `${data.images.length} view(s) retrieved.`;
  }
  if ((tool === 'python' || tool === 'calculate' || tool === 'run_python') && data && typeof data === 'object') {
    const hasResult = data.result !== undefined && data.result !== null;
    const stdout = typeof data.stdout === 'string' ? data.stdout.trim() : '';
    if (hasResult) {
      const resStr = typeof data.result === 'object' ? JSON.stringify(data.result) : String(data.result);
      return stdout ? `${stdout}\nResult: ${resStr}` : `Result: ${resStr}`;
    }
    return stdout || 'Calculation completed.';
  }
  if (data && typeof data === 'object') {
    const output = data.stdout || data.stderr || data.message || data.summary;
    if (typeof output === 'string' && output.trim()) return output.trim();
  }
  return (status === 'error' || status === 'failed') ? 'The step failed.' : 'Completed.';
}

const activityItems = new Map();
const activeRunningTools = new Set();
const TERMINAL_STATUSES = new Set(['completed', 'error', 'failed', 'stopped', 'pass', 'fail']);

function clearActivity({ resetUsage = false } = {}) {
  activeRunningTools.clear();
  activityItems.clear();
  if (resetUsage) resetUsagePill();
}

function getToolShortMeta(item) {
  const tool = item.tool || '';
  const args = item.arguments;
  const firstLine = (str) => String(str || '').split(/\r?\n/)[0].trim();

  const isErr = item.status === 'error' || item.status === 'failed' || item.status === 'fail';
  if (isErr && item.result) {
    const line = firstLine(item.result);
    if (line) {
      return line.length > 90 ? `${line.slice(0, 87)}…` : line;
    }
  }
  if (args && typeof args === 'object') {
    if (tool === 'write_file' && typeof args.content === 'string') {
      const lines = args.content.split(/\r?\n/).length;
      return `model.scad · ${lines} ${lines === 1 ? 'line' : 'lines'}`;
    }
    if (tool === 'edit_file') {
      const edits = Array.isArray(args.edits) ? args.edits.length : (args.old_string !== undefined ? 1 : 0);
      return edits > 0 ? `model.scad · ${edits} ${edits === 1 ? 'edit' : 'edits'}` : 'model.scad';
    }
    if (tool === 'read_file') {
      const range = args.offset || args.limit ? ` (offset ${args.offset || 1}${args.limit ? `, limit ${args.limit}` : ''})` : '';
      return `model.scad${range}`;
    }
    if (tool === 'get_view_images' && args.views) {
      const views = Array.isArray(args.views) ? args.views.join(', ') : String(args.views);
      return views;
    }
    if (tool === 'cad_build' && args.views) {
      const views = Array.isArray(args.views) ? args.views.join(', ') : String(args.views);
      return `views: ${views}`;
    }
  }
  if (item.result) {
    const line = firstLine(item.result);
    if (line && line !== 'Completed.') {
      return line.length > 90 ? `${line.slice(0, 87)}…` : line;
    }
  }
  return '';
}

function appendDetailSection(container, label, text, { isCode = false, isError = false } = {}) {
  if (!text || !String(text).trim()) return;
  const section = document.createElement('div');
  section.className = 'tool-detail-section';
  const labelEl = document.createElement('span');
  labelEl.className = 'tool-detail-label';
  labelEl.textContent = label;
  section.appendChild(labelEl);

  const raw = String(text);
  const clipped = raw.length > 3500 ? `${raw.slice(0, 3500)}\n… [truncated]` : raw;
  if (isCode) {
    const pre = document.createElement('pre');
    pre.className = 'tool-detail-code';
    pre.textContent = clipped;
    section.appendChild(pre);
  } else {
    const div = document.createElement('div');
    div.className = `tool-detail-text${isError ? ' is-error' : ''}`;
    div.textContent = clipped;
    section.appendChild(div);
  }
  container.appendChild(section);
}

function populateToolBubbleBody(bodyEl, item) {
  bodyEl.replaceChildren();
  const tool = item.tool || '';
  const args = item.arguments;
  const isErr = item.status === 'error' || item.status === 'failed' || item.status === 'fail';

  if (args && typeof args === 'object') {
    if (tool === 'write_file' && typeof args.content === 'string' && args.content.trim()) {
      appendDetailSection(bodyEl, 'model.scad content', args.content, { isCode: true });
    } else if (tool === 'edit_file') {
      const edits = Array.isArray(args.edits)
        ? args.edits
        : (typeof args.old_string === 'string' ? [{ old_string: args.old_string, new_string: args.new_string ?? '' }] : []);
      if (edits.length > 0) {
        const snippet = edits.map((e, idx) => {
          const header = edits.length > 1 ? `// Edit ${idx + 1}\n` : '';
          return `${header}- ${(e.old_string || '').trim()}\n+ ${(e.new_string || '').trim()}`;
        }).join('\n\n');
        appendDetailSection(bodyEl, `Changes (${edits.length})`, snippet, { isCode: true });
      }
    } else if ((tool === 'python' || tool === 'calculate' || tool === 'run_python') && typeof args.code === 'string' && args.code.trim()) {
      appendDetailSection(bodyEl, 'Python script', args.code, { isCode: true });
    } else if (tool === 'read_file') {
      const parts = ['File: model.scad'];
      if (args.offset) parts.push(`Offset: line ${args.offset}`);
      if (args.limit) parts.push(`Limit: ${args.limit} lines`);
      appendDetailSection(bodyEl, 'Parameters', parts.join(' · '));
    } else if (tool === 'cad_build' || tool === 'get_view_images') {
      const parts = [];
      if (args.views) parts.push(`Views: ${Array.isArray(args.views) ? args.views.join(', ') : args.views}`);
      if (args.resolution) parts.push(`Resolution: ${args.resolution}px`);
      if (args.crop) parts.push(`Crop: ${JSON.stringify(args.crop)}`);
      if (parts.length > 0) appendDetailSection(bodyEl, 'Parameters', parts.join(' · '));
    } else if (tool === 'question' && Array.isArray(args.questions)) {
      const qList = args.questions.map(q => `• ${q.question || q.id || ''}`).join('\n');
      appendDetailSection(bodyEl, 'Questions', qList);
    }
  }

  if (item.result) {
    appendDetailSection(bodyEl, isErr ? 'Error' : 'Result', item.result, { isError: isErr });
  } else if (!bodyEl.children.length) {
    appendDetailSection(
      bodyEl,
      'Status',
      TERMINAL_STATUSES.has(item.status) ? activityLabel(item.status) : `${activityLabel(tool)} in progress…`
    );
  }
}

function renderToolBubbleContent(bubble, item) {
  const status = item.status || 'running';
  bubble.dataset.status = status;
  bubble.dataset.tool = item.tool || 'tool';
  if (status === 'error' || status === 'failed' || status === 'fail') {
    bubble.open = true;
  }

  let summary = bubble.querySelector('.tool-bubble-summary');
  let body = bubble.querySelector('.tool-bubble-body');
  if (!summary) {
    summary = document.createElement('summary');
    summary.className = 'tool-bubble-summary';
    bubble.appendChild(summary);
  }
  if (!body) {
    body = document.createElement('div');
    body.className = 'tool-bubble-body';
    bubble.appendChild(body);
  }

  let iconClass = 'is-running';
  let iconChar = '';
  if (status === 'completed' || status === 'pass') {
    iconClass = 'is-completed';
    iconChar = '✓';
  } else if (status === 'error' || status === 'failed' || status === 'fail') {
    iconClass = 'is-error';
    iconChar = '✕';
  } else if (status === 'stopped') {
    iconClass = 'is-stopped';
    iconChar = '■';
  }

  summary.replaceChildren();
  const iconEl = document.createElement('span');
  iconEl.className = `tool-status-icon ${iconClass}`;
  iconEl.setAttribute('aria-hidden', 'true');
  iconEl.textContent = iconChar;

  const titleEl = document.createElement('span');
  titleEl.className = 'tool-bubble-title';
  titleEl.textContent = activityLabel(item.tool);

  const badgeEl = document.createElement('span');
  badgeEl.className = 'tool-bubble-badge';
  badgeEl.textContent = item.tool || 'tool';

  const metaEl = document.createElement('span');
  metaEl.className = 'tool-bubble-meta';
  metaEl.textContent = getToolShortMeta(item);

  const stateEl = document.createElement('span');
  stateEl.className = 'tool-bubble-state';
  stateEl.textContent = item.displayState || activityLabel(status);

  const chevronEl = document.createElement('span');
  chevronEl.className = 'tool-bubble-chevron';
  chevronEl.setAttribute('aria-hidden', 'true');
  chevronEl.textContent = '⌄';

  summary.append(iconEl, titleEl, badgeEl, metaEl, stateEl, chevronEl);
  populateToolBubbleBody(body, item);
}

function addToolMessage(data, { target = feed } = {}) {
  const callId = data.call_id || generateUUID();
  if (callId === 'agent-run' || data.tool === 'agent') {
    return null;
  }
  const status = data.status || 'running';
  if (!TERMINAL_STATUSES.has(status)) {
    activeRunningTools.add(callId);
  } else {
    activeRunningTools.delete(callId);
  }

  const item = activityItems.get(callId) || { callId, tool: data.tool || 'tool' };
  item.tool = data.tool || item.tool;
  item.status = status;
  if (data.arguments && typeof data.arguments === 'object') {
    item.arguments = data.arguments;
  }
  if (data.result !== undefined && data.result !== null) {
    item.rawResult = data.result;
  }
  const detail = activityDetail(item.tool, status, data.result);
  if (detail) item.result = detail;
  activityItems.set(callId, item);

  const container = target || feed;
  if (!container) return null;

  let bubble = container.querySelector(`.tool-bubble[data-call-id="${CSS.escape(callId)}"]`);
  if (!bubble) {
    container.querySelector('.empty-state')?.remove();
    if (container === feed && activeRunCard?.isConnected && activeRunCard.parentElement === container) {
      if (activeRunCard._rawText?.trim() || activeRunCard._rawReasoning?.trim()) {
        setCardState(activeRunCard, 'done');
        activeRunCard = null;
      } else {
        activeRunCard.remove();
        activeRunCard = null;
      }
    }
    bubble = document.createElement('details');
    bubble.className = 'tool-bubble';
    bubble.dataset.callId = callId;
    container.appendChild(bubble);
  }

  renderToolBubbleContent(bubble, item);
  if (container === feed) scrollFeedToBottom();
  return bubble;
}

function refreshLatestToolRow(tool, updates = {}) {
  const items = [...activityItems.values()];
  for (let i = items.length - 1; i >= 0; i--) {
    const item = items[i];
    if (item.tool !== tool) continue;

    if (updates.state) item.displayState = updates.state;
    if (updates.status) item.status = updates.status;
    if (Object.prototype.hasOwnProperty.call(updates, 'result')) item.result = updates.result;

    const bubble = feed?.querySelector(`.tool-bubble[data-call-id="${CSS.escape(item.callId)}"]`);
    if (bubble) {
      renderToolBubbleContent(bubble, item);
    }
    return bubble;
  }
  return null;
}

function markActivityRecovered(terminalStatus = 'completed') {
  activeRunningTools.clear();
  for (const item of activityItems.values()) {
    if (!TERMINAL_STATUSES.has(item.status)) {
      item.status = terminalStatus;
      const bubble = feed?.querySelector(`.tool-bubble[data-call-id="${CSS.escape(item.callId)}"]`);
      if (bubble) {
        renderToolBubbleContent(bubble, item);
      }
    }
  }
}

// ── Dual Token Usage Bar (Last Turn & All Chat) ──
let turnUsageBuckets = [];

function createUsageBucket(runId = null) {
  return {
    runId: runId || null,
    prompt: 0,
    completion: 0,
    cached: 0,
    cost: 0,
    hasCost: false,
    calls: 0,
  };
}

function startNewTurnUsageBucket(runId = null) {
  const last = turnUsageBuckets.at(-1);
  if (last && last.calls === 0) {
    if (runId) last.runId = runId;
    return last;
  }
  const bucket = createUsageBucket(runId);
  turnUsageBuckets.push(bucket);
  return bucket;
}

function rewindUsageBucketsToUserIndex(userIndex) {
  if (typeof userIndex === 'number' && userIndex >= 0) {
    turnUsageBuckets.length = Math.min(turnUsageBuckets.length, userIndex);
  } else if (turnUsageBuckets.length > 0) {
    turnUsageBuckets.pop();
  }
  startNewTurnUsageBucket();
  renderUsageBar();
}

function formatTokenCount(value) {
  const n = Number(value);
  if (!Number.isFinite(n) || n < 0) return '—';
  if (n >= 10000) return `${Math.round(n / 1000)}k`;
  return String(n);
}

function formatCost(amount) {
  if (amount == null || !Number.isFinite(Number(amount))) return '';
  const num = Number(amount);
  if (num === 0) return '$0.00';
  if (num < 0.0001) return '<$0.0001';
  if (num < 0.01) return `$${num.toFixed(4)}`;
  return `$${num.toFixed(3)}`;
}

function applyUsageBadge(pillEl, metricsEl, scopeTitle, snapshot) {
  if (!pillEl || !metricsEl) return;
  const promptLabel = formatTokenCount(snapshot.prompt);
  const completionLabel = formatTokenCount(snapshot.completion);
  const cachedLabel = formatTokenCount(snapshot.cached);
  const hitRate = snapshot.prompt > 0 ? (snapshot.cached / snapshot.prompt) * 100 : 0;
  const hitRatePercent = Math.round(hitRate);

  let cacheState = 'miss';
  let cacheLabel = ' · 0% hit';
  if (snapshot.cached > 0 || hitRatePercent > 0) {
    cacheState = hitRatePercent >= 80 ? 'hit' : 'partial';
    cacheLabel = ` · ${hitRatePercent}% hit (${cachedLabel})`;
  }

  const costVal = snapshot.hasCost ? snapshot.cost : null;
  const costLabel = costVal != null ? ` · ${formatCost(costVal)}` : '';

  pillEl.dataset.cache = cacheState;
  metricsEl.textContent = `↑${promptLabel} ↓${completionLabel}${cacheLabel}${costLabel}`;
  pillEl.title =
    `${scopeTitle} (${snapshot.calls} LLM ${snapshot.calls === 1 ? 'step' : 'steps'}) — ` +
    `Prompt: ${snapshot.prompt.toLocaleString()} tokens | ` +
    `Completion: ${snapshot.completion.toLocaleString()} tokens | ` +
    `Cached: ${snapshot.cached.toLocaleString()} tokens (${hitRatePercent}% hit)` +
    (costVal != null ? ` | Cost: ${formatCost(costVal)}` : '');
  pillEl.hidden = false;
}

function renderUsageBar() {
  if (!usageBar) return;
  const activeBuckets = turnUsageBuckets.filter(b => b.calls > 0);
  if (!activeBuckets.length) {
    usageBar.hidden = true;
    if (usagePillTurn) usagePillTurn.hidden = true;
    if (usagePillTotal) usagePillTotal.hidden = true;
    return;
  }

  const lastTurn = activeBuckets.at(-1);
  const totalChat = activeBuckets.reduce(
    (acc, b) => {
      acc.prompt += b.prompt;
      acc.completion += b.completion;
      acc.cached += b.cached;
      if (b.hasCost) {
        acc.cost += b.cost;
        acc.hasCost = true;
      }
      acc.calls += b.calls;
      return acc;
    },
    createUsageBucket()
  );

  applyUsageBadge(usagePillTurn, usageTurnMetrics, 'Last Turn', lastTurn);
  applyUsageBadge(usagePillTotal, usageTotalMetrics, 'All Chat', totalChat);
  usageBar.hidden = false;
}

function updateUsagePill(data = {}) {
  const prompt = Number(data.prompt_tokens);
  const completion = Number(data.completion_tokens);
  const cached = Number(data.cached_tokens || 0);
  const cost = data.cost != null && Number.isFinite(Number(data.cost)) ? Number(data.cost) : null;

  if (!Number.isFinite(prompt) && !Number.isFinite(completion) && cost === null) {
    return;
  }

  let bucket = turnUsageBuckets.at(-1);
  if (!bucket) {
    bucket = startNewTurnUsageBucket(data.run_id || null);
  } else if (data.run_id && bucket.runId && bucket.runId !== data.run_id && bucket.calls > 0) {
    bucket = startNewTurnUsageBucket(data.run_id);
  } else if (data.run_id && !bucket.runId) {
    bucket.runId = data.run_id;
  }

  if (Number.isFinite(prompt) && prompt >= 0) bucket.prompt += prompt;
  if (Number.isFinite(completion) && completion >= 0) bucket.completion += completion;
  if (Number.isFinite(cached) && cached >= 0) bucket.cached += cached;
  if (cost !== null && cost >= 0) {
    bucket.cost += cost;
    bucket.hasCost = true;
  }
  bucket.calls += 1;

  renderUsageBar();
}

function resetUsagePill() {
  turnUsageBuckets = [];
  if (usageBar) usageBar.hidden = true;
  if (usagePillTurn) {
    usagePillTurn.hidden = true;
    delete usagePillTurn.dataset.cache;
  }
  if (usageTurnMetrics) usageTurnMetrics.textContent = '';
  if (usagePillTotal) {
    usagePillTotal.hidden = true;
    delete usagePillTotal.dataset.cache;
  }
  if (usageTotalMetrics) usageTotalMetrics.textContent = '';
}

// ── Run-Card & Streaming State Machine ──
let activeRunCard = null;
let isUserScrolledUp = false;
let unreadFeedCount = 0;
let pendingRenderCard = null;
let renderThrottleTimer = null;
const STREAMING_RENDER_THROTTLE_MS = 80;

const CARD_STATES = {
  thinking: { cls: 'state-thinking', defaultLabel: 'Thinking', streaming: true },
  writing:  { cls: 'state-writing',  defaultLabel: 'Writing',  streaming: true },
  working:  { cls: 'state-working',  defaultLabel: 'Working',  streaming: true },
  done:     { cls: '',               defaultLabel: '',         streaming: false },
  error:    { cls: 'state-error',    defaultLabel: 'Failed',   streaming: false },
  stopped:  { cls: 'state-stopped',  defaultLabel: 'Stopped',  streaming: false },
};

function updateScrollBottomBtn() {
  if (!scrollBottomBtn) return;
  if (isUserScrolledUp) {
    scrollBottomBtn.hidden = false;
    if (scrollBottomBadge) {
      if (unreadFeedCount > 0) {
        scrollBottomBadge.textContent = unreadFeedCount > 99 ? '99+' : String(unreadFeedCount);
        scrollBottomBadge.hidden = false;
      } else {
        scrollBottomBadge.hidden = true;
      }
    }
  } else {
    scrollBottomBtn.hidden = true;
    unreadFeedCount = 0;
    if (scrollBottomBadge) scrollBottomBadge.hidden = true;
  }
}

feed?.addEventListener('scroll', () => {
  if (feed) {
    const atBottom = (feed.scrollHeight - feed.scrollTop - feed.clientHeight) <= 50;
    isUserScrolledUp = !atBottom;
    if (atBottom) unreadFeedCount = 0;
    updateScrollBottomBtn();
  }
}, { passive: true });

scrollBottomBtn?.addEventListener('click', () => {
  scrollFeedToBottom(true);
  unreadFeedCount = 0;
  updateScrollBottomBtn();
});

function scrollFeedToBottom(force = false) {
  if (!feed) return;
  if (force) {
    isUserScrolledUp = false;
    unreadFeedCount = 0;
    feed.scrollTop = feed.scrollHeight;
    updateScrollBottomBtn();
  } else if (!isUserScrolledUp) {
    feed.scrollTop = feed.scrollHeight;
  } else {
    unreadFeedCount += 1;
    updateScrollBottomBtn();
  }
}

function resolveRunCard(runId = null) {
  if (activeRunCard?.isConnected) {
    if (!runId || !activeRunCard.dataset.runId || activeRunCard.dataset.runId === runId) {
      if (runId && !activeRunCard.dataset.runId) activeRunCard.dataset.runId = runId;
      return activeRunCard;
    }
  }
  if (runId && feed) {
    const matching = feed.querySelectorAll(`.message.agent[data-run-id="${CSS.escape(runId)}"]`);
    if (matching.length) {
      const last = matching[matching.length - 1];
      activeRunCard = last;
      return last;
    }
  }
  const streaming = feed?.querySelector('.message.agent.streaming');
  if (streaming) {
    if (runId && !streaming.dataset.runId) streaming.dataset.runId = runId;
    activeRunCard = streaming;
    return streaming;
  }
  return null;
}

function getActiveCard(runId = null) {
  const card = resolveRunCard(runId);
  if (!card) return null;
  const state = card.dataset.state;
  return (state !== 'done' && state !== 'error' && state !== 'stopped') ? card : null;
}

function setCardState(card, state, labelText = null) {
  if (!card) return;
  card.dataset.state = state;
  const stateEl = card.querySelector('.message-state');
  const conf = CARD_STATES[state] || CARD_STATES.done;

  if (state === 'done') {
    const disclosure = card.querySelector('.thought-disclosure');
    if (disclosure) disclosure.open = false;
  }

  card.classList.toggle('streaming', conf.streaming);
  card.classList.toggle('thinking', state === 'thinking');
  card.classList.toggle('writing', state === 'writing');
  card.classList.toggle('working', state === 'working');

  if (stateEl) {
    stateEl.className = `message-state ${conf.cls}`.trim();
    stateEl.textContent = labelText ?? conf.defaultLabel;
  }

  const skeleton = card.querySelector('.message-skeleton');
  if (skeleton) {
    if (!conf.streaming || (state === 'thinking' && card._rawReasoning) || (state === 'writing' && card._rawText?.trim())) {
      skeleton.remove();
    }
  }

  if (state === 'thinking' && card._rawReasoning) {
    ensureThoughtDisclosure(card, card._rawReasoning, true, true);
  }
}

function finalizeAllActiveCards() {
  const candidates = new Set();
  if (activeRunCard?.isConnected) candidates.add(activeRunCard);
  feed?.querySelectorAll('.message.agent.streaming').forEach(c => candidates.add(c));

  for (const card of candidates) {
    if (card.dataset.state !== 'done' && card.dataset.state !== 'error' && card.dataset.state !== 'stopped') {
      setCardState(card, 'done');
    }
    if (!card._rawText && !card._rawReasoning) {
      card.remove();
    }
  }
  if (activeRunCard && !activeRunCard.isConnected) activeRunCard = null;
}

function getOrCreateRunCard(runId, { target = feed } = {}) {
  const existing = getActiveCard(runId);
  if (existing) {
    if (runId) existing.dataset.runId = runId;
    return existing;
  }

  finalizeAllActiveCards();
  target.querySelector('.empty-state')?.remove();

  const item = createAgentCardElement({ runId, streaming: true, showSkeleton: true });
  target.appendChild(item);
  activeRunCard = item;
  scrollFeedToBottom(true);
  return item;
}

function applyCardRender(card) {
  if (!card) return;
  card._lastRenderTime = performance.now();
  renderAgentContent(card, card._rawText || '');
  if (card._rawReasoning) {
    const isThinking = card.dataset.state === 'thinking';
    const disclosure = ensureThoughtDisclosure(
      card,
      card._rawReasoning,
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

function scheduleStreamingRender(card) {
  pendingRenderCard = card;
  if (renderThrottleTimer) return;
  const now = performance.now();
  const elapsed = now - (card._lastRenderTime || 0);
  const delay = Math.max(0, STREAMING_RENDER_THROTTLE_MS - elapsed);
  renderThrottleTimer = setTimeout(() => {
    renderThrottleTimer = null;
    const target = pendingRenderCard;
    pendingRenderCard = null;
    applyCardRender(target);
  }, delay);
}

function flushStreamingRender() {
  if (renderThrottleTimer) {
    clearTimeout(renderThrottleTimer);
    renderThrottleTimer = null;
  }
  if (pendingRenderCard) {
    const target = pendingRenderCard;
    pendingRenderCard = null;
    applyCardRender(target);
  } else if (activeRunCard) {
    applyCardRender(activeRunCard);
  }
}

function appendStreamingReasoningDelta(runId, delta, messageId = null) {
  if (!delta || /^[A-Za-z0-9+/=]{50,}\s*$/.test(delta.trim())) return;
  const card = getOrCreateRunCard(runId);
  if (card.dataset.state !== 'thinking' && card.dataset.state !== 'writing') {
    setCardState(card, 'thinking');
  }
  if (messageId && card.dataset.lastReasoningTurnId && card.dataset.lastReasoningTurnId !== messageId && card._rawReasoning) {
    if (!card._rawReasoning.endsWith('\n\n')) card._rawReasoning += '\n\n';
  }
  if (messageId) card.dataset.lastReasoningTurnId = messageId;
  card._rawReasoning = (card._rawReasoning || '') + delta;
  scheduleStreamingRender(card);
}

function appendStreamingDelta(runId, delta, messageId = null) {
  if (!delta) return;
  const card = getOrCreateRunCard(runId);
  if (card.dataset.state !== 'writing') {
    setCardState(card, 'writing');
  }
  if (messageId && card.dataset.lastContentTurnId && card.dataset.lastContentTurnId !== messageId && card._rawText) {
    if (!card._rawText.endsWith('\n\n')) card._rawText += '\n\n';
  }
  if (messageId) card.dataset.lastContentTurnId = messageId;
  card._rawText = (card._rawText || '') + delta;
  scheduleStreamingRender(card);
}

function handleStreamingToolDelta(runId, data) {
  const card = getActiveCard(runId) || getOrCreateRunCard(runId);
  const toolName = data?.name || data?.tool || (data?.function && data.function.name) || '';
  const slot = `${data?.index ?? data?.id ?? 0}`;
  if (!card._streamedToolCalls) card._streamedToolCalls = new Map();
  let entry = card._streamedToolCalls.get(slot);
  if (!entry) {
    entry = { name: toolName, args: '' };
    card._streamedToolCalls.set(slot, entry);
  } else if (toolName && !entry.name) {
    entry.name = toolName;
  }
  if (typeof data?.arguments_delta === 'string') entry.args += data.arguments_delta;
  let label = entry.name ? activityLabel(entry.name) : 'Calling tool';
  const args = entry.args;
  if (args.length > 40) {
    const lineCount = (args.match(/\\n|\n/g) || []).length;
    if (lineCount > 0) {
      label = `${label} (${lineCount} ${lineCount === 1 ? 'line' : 'lines'})`;
    } else if (args.length >= 256) {
      label = `${label} (${(args.length / 1024).toFixed(1)} KB)`;
    }
  }
  const totalParallel = card._streamedToolCalls.size;
  if (totalParallel > 1) {
    label = `${label} [${totalParallel} parallel]`;
  }
  setCardState(card, 'working', label);
}

function finalizeRunCard(runId, finalText, incomingReasoning) {
  flushStreamingRender();
  const card = resolveRunCard(runId);
  const cardText = card?._rawText || '';
  const cardReasoning = card?._rawReasoning || '';
  const { text, reasoning } = normalizeAgentTurn({
    content: finalText || cardText,
    reasoning: cardReasoning || incomingReasoning,
  });

  if (card) {
    if (!text && !reasoning) {
      card.remove();
      if (activeRunCard === card) activeRunCard = null;
      scrollFeedToBottom();
      return null;
    }
    card._rawText = text;
    card._rawReasoning = reasoning;
    renderAgentContent(card, text);
    setCardState(card, 'done');
    if (reasoning) {
      ensureThoughtDisclosure(card, reasoning, false, false);
    } else {
      card.querySelector('.thought-disclosure')?.remove();
    }
    if (activeRunCard === card) activeRunCard = null;
    finalizeAllActiveCards();
    scrollFeedToBottom();
    return card;
  }

  if (text || reasoning) {
    const item = addMessage(text || '', 'agent', { reasoning });
    scrollFeedToBottom();
    return item;
  }
  return null;
}

function isAgentActive() {
  return sendToggle?.dataset.mode === 'stop';
}

function updateRetryButtonsState(active) {
  document.querySelectorAll('.error-retry-btn, .message-retry-btn').forEach(btn => {
    btn.disabled = active;
  });
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
  updateRetryButtonsState(active);
  if (active) {
    if (feed && !getActiveCard()) {
      getOrCreateRunCard(null, { optimistic: true });
    }
  } else {
    feed?.querySelectorAll('.message.agent').forEach(card => {
      if (card.dataset.state !== 'error' && card.dataset.state !== 'stopped') {
        card.classList.remove('streaming', 'thinking', 'writing', 'working');
      }
    });
  }
}

// ── Network & Chat Requests ──
let lastUserSubmission = null;
let selectedFiles = [];

async function sendChatRequest({ text, files = [] }) {
  activeRunningTools.clear();
  startNewTurnUsageBucket();
  finalizeAllActiveCards();
  setThinking(true);
  scrollFeedToBottom(true);

  const idempotencyKey = generateUUID();
  const body = new FormData();
  body.append('project', currentProject);
  body.append('message', text);
  body.append('idempotency_key', idempotencyKey);
  body.append('reasoning_effort', getActiveReasoningEffort());
  body.append('model_key', getActiveModelKey());
  if (Array.isArray(files)) {
    files.forEach(file => body.append('attachments', file));
  }
  const response = await api('/api/chat', { method: 'POST', body });
  if (response.duplicate) {
    finalizeAllActiveCards();
    setThinking(false);
    return response;
  }
  if (response.attachments?.length) {
    addToolMessage({
      call_id: `attachments-${generateUUID()}`,
      tool: 'Images',
      status: 'completed',
      result: `${response.attachments.length} reference image(s) uploaded.`,
    });
  }
  return response;
}

function pruneFeedAfterUserMessage(userIndex = -1) {
  const allUserMessages = Array.from(feed?.querySelectorAll('.message.user') || []);
  if (!allUserMessages.length) return;
  const targetIdx = (typeof userIndex === 'number' && userIndex >= 0 && userIndex < allUserMessages.length)
    ? userIndex
    : allUserMessages.length - 1;
  const targetUserEl = allUserMessages[targetIdx];
  if (!targetUserEl) return;

  let sibling = targetUserEl.nextElementSibling;
  while (sibling) {
    const next = sibling.nextElementSibling;
    if (sibling.classList.contains('tool-bubble') && sibling.dataset.callId) {
      activityItems.delete(sibling.dataset.callId);
      activeRunningTools.delete(sibling.dataset.callId);
    }
    sibling.remove();
    sibling = next;
  }
}

async function handleRetry(errorElement = null, context = null, triggerBtn = null) {
  if (!currentProject || isAgentActive()) return;
  const retryBtn = triggerBtn || errorElement?.querySelector('.error-retry-btn') || document.querySelector('.message-retry-btn');
  if (retryBtn) {
    retryBtn.disabled = true;
    retryBtn.classList.add('retrying');
  }
  try {
    if (errorElement) errorElement.remove();
    feed?.querySelectorAll('.message.error').forEach(el => el.remove());

    if (context?.clientSubmission) {
      const sub = context.clientSubmission;
      if (sub.project && sub.project !== currentProject) {
        throw new Error('Cannot retry prompt for a different project.');
      }
      await sendChatRequest({ text: sub.text, files: sub.files });
      return;
    }

    const userIndex = (context && typeof context.userIndex === 'number' && context.userIndex >= 0)
      ? context.userIndex
      : -1;
    pruneFeedAfterUserMessage(userIndex);
    rewindUsageBucketsToUserIndex(userIndex);
    activeRunningTools.clear();
    finalizeAllActiveCards();
    setThinking(true);
    scrollFeedToBottom(true);

    const retryPayload = {
      project: currentProject,
      reasoning_effort: getActiveReasoningEffort(),
      model_key: getActiveModelKey(),
    };
    if (userIndex >= 0) {
      retryPayload.user_index = userIndex;
    }

    const response = await api('/api/chat/retry', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(retryPayload),
    });
    if (!response.accepted) {
      throw new Error(response.error || 'Retry request not accepted.');
    }
  } catch (error) {
    finalizeAllActiveCards();
    setThinking(false);
    addMessage(`Retry failed: ${error.message}`, 'error', {retryable: true, retryContext: context});
  } finally {
    if (retryBtn) {
      retryBtn.disabled = false;
      retryBtn.classList.remove('retrying');
    }
  }
}

// ── Preview & State Sync ──
let previewProject = '';
let loadedPreviewRevision = '';
let previewLoadPromise = null;
let pendingPreviewRequest = null;
let busyWithOtherProject = null;

function setScreenshotButtonsDisabled(disabled) {
  if (screenshotBtn) screenshotBtn.disabled = disabled;
  if (screenshotMenuBtn) screenshotMenuBtn.disabled = disabled;
  if (screenshotSingleBtn) screenshotSingleBtn.disabled = disabled;
  if (screenshotAllBtn) screenshotAllBtn.disabled = disabled;
  if (disabled && screenshotMenu && !screenshotMenu.hidden) {
    screenshotMenu.hidden = true;
    screenshotMenuBtn?.setAttribute('aria-expanded', 'false');
  }
}

async function loadCurrentPreview(previewId) {
  if (!currentProject) return;
  if (previewLoadPromise) {
    pendingPreviewRequest = { previewId };
    return previewLoadPromise;
  }
  previewLoadPromise = (async () => {
    const project = currentProject;
    try {
      const url = `/api/projects/${encodeURIComponent(project)}/preview?ts=${Date.now()}`;
      await viewer.load(url);
      previewProject = project;
      if (document.body.dataset.mobileView === 'chat') {
        document.querySelector('#mobile-preview-badge')?.classList.remove('hidden');
      }
      if (downloadModelBtn) downloadModelBtn.disabled = false;
      setScreenshotButtonsDisabled(false);
      api(`/api/projects/${encodeURIComponent(project)}/preview/meta`)
        .then(meta => {
          if (meta) {
            loadedPreviewRevision = meta.revision || loadedPreviewRevision;
            updateVerificationUI(meta.verification, meta.metrics);
          }
        })
        .catch(() => {});
    } catch (error) {
      if (error.message?.includes('superseded')) return;
      if (downloadModelBtn) downloadModelBtn.disabled = true;
      setScreenshotButtonsDisabled(true);
      if (error.message?.includes('404') || error.message?.includes('NOT FOUND')) {
        viewer.clear('No preview yet');
        return;
      }
      addMessage(`Preview failed: ${error.message}`, 'error');
    } finally {
      previewLoadPromise = null;
      if (pendingPreviewRequest) {
        const next = pendingPreviewRequest;
        pendingPreviewRequest = null;
        loadCurrentPreview(next.previewId);
      }
    }
  })();
  return previewLoadPromise;
}

function hideUnapprovedPreview(reviewStatus = 'pending') {
  previewProject = '';
  loadedPreviewRevision = '';
  if (downloadModelBtn) downloadModelBtn.disabled = true;
  setScreenshotButtonsDisabled(true);
  viewer.clear(reviewStatus === 'fail' ? 'Preview was rejected by review' : 'Preview is awaiting review');
}

async function syncCurrentPreview() {
  if (!currentProject || previewLoadPromise) return;
  try {
    const meta = await api(`/api/projects/${encodeURIComponent(currentProject)}/preview/meta`);
    if (meta.available) {
      if (meta.verification || meta.metrics) updateVerificationUI(meta.verification, meta.metrics);
      if (previewProject !== currentProject || loadedPreviewRevision !== meta.revision) {
        if (meta.displayable) await loadCurrentPreview();
        else hideUnapprovedPreview(meta.review_status);
      }
    }
  } catch {}
}

function setBusyWithOther(activeProject) {
  busyWithOtherProject = activeProject || null;
  const banner = document.querySelector('#busy-project-banner');
  if (!banner) return;
  if (activeProject) {
    const safeProj = escapeHTML(activeProject);
    banner.innerHTML = `
      <div class="busy-banner-content">
        <span class="busy-banner-icon" aria-hidden="true">⚡</span>
        <span class="busy-banner-text">Agent task is currently running in <strong>${safeProj}</strong>.</span>
      </div>
      <div class="busy-banner-actions">
        <a href="/project/${encodeURIComponent(activeProject)}" class="btn-banner-link">Go to ${safeProj}</a>
        <button type="button" id="busy-banner-stop-btn" class="btn-banner-stop">Stop Task</button>
      </div>
    `;
    banner.hidden = false;
    const stopBtn = banner.querySelector('#busy-banner-stop-btn');
    stopBtn?.addEventListener('click', async () => {
      stopBtn.disabled = true;
      stopBtn.textContent = 'Stopping…';
      try {
        await api('/api/stop', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({project: activeProject}),
        });
        setBusyWithOther(null);
        addMessage(`Agent task in "${activeProject}" was stopped.`, 'info');
      } catch (err) {
        addMessage(`Failed to stop agent: ${err.message}`, 'error');
        stopBtn.disabled = false;
        stopBtn.textContent = 'Stop Task';
      }
    });
    if (message) {
      message.placeholder = `Agent is running in project "${activeProject}"…`;
    }
  } else {
    banner.hidden = true;
    banner.innerHTML = '';
    if (message && message.placeholder.startsWith('Agent is running in project')) {
      message.placeholder = 'Describe the part, dimensions, fit, and required features…';
    }
  }
}

async function loadCurrentState() {
  if (!currentProject) return;
  const data = await api(`/api/projects/${encodeURIComponent(currentProject)}/state`);
  if (data.busy_with_other && data.active_project) {
    setBusyWithOther(data.active_project);
  } else {
    setBusyWithOther(null);
  }
  if (data.status === 'waiting_for_user') {
    setThinking(false);
    const q = data.question || {};
    if (!questionArea.querySelector('.question-form')) {
      showQuestion({project: currentProject, ...q});
    }
  } else {
    if (questionArea.querySelector('.question-form')) clearQuestionArea();
    if (data.status === 'running') {
      setThinking(true);
    } else if (data.status === 'idle') {
      setThinking(false);
      finalizeAllActiveCards();
      markActivityRecovered();
    }
  }
}

function restoreEmptyState(target = feed, projectName = currentProject) {
  const emptyTpl = document.querySelector('#chat-empty');
  if (emptyTpl) {
    target.appendChild(emptyTpl.content.cloneNode(true));
  } else {
    const empty = document.createElement('div');
    empty.className = 'empty-state';
    empty.textContent = `Project "${projectName}" selected. Describe a part to begin.`;
    target.appendChild(empty);
  }
}

async function loadHistory(projectName) {
  const target = feed;
  try {
    const data = await api(`/api/projects/${encodeURIComponent(projectName)}/history`);
    clearActivity({ resetUsage: true });
    target.replaceChildren();
    clearQuestionArea();
    let renderedAny = false;

    let accumulatedReasoning = '';
    const flushAccumulatedReasoning = () => {
      if (accumulatedReasoning) {
        addMessage('', 'agent', { target, reasoning: accumulatedReasoning });
        accumulatedReasoning = '';
        renderedAny = true;
      }
    };

    for (const evt of data.events) {
      const role = evt.role || '';
      if (role === 'user') {
        flushAccumulatedReasoning();
        startNewTurnUsageBucket();
        const userImages = extractHistoryImages(evt.content);
        const text = normalizeHistoryContent(evt.content);
        addMessage(text, 'user', { target, images: userImages });
        renderedAny = true;
      } else if (role === 'assistant' || role === 'agent') {
        const { text, reasoning } = normalizeAgentTurn(evt);
        if (reasoning) {
          accumulatedReasoning = accumulatedReasoning
            ? `${accumulatedReasoning}\n\n${reasoning}`
            : reasoning;
        }
        const hasToolCalls = Array.isArray(evt.tool_calls) && evt.tool_calls.length > 0;
        if (!text) {
          if (hasToolCalls && accumulatedReasoning) {
            flushAccumulatedReasoning();
          }
          continue;
        }
        const cleanReasoning = (text === accumulatedReasoning.trim()) ? '' : accumulatedReasoning;
        addMessage(text, 'agent', { target, reasoning: cleanReasoning });
        accumulatedReasoning = '';
        renderedAny = true;
      } else if (evt.type === 'tool_status') {
        flushAccumulatedReasoning();
        const bubble = addToolMessage(evt.data, { target });
        if (bubble) renderedAny = true;
      } else if (evt.type === 'agent_error') {
        flushAccumulatedReasoning();
        addMessage(evt.data?.message || evt.content || 'Agent error.', 'error', { target });
        renderedAny = true;
      } else if (evt.type === 'agent_usage') {
        updateUsagePill(evt.data);
      } else if (showInfoMessages) {
        addInfoMessage(evt.type, evt.data, { target });
      }
    }
    flushAccumulatedReasoning();
    if (!isAgentActive()) {
      markActivityRecovered();
    }
    if (!renderedAny) restoreEmptyState(target, projectName);
  } catch (error) {
    addMessage(error.message, 'error', { target });
  }
}

function addInfoMessage(type, data = {}, options = {}) {
  if (type === 'tool_status') {
    addToolMessage(data, options);
  } else if (type === 'agent_usage') {
    updateUsagePill(data);
  } else if (type === 'agent_stopped') {
    markActivityRecovered('stopped');
  }
}

// ── Slash Commands ──
const COMMANDS = [
  {
    name: '/3mf',
    args: '[notes]',
    desc: 'Generate a ready-to-print 3MF with AI-tuned slicer settings',
    iconSvg: '<svg class="icon" viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="M6 9V2h12v7M6 18H4a2 2 0 0 1-2-2v-5a2 2 0 0 1 2-2h16a2 2 0 0 1 2 2v5a2 2 0 0 1-2 2h-2" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/><path d="M6 14h12v8H6z" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>',
    action: () => {
      message.value = '/3mf ';
      autoResizeMessage();
      message.focus();
      closeSlashPopup();
    },
  },
  {
    name: '/audit',
    args: '[notes]',
    desc: 'Audit model for geometric, manifold & manufacturing defects',
    iconSvg: '<svg class="icon" viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="m9 11 3 3L22 4" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/><path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>',
    action: () => {
      message.value = '/audit ';
      autoResizeMessage();
      message.focus();
      closeSlashPopup();
    },
  },
  {
    name: '/reset',
    alias: '/clear',
    args: '',
    desc: 'Reset conversation memory for this project',
    iconSvg: '<svg class="icon" viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="M4 12a8 8 0 0 1 13.7-5.6M20 4v4h-4M20 12a8 8 0 0 1-13.7 5.6M4 20v-4h4" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>',
    action: async () => {
      closeSlashPopup();
      message.value = '';
      autoResizeMessage();
      await executeResetContext();
    },
  },
];

let visibleCommands = [];
let slashSelectedIndex = 0;

function renderSlashPopup(items) {
  if (!slashPopupList) return;
  slashPopupList.replaceChildren();
  items.forEach((cmd, idx) => {
    const item = document.createElement('div');
    item.className = `slash-item${idx === slashSelectedIndex ? ' selected' : ''}`;
    item.setAttribute('role', 'option');
    item.setAttribute('aria-selected', String(idx === slashSelectedIndex));
    item.innerHTML = `
      <div class="slash-item-icon">${cmd.iconSvg}</div>
      <div class="slash-item-details">
        <span class="slash-item-name">${cmd.name}${cmd.args ? `<span class="slash-item-args">${cmd.args}</span>` : ''}</span>
        <span class="slash-item-desc">${cmd.desc}</span>
      </div>
    `;
    item.addEventListener('mousedown', (e) => {
      e.preventDefault();
      cmd.action();
    });
    slashPopupList.appendChild(item);
  });
}

function openSlashPopup(items) {
  if (!slashPopup || !items.length) {
    closeSlashPopup();
    return;
  }
  visibleCommands = items;
  slashSelectedIndex = 0;
  renderSlashPopup(items);
  slashPopup.classList.remove('hidden');
  slashPopup.hidden = false;
}

function closeSlashPopup() {
  if (!slashPopup) return;
  slashPopup.hidden = true;
  visibleCommands = [];
  slashSelectedIndex = 0;
}

function updateSlashSelection(newIndex) {
  if (!visibleCommands.length || !slashPopupList) return;
  slashSelectedIndex = (newIndex + visibleCommands.length) % visibleCommands.length;
  Array.from(slashPopupList.children).forEach((child, idx) => {
    const isSelected = idx === slashSelectedIndex;
    child.classList.toggle('selected', isSelected);
    child.setAttribute('aria-selected', String(isSelected));
  });
  slashPopupList.children[slashSelectedIndex]?.scrollIntoView({ block: 'nearest' });
}

async function executeResetContext() {
  if (!currentProject) return addMessage('Create or select a project first.', 'error');
  const proceed = window.confirm(
    'Reset the AI\u2019s conversation memory for this project?\n\nThe model, preview, and revisions are kept. The next message starts a fresh context.'
  );
  if (!proceed) return;
  try {
    await api(`/api/projects/${encodeURIComponent(currentProject)}/reset`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
    });
  } catch (error) {
    addMessage(error.message, 'error');
  }
}

function autoResizeMessage() {
  if (!message) return;
  message.style.height = 'auto';
  message.style.height = `${Math.min(Math.max(message.scrollHeight, 46), 140)}px`;
}

message?.addEventListener('input', () => {
  autoResizeMessage();
  const val = message.value;
  if (val.startsWith('/') && !val.includes(' ') && !val.includes('\n')) {
    const query = val.toLowerCase();
    const matches = COMMANDS.filter(cmd =>
      cmd.name.startsWith(query) || (cmd.alias && cmd.alias.startsWith(query))
    );
    if (matches.length > 0) {
      openSlashPopup(matches);
      return;
    }
  }
  closeSlashPopup();
});

message?.addEventListener('keydown', (e) => {
  if (slashPopup && !slashPopup.hidden && visibleCommands.length > 0) {
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      updateSlashSelection(slashSelectedIndex + 1);
      return;
    }
    if (e.key === 'ArrowUp') {
      e.preventDefault();
      updateSlashSelection(slashSelectedIndex - 1);
      return;
    }
    if (e.key === 'Enter' || e.key === 'Tab') {
      if (e.isComposing) return;
      e.preventDefault();
      visibleCommands[slashSelectedIndex]?.action();
      return;
    }
    if (e.key === 'Escape') {
      e.preventDefault();
      closeSlashPopup();
      return;
    }
  }
  if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    chatForm.dispatchEvent(new Event('submit'));
  }
});

chatForm?.addEventListener('submit', async event => {
  event.preventDefault();
  closeSlashPopup();
  if (!currentProject) return addMessage('Create a project first.', 'error');
  const text = message.value.trim();
  if (!text) return;
  if (text.toLowerCase() === '/reset' || text.toLowerCase() === '/clear') {
    message.value = '';
    autoResizeMessage();
    await executeResetContext();
    return;
  }
  if (sendToggle) sendToggle.disabled = true;
  message.disabled = true;
  const imagePayload = selectedFiles.map((file) => ({
    src: file._previewUrl || URL.createObjectURL(file),
    alt: file.name,
  }));
  try {
    addMessage(text, 'user', { images: imagePayload });
    message.value = '';
    autoResizeMessage();
    let promptText = text;
    if (text.toLowerCase() === '/audit' || text.toLowerCase().startsWith('/audit ')) {
      const extraInstructions = text.length > 6 ? text.slice(6).trim() : '';
      promptText =
        'Please inspect the current model.scad for any physical, geometric, or manufacturing issues ' +
        '(such as non-manifold edges, self-intersections, inverted normals, zero-thickness walls, or fragile parts). ' +
        'If you find any issues, please fix them directly in model.scad, run cad_build to verify, ' +
        'and briefly let me know what was wrong and how you resolved it.';
      if (extraInstructions) {
        promptText += `\n\nAlso, please pay special attention to: ${extraInstructions}`;
      }
    }
    const currentFiles = [...selectedFiles];
    lastUserSubmission = {
      project: currentProject,
      text: promptText,
      originalText: text,
      files: currentFiles,
      images: imagePayload,
    };
    await sendChatRequest({ text: promptText, files: currentFiles });
    clearAttachments();
  } catch (error) {
    finalizeAllActiveCards();
    addMessage(error.message, 'error', {
      retryable: true,
      retryContext: { clientSubmission: lastUserSubmission },
    });
    setThinking(false);
    loadCurrentState().catch(() => {});
  } finally {
    if (!isQuestionPending) {
      if (sendToggle) sendToggle.disabled = false;
      message.disabled = false;
      message.focus();
    }
  }
});

sendToggle?.addEventListener('click', async () => {
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
      if (activeCard) setCardState(activeCard, 'stopped', 'Stopped');
      finalizeAllActiveCards();
      markActivityRecovered('stopped');
    } catch (error) {
      addMessage(error.message, 'error');
    } finally {
      sendToggle.disabled = false;
    }
    return;
  }
  chatForm.dispatchEvent(new Event('submit'));
});

// ── Attachments & Clipboard ──
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

attachments?.addEventListener('change', () => {
  handleFiles(Array.from(attachments.files || []));
  attachments.value = '';
});

['dragenter', 'dragover'].forEach(eventName => {
  dropzone?.addEventListener(eventName, event => {
    event.preventDefault();
    dropzone.classList.add('is-dragging');
  });
});
dropzone?.addEventListener('dragleave', event => {
  event.preventDefault();
  dropzone.classList.remove('is-dragging');
});
dropzone?.addEventListener('drop', event => {
  event.preventDefault();
  dropzone.classList.remove('is-dragging');
  const files = Array.from(event.dataTransfer?.files || []);
  if (files.length) handleFiles(files);
});

function clearAttachments() {
  selectedFiles = [];
  if (attachments) attachments.value = '';
  if (attachmentLabel) attachmentLabel.textContent = 'Attach';
  attachmentPreview?.replaceChildren();
}

function renderAttachmentPreview() {
  attachmentPreview?.replaceChildren();
  if (attachmentLabel) {
    attachmentLabel.textContent = selectedFiles.length ? `Images (${selectedFiles.length})` : 'Attach';
  }
  selectedFiles.forEach((file, index) => {
    const item = document.createElement('div');
    item.className = 'attachment-item';
    const image = document.createElement('img');
    if (!file._previewUrl) file._previewUrl = URL.createObjectURL(file);
    image.src = file._previewUrl;
    image.alt = file.name;
    image.title = 'Click to preview';
    const name = document.createElement('span');
    name.textContent = file.name;
    const remove = document.createElement('button');
    remove.type = 'button';
    remove.className = 'quiet icon-only';
    remove.textContent = '×';
    remove.title = `Remove ${file.name}`;
    remove.addEventListener('click', (event) => {
      event.stopPropagation();
      const [removed] = selectedFiles.splice(index, 1);
      if (removed?._previewUrl) URL.revokeObjectURL(removed._previewUrl);
      renderAttachmentPreview();
    });
    item.append(image, name, remove);
    attachmentPreview?.appendChild(item);
  });
}

window.addEventListener('paste', event => {
  if (event.target && event.target !== message && (event.target.tagName === 'INPUT' || event.target.tagName === 'TEXTAREA')) {
    return;
  }
  const items = event.clipboardData?.items;
  if (!items || !items.length) return;
  const imageFiles = [];
  for (const item of items) {
    if (item.kind === 'file' && item.type.startsWith('image/')) {
      const file = item.getAsFile();
      if (file) {
        const ext = file.type.split('/')[1] === 'jpeg' ? 'jpg' : (file.type.split('/')[1] || 'png');
        imageFiles.push(new File([file], `pasted-${Date.now()}.${ext}`, { type: file.type }));
      }
    }
  }
  if (imageFiles.length > 0) {
    event.preventDefault();
    handleFiles(imageFiles);
  }
});

function openImageModal(src, title = 'Reference Image') {
  if (!imagePreviewModal || !imageModalImg) return;
  imageModalImg.src = src;
  imageModalImg.alt = title || 'Reference Image';
  if (imageModalTitle) imageModalTitle.textContent = title;
  imagePreviewModal.hidden = false;
  imagePreviewModal.setAttribute('aria-hidden', 'false');
  imageModalClose?.focus();
}

function closeImageModal() {
  if (!imagePreviewModal) return;
  imagePreviewModal.hidden = true;
  imagePreviewModal.setAttribute('aria-hidden', 'true');
  if (imageModalImg) imageModalImg.src = '';
}

imageModalClose?.addEventListener('click', closeImageModal);
imagePreviewModal?.addEventListener('click', (event) => {
  if (event.target === imagePreviewModal) closeImageModal();
});

// ── Viewer Toolbar & Export ──
document.querySelector('#toggle-wireframe')?.addEventListener('click', event => {
  event.currentTarget.setAttribute('aria-pressed', String(viewer.toggleWireframe()));
});
document.querySelector('#toggle-grid')?.addEventListener('click', event => {
  event.currentTarget.setAttribute('aria-pressed', String(viewer.toggleGrid()));
});
document.querySelector('#reset-view')?.addEventListener('click', () => {
  viewer.fit();
  document.querySelectorAll('[data-view]').forEach(item => {
    const isIso = item.dataset.view === 'iso';
    item.classList.toggle('active', isIso);
    item.setAttribute('aria-pressed', String(isIso));
  });
});

downloadModelBtn?.addEventListener('click', async () => {
  if (!currentProject || downloadModelBtn.disabled) return;
  const format = exportFormatSelect?.value || 'stl';
  downloadModelBtn.disabled = true;
  try {
    const exportUrl = `/api/projects/${encodeURIComponent(currentProject)}/export?format=${encodeURIComponent(format)}`;
    const res = await fetch(exportUrl);
    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: 'Export failed' }));
      throw new Error(err.error || res.statusText);
    }
    const blob = await res.blob();
    triggerBlobDownload(blob, `${currentProject}.${format}`);
  } catch (error) {
    addMessage(`Export failed: ${error.message}`, 'error');
  } finally {
    downloadModelBtn.disabled = !previewProject;
  }
});

function closeScreenshotMenu() {
  if (screenshotMenu) screenshotMenu.hidden = true;
  screenshotMenuBtn?.setAttribute('aria-expanded', 'false');
}

function toggleScreenshotMenu() {
  if (!screenshotMenu || screenshotMenuBtn?.disabled) return;
  const isHidden = screenshotMenu.hidden;
  screenshotMenu.hidden = !isHidden;
  screenshotMenuBtn?.setAttribute('aria-expanded', String(isHidden));
}

screenshotMenuBtn?.addEventListener('click', (event) => {
  event.stopPropagation();
  toggleScreenshotMenu();
});

let isCapturingScreenshot = false;

async function handleSingleScreenshot() {
  closeScreenshotMenu();
  if (isCapturingScreenshot || !previewProject || !viewer.hasModel() || screenshotBtn?.disabled) return;
  isCapturingScreenshot = true;
  setScreenshotButtonsDisabled(true);
  try {
    const blob = await viewer.captureSingleScreenshot();
    triggerBlobDownload(blob, `${currentProject || 'model'}-view.png`);
    flashButtonFeedback(screenshotBtn, 'Saved!', { duration: 1500 });
  } catch (error) {
    addMessage(`Screenshot failed: ${error.message}`, 'error');
  } finally {
    isCapturingScreenshot = false;
    setScreenshotButtonsDisabled(!previewProject);
  }
}

async function handleAllAnglesScreenshot() {
  closeScreenshotMenu();
  if (isCapturingScreenshot || !previewProject || !viewer.hasModel() || screenshotBtn?.disabled) return;
  isCapturingScreenshot = true;
  setScreenshotButtonsDisabled(true);
  try {
    const dimensionsText = document.querySelector('#dimensions')?.textContent || '';
    const blob = await viewer.captureMultiViewScreenshot({
      projectName: currentProject,
      dimensionsText,
    });
    triggerBlobDownload(blob, `${currentProject || 'model'}-all-angles.png`);
    flashButtonFeedback(screenshotBtn, 'Saved!', { duration: 1500 });
  } catch (error) {
    addMessage(`Multi-angle screenshot failed: ${error.message}`, 'error');
  } finally {
    isCapturingScreenshot = false;
    setScreenshotButtonsDisabled(!previewProject);
  }
}

screenshotBtn?.addEventListener('click', handleSingleScreenshot);
screenshotSingleBtn?.addEventListener('click', handleSingleScreenshot);
screenshotAllBtn?.addEventListener('click', handleAllAnglesScreenshot);

document.querySelectorAll('[data-view]').forEach(button => {
  button.addEventListener('click', () => {
    viewer.setView(button.dataset.view);
    document.querySelectorAll('[data-view]').forEach(item => {
      const isActive = item === button;
      item.classList.toggle('active', isActive);
      item.setAttribute('aria-pressed', String(isActive));
    });
  });
});

viewer.controls.addEventListener('start', () => {
  document.querySelectorAll('[data-view]').forEach(item => {
    item.classList.remove('active');
    item.setAttribute('aria-pressed', 'false');
  });
});

document.querySelectorAll('[data-mobile-view]').forEach(button => {
  button.addEventListener('click', () => {
    document.body.dataset.mobileView = button.dataset.mobileView;
    document.querySelectorAll('[data-mobile-view]').forEach(item => {
      item.setAttribute('aria-pressed', String(item === button));
    });
    if (button.dataset.mobileView === 'viewer') {
      document.querySelector('#mobile-preview-badge')?.classList.add('hidden');
      viewer?.resize();
    }
  });
});

// ── Interactive Question Cards ──
let isQuestionPending = false;

function setComposerBlockedForQuestion(blocked) {
  isQuestionPending = blocked;
  message.disabled = blocked;
  if (sendToggle) sendToggle.disabled = blocked;
  message.placeholder = blocked ? 'Please answer the pending question above…' : 'Describe the part, dimensions, fit, and required features…';
}

function clearQuestionArea() {
  questionArea?.replaceChildren();
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
    const cardEl = document.createElement('div');
    cardEl.className = 'question-card-item';

    const labelEl = document.createElement('div');
    labelEl.className = 'question-card-label';
    labelEl.textContent = field.label;
    cardEl.appendChild(labelEl);

    if (field.type === 'select' || (!field.type && field.options?.length) || field.type === 'multiselect') {
      const isMulti = field.type === 'multiselect';
      const hiddenInput = document.createElement('input');
      hiddenInput.type = 'hidden';
      hiddenInput.name = field.id;
      cardEl.appendChild(hiddenInput);

      const optionsGrid = document.createElement('div');
      optionsGrid.className = 'question-options-grid';
      const selectedVals = new Set();
      const optionButtons = [];

      for (const opt of field.options || []) {
        const val = String(opt.value !== undefined ? opt.value : opt);
        const text = String(opt.label !== undefined ? opt.label : opt);
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'option-card';
        btn.dataset.value = val;
        btn.innerHTML = `
          <span class="option-indicator" aria-hidden="true"></span>
          <span class="option-title">${escapeHTML(text)}</span>
        `;
        btn.addEventListener('click', () => {
          if (isMulti) {
            if (selectedVals.has(val)) {
              selectedVals.delete(val);
              btn.classList.remove('selected');
            } else {
              selectedVals.add(val);
              btn.classList.add('selected');
            }
            hiddenInput.value = JSON.stringify(Array.from(selectedVals));
          } else {
            selectedVals.clear();
            selectedVals.add(val);
            optionButtons.forEach(b => b.classList.remove('selected'));
            btn.classList.add('selected');
            hiddenInput.value = val;
          }
        });
        optionButtons.push(btn);
        optionsGrid.appendChild(btn);
      }
      cardEl.appendChild(optionsGrid);
    } else if (field.type === 'number') {
      const numInput = document.createElement('input');
      numInput.type = 'number';
      numInput.name = field.id;
      numInput.className = 'question-input';
      numInput.step = 'any';
      if (field.required) numInput.required = true;
      cardEl.appendChild(numInput);
    } else if (field.type === 'textarea') {
      const ta = document.createElement('textarea');
      ta.name = field.id;
      ta.className = 'question-textarea';
      ta.rows = 2;
      cardEl.appendChild(ta);
    } else {
      const textInput = document.createElement('input');
      textInput.type = 'text';
      textInput.name = field.id;
      textInput.className = 'question-input';
      if (field.required) textInput.required = true;
      cardEl.appendChild(textInput);
    }
    form.appendChild(cardEl);
  }

  const submit = document.createElement('button');
  submit.type = 'submit';
  submit.className = 'primary question-submit-btn';
  submit.textContent = 'Submit Answer';
  form.appendChild(submit);

  form.addEventListener('submit', async event => {
    event.preventDefault();
    const answers = {};
    const answerLines = ['User answers:'];
    for (const field of fields) {
      const el = form.elements.namedItem(field.id);
      if (el) {
        let val = el.value;
        if (field.type === 'multiselect') {
          try { val = JSON.parse(val || '[]'); } catch { val = []; }
        }
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
        body: JSON.stringify({ project: currentProject, answers }),
      });
      clearQuestionArea();
      finalizeAllActiveCards();
      startNewTurnUsageBucket();
      addMessage(answerLines.join('\n'), 'user');
      getOrCreateRunCard(null, { optimistic: true });
      setThinking(true);
    } catch (error) {
      addMessage(error.message, 'error');
      submit.disabled = false;
    }
  });
  questionArea?.appendChild(form);
}

// ── SSE Stream ──
let eventSource = null;
let streamReconnectTimer = null;
let streamHadError = false;

async function syncAfterStreamReset() {
  if (!currentProject) return;
  await Promise.all([
    loadHistory(currentProject),
    loadCurrentState(),
    syncCurrentPreview(),
  ]);
}

function applyConversationReset() {
  flushStreamingRender();
  activeRunCard = null;
  clearActivity({ resetUsage: true });
  feed?.replaceChildren();
  clearQuestionArea();
  restoreEmptyState(feed, currentProject);
  setThinking(false);
}

function connectStream() {
  if (streamReconnectTimer) {
    clearTimeout(streamReconnectTimer);
    streamReconnectTimer = null;
  }
  if (eventSource) eventSource.close();
  const status = document.querySelector('#connection-status');
  eventSource = new EventSource('/api/stream');

  eventSource.addEventListener('error', () => {
    streamHadError = true;
    if (status) {
      status.classList.remove('connected');
      status.setAttribute('aria-label', 'Disconnected');
    }
    if (eventSource) {
      eventSource.close();
      eventSource = null;
    }
    if (!streamReconnectTimer) {
      streamReconnectTimer = setTimeout(() => {
        streamReconnectTimer = null;
        connectStream();
      }, 2000);
    }
  });

  eventSource.addEventListener('open', () => {
    if (status) {
      status.classList.add('connected');
      status.setAttribute('aria-label', 'Connected');
    }
    if (streamHadError) {
      streamHadError = false;
      syncAfterStreamReset().catch(err => console.error('stream reconnect sync failed', err));
    }
  });

  const handlers = {
    agent_status: data => {
      if (busyWithOtherProject && data.project === busyWithOtherProject) {
        if (['stopped', 'failed', 'completed'].includes(data.status)) setBusyWithOther(null);
      }
      if (data.project !== currentProject) return;
      const runId = data.run_id;
      const card = getActiveCard(runId);

      if (data.status === 'started') {
        setThinking(true);
        if (modelSelect && validModelKeys.has(data.model_key)) {
          modelSelect.value = data.model_key;
          syncEffortOptionsForModel(data.model_key);
        }
        const activeCard = card || getOrCreateRunCard(runId);
        if (activeCard) {
          const label = data.message ? data.message.replace(/\.+$/, '') : 'Planning CAD task';
          setCardState(activeCard, 'thinking', label);
        }
      } else if (data.status === 'reviewing' || data.status === 'verifying') {
        setThinking(true);
        if (card && card.dataset.state !== 'done' && card.dataset.state !== 'error' && card.dataset.state !== 'stopped') {
          setCardState(card, 'working', data.status === 'verifying' ? 'Verifying model' : 'Reviewing build');
        }
      } else if (data.status === 'running') {
        setThinking(true);
        if (card && card.dataset.state !== 'done' && card.dataset.state !== 'error' && card.dataset.state !== 'stopped') {
          setCardState(card, 'working', data.message || 'Working');
        }
      } else if (['stopped', 'failed', 'completed', 'waiting_for_user'].includes(data.status)) {
        setThinking(false);
        if (data.status === 'stopped' && card) setCardState(card, 'stopped', 'Stopped');
        else if (data.status === 'failed' && card) setCardState(card, 'error', 'Failed');
        if (data.status === 'completed') markActivityRecovered();
        else if (data.status === 'stopped') markActivityRecovered('stopped');
        else if (data.status === 'failed') markActivityRecovered('error');
        finalizeAllActiveCards();
        checkProviderBalance();
      }
    },
    question: data => {
      if (data.project !== currentProject) return;
      setThinking(false);
      showQuestion(data);
      finalizeAllActiveCards();
    },
    agent_stream_reset: data => {
      if (data.project !== currentProject) return;
      const runId = data.run_id;
      const card = getActiveCard(runId);
      if (card) {
        card._rawText = '';
        card._rawReasoning = '';
        card._streamedToolCalls = null;
        delete card.dataset.lastContentTurnId;
        delete card.dataset.lastReasoningTurnId;
        card.querySelector('.thought-disclosure')?.remove();
        renderAgentContent(card, '');
        setCardState(card, 'thinking', 'Retrying response...');
      }
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
      handleStreamingToolDelta(data.run_id || data.message_id, data);
    },
    agent_stream_end: data => {
      if (data.project !== currentProject) return;
      flushStreamingRender();
      const runId = data.run_id || data.message_id;
      const card = resolveRunCard(runId);
      if (card) {
        card._streamedToolCalls = null;
        const combined = stripEncryptedReasoning(card._rawReasoning || data.reasoning || '');
        if (combined) {
          card._rawReasoning = combined;
          ensureThoughtDisclosure(card, combined, Boolean(data.has_tools || data.continuing), false);
        }
        const text = data.continuing
          ? stripToolCallTags(data.message || '').trim()
          : stripToolCallTags(data.message || card._rawText || '').trim();
        if (text) {
          card._rawText = text;
          renderAgentContent(card, text);
        } else if (data.continuing) {
          card._rawText = '';
          renderAgentContent(card, '');
        }
        if (data.has_tools) {
          if (text || combined) {
            if (combined) {
              ensureThoughtDisclosure(card, combined, false, false);
            }
            setCardState(card, 'done');
            if (activeRunCard === card) activeRunCard = null;
          } else {
            card.remove();
            if (activeRunCard === card) activeRunCard = null;
          }
        } else if (data.continuing) {
          setCardState(card, 'thinking', 'Continuing');
        } else {
          setCardState(card, 'done');
        }
      }
    },
    agent_message: data => {
      if (data.project !== currentProject) return;
      finalizeRunCard(data.run_id || data.message_id, data.message || '', stripEncryptedReasoning(data.reasoning || ''));
      markActivityRecovered();
      setThinking(false);
      scrollFeedToBottom();
    },
    agent_usage: data => {
      if (data.project !== currentProject) return;
      updateUsagePill(data);
    },
    tool_status: data => {
      if (data.project !== currentProject) return;
      addToolMessage(data, { target: feed });
    },
    preview_updated: data => {
      if (data.project !== currentProject) return;
      if (data.preview_id) loadCurrentPreview(data.preview_id);
      else hideUnapprovedPreview('pending');
      if (!isCodeDirty()) loadModelCode();
    },
    revision_updated: data => {
      if (data.project !== currentProject) return;
      syncCurrentPreview();
      if (!isCodeDirty()) loadModelCode();
      if (isRevisionsDrawerOpen()) loadRevisions();
    },
    screenshot_updated: data => {
      if (data.project !== currentProject) return;
      const cacheLabel = data.cache_hit ? 'Cache hit' : 'Re-rendered';
      refreshLatestToolRow('cad_screenshot', { state: cacheLabel });
    },
    review_updated: data => {
      if (data.project !== currentProject) return;
      const verdict = activityLabel(data.status || 'inconclusive');
      refreshLatestToolRow('cad_review', { state: verdict });
    },
    agent_error: data => {
      if (data.project !== currentProject) return;
      const runId = data.run_id || data.message_id;
      const card = getActiveCard(runId);
      if (card) {
        if (!card._rawText && !card._rawReasoning) card.remove();
        else setCardState(card, 'error', 'Failed');
      }
      finalizeAllActiveCards();
      markActivityRecovered('error');
      addMessage(data.message || 'Agent error.', 'error', {retryable: true});
      setThinking(false);
    },
    agent_stopped: data => {
      if (busyWithOtherProject && (data.project === busyWithOtherProject || (data.affected_projects && data.affected_projects.includes(busyWithOtherProject)))) {
        setBusyWithOther(null);
      }
      if (data.project !== currentProject) return;
      const runId = data.run_id || data.message_id;
      const card = getActiveCard(runId);
      if (card) {
        if (!card._rawText && !card._rawReasoning) card.remove();
        else setCardState(card, 'stopped', 'Stopped');
      }
      finalizeAllActiveCards();
      markActivityRecovered('stopped');
      setThinking(false);
    },
    export_ready: async data => {
      if (data.project !== currentProject) return;
      if (data.format === '3mf') {
        try {
          const res = await fetch(`/api/projects/${encodeURIComponent(currentProject)}/export?format=3mf`);
          if (!res.ok) {
            const err = await res.json().catch(() => ({ error: '3MF download failed' }));
            addMessage(`3MF download failed: ${err.error || res.statusText}`, 'error');
            return;
          }
          const blob = await res.blob();
          triggerBlobDownload(blob, `${currentProject}.3mf`);
        } catch (err) {
          addMessage(`3MF download failed: ${err.message}`, 'error');
        }
      }
    },
    conversation_reset: data => {
      if (data?.project === currentProject) applyConversationReset();
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

  eventSource.addEventListener('stream_reset', () => {
    syncAfterStreamReset().catch(err => console.error('stream_reset sync failed', err));
  });
}

// ── Global Event Delegation ──
document.addEventListener('click', async event => {
  const exampleBtn = event.target.closest('.example-prompt');
  if (exampleBtn) {
    message.value = exampleBtn.dataset.prompt || message.value;
    autoResizeMessage();
    message.focus();
    return;
  }

  const thoughtCopyBtn = event.target.closest('.thought-copy-btn');
  if (thoughtCopyBtn) {
    event.preventDefault();
    event.stopPropagation();
    const disclosure = thoughtCopyBtn.closest('.thought-disclosure');
    const content = disclosure?.querySelector('.thought-content');
    const rawReasoning = content?.dataset.raw || '';
    if (rawReasoning.trim()) await copyTextToClipboard(rawReasoning, thoughtCopyBtn);
    return;
  }

  const messageCopyBtn = event.target.closest('.message-copy-btn');
  if (messageCopyBtn) {
    event.preventDefault();
    event.stopPropagation();
    const messageEl = messageCopyBtn.closest('.message');
    if (!messageEl) return;
    let rawText = messageEl._rawText || messageEl.dataset.raw || '';
    if (messageEl.classList.contains('agent')) rawText = stripToolCallTags(rawText);
    if (rawText.trim()) await copyTextToClipboard(rawText, messageCopyBtn);
    return;
  }

  const messageRetryBtn = event.target.closest('.message-retry-btn');
  if (messageRetryBtn) {
    event.preventDefault();
    event.stopPropagation();
    if (isAgentActive()) return;
    const messageEl = messageRetryBtn.closest('.message');
    if (!messageEl) return;
    messageRetryBtn.disabled = true;
    messageRetryBtn.classList.add('retrying');

    const allUserMessages = Array.from(feed?.querySelectorAll('.message.user') || []);
    const userIndex = allUserMessages.indexOf(messageEl);

    pruneFeedAfterUserMessage(userIndex);
    await handleRetry(null, { userIndex }, messageRetryBtn);
    return;
  }

  if (!event.target.closest('#chat-form')) {
    closeSlashPopup();
  }

  if (screenshotMenu && !screenshotMenu.hidden && !event.target.closest('.screenshot-combo')) {
    closeScreenshotMenu();
  }

  if (!event.target.closest('.attachment-item button')) {
    const item = event.target.closest('.attachment-thumb, .attachment-item');
    if (item) {
      const img = item.tagName === 'IMG' ? item : item.querySelector('img');
      const src = img?.currentSrc || img?.src || img?.getAttribute('src');
      if (src) openImageModal(src, img.alt || 'Reference Image');
    }
  }

  if (telemetryPopover && !telemetryPopover.hidden) {
    if (!telemetryPopover.contains(event.target) && !verificationPill?.contains(event.target)) {
      telemetryPopover.hidden = true;
    }
  }
});

// Centralized Escape Key Dispatcher
window.addEventListener('keydown', (event) => {
  if (event.key === 'Escape') {
    if (screenshotMenu && !screenshotMenu.hidden) {
      closeScreenshotMenu();
      screenshotMenuBtn?.focus();
      return;
    }
    if (imagePreviewModal && !imagePreviewModal.hidden) {
      closeImageModal();
      return;
    }
    if (hardwareModal && !hardwareModal.hidden) {
      closeHardwareModal();
      return;
    }
    if (isRevisionsDrawerOpen()) {
      closeRevisionsDrawer();
      return;
    }
    if (telemetryPopover && !telemetryPopover.hidden) {
      telemetryPopover.hidden = true;
      return;
    }
  }
});

// ── Provider Balance Pill ──
async function checkProviderBalance() {
  if (!balancePill || !balanceAmount) return;
  try {
    const data = await api('/api/provider/balance');
    if (data.supported && data.credits != null) {
      const credits = Number(data.credits);
      balanceAmount.textContent = `$${credits.toFixed(2)}`;
      balancePill.title = `${data.provider || 'Provider'} Balance: $${credits.toFixed(2)} ${data.currency || 'USD'}`;
      balancePill.classList.toggle('balance-low', credits < 1.0);
      balancePill.hidden = false;
    } else {
      balancePill.hidden = true;
    }
  } catch {}
}

// ── CAD Verification & Telemetry ──
function updateVerificationUI(verification, metrics) {
  if (!verificationPill || !telemetryBody) return;
  if (!verification && !metrics) {
    verificationPill.hidden = true;
    if (telemetryPopover) telemetryPopover.hidden = true;
    return;
  }
  verificationPill.hidden = false;

  let statusClass = 'status-info';
  let icon = 'ℹ';
  let label = 'Metrics';

  const findings = verification?.findings || [];
  const status = (verification?.status || '').toUpperCase();
  const criticalFindings = findings.filter(f => (f.category || '').toUpperCase() === 'CRITICAL');
  const warningFindings = findings.filter(f => {
    const cat = (f.category || '').toUpperCase();
    return cat === 'MAJOR' || cat === 'WARNING' || cat === 'WARN';
  });
  const infoFindings = findings.filter(f => (f.category || '').toUpperCase() === 'INFO');

  const criticalCount = verification?.critical_count ?? criticalFindings.length;
  const warningCount = verification?.warning_count ?? warningFindings.length;
  const infoCount = verification?.info_count ?? infoFindings.length;

  if (criticalCount > 0 || status === 'FAILED' || verification?.passed === false) {
    statusClass = 'status-fail';
    icon = '✕';
    label = criticalCount === 1 ? '1 Critical' : `${criticalCount} Critical`;
  } else if (warningCount > 0 || status === 'WARNING') {
    statusClass = 'status-warn';
    icon = '⚠';
    label = warningCount === 1 ? '1 Warning' : `${warningCount} Warnings`;
  } else if (status === 'PASSED' || verification?.passed === true || (metrics && criticalCount === 0)) {
    statusClass = 'status-pass';
    icon = '✓';
    label = 'Verified';
  }

  verificationPill.className = `verification-pill ${statusClass}`;
  if (verificationIcon) verificationIcon.textContent = icon;
  if (verificationLabel) verificationLabel.textContent = label;

  let html = '';
  if (verification) {
    const summary = verification.summary || '';
    const riskScore = verification.risk_score != null ? verification.risk_score : null;
    const riskLevel = verification.risk_level || '';
    html += `
      <div class="verification-summary-card">
        <div class="verification-summary-header">
          <span class="verification-summary-badge ${statusClass}">
            <span class="verification-icon">${icon}</span> ${escapeHTML(status || 'VERIFICATION')}
          </span>
          ${riskScore != null ? `<span class="verification-risk-badge">Risk ${riskScore}/100${riskLevel ? ` · ${escapeHTML(riskLevel)}` : ''}</span>` : ''}
        </div>
        ${summary ? `<div class="verification-summary-text">${escapeHTML(summary)}</div>` : ''}
      </div>
    `;
  }
  if (metrics) {
    let dims = '—';
    if (Array.isArray(metrics.dimensions_mm)) {
      dims = `${metrics.dimensions_mm.map(d => Number(d).toFixed(1)).join(' × ')} mm`;
    } else if (metrics.dimensions_mm && typeof metrics.dimensions_mm === 'object') {
      const { x = 0, y = 0, z = 0 } = metrics.dimensions_mm;
      dims = `${Number(x).toFixed(1)} × ${Number(y).toFixed(1)} × ${Number(z).toFixed(1)} mm`;
    } else if (metrics.bbox) {
      dims = `${(metrics.bbox.max_x - metrics.bbox.min_x).toFixed(1)} × ${(metrics.bbox.max_y - metrics.bbox.min_y).toFixed(1)} × ${(metrics.bbox.max_z - metrics.bbox.min_z).toFixed(1)} mm`;
    }
    const vol = metrics.volume_mm3 != null ? `${(metrics.volume_mm3 / 1000).toFixed(2)} cm³` : '—';
    const boundaryEdges = metrics.boundary_edges ?? metrics.boundary_edge_count ?? 0;
    const watertight = boundaryEdges === 0 || metrics.watertight === true;
    const watertightText = watertight ? '<span class="badge-success">Watertight</span>' : `<span class="badge-danger">${boundaryEdges} open edges</span>`;
    const componentsRaw = metrics.components ?? metrics.solid_count ?? metrics.connected_components;
    let components = '—';
    if (Array.isArray(componentsRaw)) {
      components = `${componentsRaw.length}`;
    } else if (componentsRaw != null && typeof componentsRaw !== 'object') {
      components = `${componentsRaw}`;
    }
    const triangleCount = metrics.triangle_count ?? metrics.triangles;
    const triangles = triangleCount != null ? Number(triangleCount).toLocaleString() : '—';

    html += `
      <div class="telemetry-section">
        <h4 class="telemetry-subtitle">Geometry & Mesh</h4>
        <dl class="telemetry-grid">
          <dt>Bounding Box</dt><dd><code>${escapeHTML(dims)}</code></dd>
          <dt>Volume</dt><dd>${vol}</dd>
          <dt>Watertight</dt><dd>${watertightText}</dd>
          <dt>Triangles</dt><dd>${triangles}</dd>
          <dt>Components</dt><dd>${components}</dd>
        </dl>
      </div>
    `;
  }

  const severityWeight = (cat) => {
    const c = (cat || '').toUpperCase();
    if (c === 'CRITICAL') return 1;
    if (c === 'MAJOR' || c === 'WARNING' || c === 'WARN') return 2;
    if (c === 'INFO') return 3;
    return 4;
  };

  const sortedFindings = [...findings].sort((a, b) => {
    const wA = severityWeight(a.category);
    const wB = severityWeight(b.category);
    if (wA !== wB) return wA - wB;
    const lineA = a.line ?? a.line_number ?? 999999;
    const lineB = b.line ?? b.line_number ?? 999999;
    return lineA - lineB;
  });

  if (sortedFindings.length > 0) {
    html += `
      <div class="telemetry-section">
        <div class="findings-section-header">
          <h4 class="telemetry-subtitle">Verification Findings (${sortedFindings.length})</h4>
          <button type="button" class="copy-findings-btn quiet" id="copy-all-findings-btn" title="Copy all findings to clipboard">
            <svg class="icon" viewBox="0 0 24 24" aria-hidden="true" focusable="false"><rect x="9" y="9" width="13" height="13" rx="2" fill="none" stroke="currentColor" stroke-width="1.6"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" fill="none" stroke="currentColor" stroke-width="1.6"/></svg>
            <span>Copy All</span>
          </button>
        </div>
        <div class="findings-list">
    `;
    for (let i = 0; i < sortedFindings.length; i++) {
      const f = sortedFindings[i];
      const cat = (f.category || 'INFO').toUpperCase();
      const catClass = cat === 'CRITICAL' ? 'finding-crit' : (cat === 'MAJOR' || cat === 'WARNING' || cat === 'WARN' ? 'finding-warn' : 'finding-info');
      const lineNum = f.line ?? f.line_number;
      const lineBtn = lineNum != null
        ? `<button type="button" class="finding-line-jump" data-line="${lineNum}" title="Jump to line ${lineNum}">Line ${lineNum}</button>`
        : '';
      const copyBtn = `
        <button type="button" class="finding-copy-btn quiet icon-only" data-index="${i}" title="Copy finding" aria-label="Copy finding">
          <svg class="icon" viewBox="0 0 24 24" aria-hidden="true" focusable="false"><rect x="9" y="9" width="13" height="13" rx="2" fill="none" stroke="currentColor" stroke-width="1.6"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" fill="none" stroke="currentColor" stroke-width="1.6"/></svg>
        </button>
      `;
      html += `
        <div class="finding-item ${catClass}">
          <div class="finding-header">
            <div class="finding-header-left">
              <span class="finding-badge">${escapeHTML(cat)}</span>
              ${lineBtn}
            </div>
            ${copyBtn}
          </div>
          <div class="finding-msg">${escapeHTML(f.message || '')}</div>
          ${f.impact ? `<div class="finding-impact">${escapeHTML(f.impact)}</div>` : ''}
          ${f.suggestion ? `<div class="finding-suggestion">💡 ${escapeHTML(f.suggestion)}</div>` : ''}
        </div>
      `;
    }
    html += `</div></div>`;
  } else if (verification) {
    html += `
      <div class="telemetry-section">
        <div class="findings-empty">No design or printability issues detected.</div>
      </div>
    `;
  }

  telemetryBody.innerHTML = html;

  const copyAllBtn = telemetryBody.querySelector('#copy-all-findings-btn');
  copyAllBtn?.addEventListener('click', async (e) => {
    e.stopPropagation();
    const textLines = [`CAD Verification Findings (${sortedFindings.length}):`];
    for (const f of sortedFindings) {
      const cat = (f.category || 'INFO').toUpperCase();
      const line = (f.line ?? f.line_number) != null ? ` (Line ${f.line ?? f.line_number})` : '';
      textLines.push(`- [${cat}]${line}: ${f.message || ''}`);
      if (f.impact) textLines.push(`  Impact: ${f.impact}`);
      if (f.suggestion) textLines.push(`  Fix: ${f.suggestion}`);
    }
    const ok = await copyTextToClipboard(textLines.join('\n'));
    if (ok) flashButtonFeedback(copyAllBtn, 'Copied!', { duration: 1500 });
  });

  telemetryBody.querySelectorAll('.finding-copy-btn').forEach(btn => {
    btn.addEventListener('click', async (e) => {
      e.stopPropagation();
      const idx = parseInt(btn.dataset.index, 10);
      const f = sortedFindings[idx];
      if (!f) return;
      const cat = (f.category || 'INFO').toUpperCase();
      const line = (f.line ?? f.line_number) != null ? ` (Line ${f.line ?? f.line_number})` : '';
      let text = `[${cat}]${line}: ${f.message || ''}`;
      if (f.impact) text += `\nImpact: ${f.impact}`;
      if (f.suggestion) text += `\nFix: ${f.suggestion}`;
      const ok = await copyTextToClipboard(text);
      if (ok) flashButtonFeedback(btn, '✓', { duration: 1200 });
    });
  });

  telemetryBody.querySelectorAll('.finding-line-jump').forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      const line = parseInt(btn.dataset.line, 10);
      if (!Number.isNaN(line)) {
        jumpToEditorLine(line, telemetryPopover);
      }
    });
  });
}

verificationPill?.addEventListener('click', (e) => {
  e.stopPropagation();
  if (telemetryPopover) telemetryPopover.hidden = !telemetryPopover.hidden;
});

telemetryClose?.addEventListener('click', () => {
  if (telemetryPopover) telemetryPopover.hidden = true;
});

// ── Hardware & Filament Preferences ──
// Server settings are in-memory; localStorage keeps the user's last saved
// choice so it can be re-applied after a server restart.
const HW_STORAGE_KEY = 'cad_hw_prefs';
let currentHardware = {
  printer: appConfig.userPreferences?.printer || '',
  nozzle: appConfig.userPreferences?.nozzle || '',
  filament: appConfig.userPreferences?.filament || 'PLA',
  additional_info: appConfig.userPreferences?.additional_info || '',
};

async function saveHardware(prefs) {
  currentHardware = await api('/api/settings/hardware', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(prefs),
  });
  try { localStorage.setItem(HW_STORAGE_KEY, JSON.stringify(currentHardware)); } catch {}
  updateHardwarePickerLabel();
  if (currentProject) {
    syncCurrentPreview().catch(() => {});
  }
}

function restoreSavedHardware() {
  let stored = null;
  try { stored = JSON.parse(localStorage.getItem(HW_STORAGE_KEY) || 'null'); } catch {}
  if (stored && Object.keys(currentHardware).some(k => (stored[k] ?? '') !== currentHardware[k])) {
    saveHardware(stored).catch(() => {});
  }
}

function updateHardwarePickerLabel() {
  if (!hardwarePickerLabel) return;
  const parts = [];
  if (currentHardware.filament) {
    const mat = String(currentHardware.filament).trim().split(/[\s(]/)[0];
    if (mat) parts.push(mat);
  }
  if (currentHardware.nozzle) {
    const rawNozzle = String(currentHardware.nozzle).trim();
    const match = rawNozzle.match(/\d+(?:\.\d+)?\s*mm/i);
    const noz = match ? match[0].replace(/\s+/g, '') : rawNozzle.split(/[\s(]/)[0];
    if (noz) parts.push(noz);
  }
  hardwarePickerLabel.textContent = parts.length > 0 ? parts.join(' · ') : 'FDM Profile';
  if (hardwarePickerBtn) {
    const specs = [currentHardware.filament, currentHardware.nozzle].filter(Boolean).join(', ');
    const titleParts = [];
    if (currentHardware.printer) titleParts.push(currentHardware.printer);
    if (specs) titleParts.push(`(${specs})`);
    if (currentHardware.additional_info) titleParts.push(`— ${currentHardware.additional_info}`);
    hardwarePickerBtn.title = titleParts.length > 0 ? titleParts.join(' ') : 'Printer & Filament Preferences';
  }
}

function openHardwareModal() {
  if (!hardwareModal) return;
  if (hwPrinter) hwPrinter.value = currentHardware.printer;
  if (hwFilament) hwFilament.value = currentHardware.filament || 'PLA';
  if (hwNozzle) hwNozzle.value = currentHardware.nozzle || '0.4mm';
  if (hwAdditional) hwAdditional.value = currentHardware.additional_info || '';
  hardwareModal.hidden = false;
  hardwareModal.setAttribute('aria-hidden', 'false');
  hwPrinter?.focus();
}

function closeHardwareModal() {
  if (!hardwareModal) return;
  hardwareModal.hidden = true;
  hardwareModal.setAttribute('aria-hidden', 'true');
}

hardwarePickerBtn?.addEventListener('click', openHardwareModal);
hardwareModalClose?.addEventListener('click', closeHardwareModal);
hardwareModalCancel?.addEventListener('click', closeHardwareModal);
hardwareModal?.addEventListener('click', (e) => {
  if (e.target === hardwareModal) closeHardwareModal();
});

hardwareForm?.addEventListener('submit', async (e) => {
  e.preventDefault();
  const updated = {
    printer: hwPrinter?.value?.trim() || '',
    filament: hwFilament?.value?.trim() || 'PLA',
    nozzle: hwNozzle?.value?.trim() || '0.4mm',
    additional_info: hwAdditional?.value?.trim() || '',
  };
  try {
    await saveHardware(updated);
    closeHardwareModal();
    flashButtonFeedback(hardwarePickerBtn, 'Saved!', { duration: 1000 });
  } catch (err) {
    addMessage(`Failed to update hardware preferences: ${err.message}`, 'error');
  }
});

// ── Application Initialization ──
(async function init() {
  autoResizeMessage();
  initModelAndEffort();
  updateHardwarePickerLabel();
  restoreSavedHardware();
  checkPreflight();
  checkProviderBalance();
  setInterval(checkProviderBalance, 60000);

  initEditor({
    getProject: () => currentProject,
    viewer,
    onBuildSuccess: (previewId) => loadCurrentPreview(previewId),
    addMessage,
  });

  initRevisions({
    getProject: () => currentProject,
    onRestoreSuccess: async () => {
      await syncCurrentPreview();
      await loadModelCode();
    },
    addMessage,
  });

  if (!currentProject) return;

  await Promise.all([
    loadHistory(currentProject),
    loadCurrentState(),
    syncCurrentPreview(),
    loadModelCode(),
  ]);
  connectStream();
})();
