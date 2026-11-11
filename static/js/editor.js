import { api, flashButtonFeedback, copyTextToClipboard } from './utils.js';

let currentGetProject = () => '';
let currentViewer = null;
let currentOnBuildSuccess = null;
let currentAddMessage = () => {};

const viewTab3d = document.querySelector('#view-tab-3d');
const viewTabCode = document.querySelector('#view-tab-code');
const codeDirtyDot = document.querySelector('#code-dirty-dot');
const codeView = document.querySelector('#code-view');
const viewerContainer = document.querySelector('#viewer');
const codeEditor = document.querySelector('#code-editor');
const codeGutter = document.querySelector('#code-gutter');
const codeStats = document.querySelector('#code-stats');
const saveCodeBtn = document.querySelector('#save-code-btn');
const btnCodeBuild = document.querySelector('#btn-code-build');
const buildBtn = document.querySelector('#build-btn');
const copyCodeBtn = document.querySelector('#copy-code-btn');

let codeDirty = false;
// True while this tab's own PUT is in flight: its ``revision_updated`` echo
// can arrive before the response clears ``codeDirty``.
let saveInFlight = false;
const REMOTE_CONFLICT_NOTE = ' · remote revision saved, unsaved edits kept';

export function isCodeDirty() {
  return codeDirty;
}

export function updateCodeGutter() {
  if (!codeGutter || !codeEditor) return;
  const lineCount = (codeEditor.value || '').split('\n').length;
  let nums = '';
  for (let i = 1; i <= lineCount; i++) {
    nums += `<div>${i}</div>`;
  }
  codeGutter.innerHTML = nums;
  codeGutter.scrollTop = codeEditor.scrollTop;
}

export function updateCodeStats() {
  if (!codeStats || !codeEditor) return;
  const lines = codeEditor.value ? codeEditor.value.split('\n').length : 0;
  const chars = codeEditor.value.length;
  codeStats.textContent = `${lines} lines · ${chars} chars${codeDirty ? ' (unsaved)' : ''}`;
  if (codeDirtyDot) codeDirtyDot.hidden = !codeDirty;
  updateCodeGutter();
}

/**
 * React to a ``revision_updated`` event: reload a clean editor, or keep the
 * user's unsaved edits and flag the conflict. Returns true only the first
 * time a conflict is flagged for the current dirty session.
 */
export function handleRemoteRevision() {
  if (saveInFlight) return false;
  if (!codeDirty) {
    loadModelCode();
    return false;
  }
  if (!codeStats || codeStats.textContent.endsWith(REMOTE_CONFLICT_NOTE)) return false;
  codeStats.textContent += REMOTE_CONFLICT_NOTE;
  return true;
}

export async function loadModelCode() {
  const project = currentGetProject();
  if (!project || !codeEditor) return;
  try {
    const data = await api(`/api/projects/${encodeURIComponent(project)}/code`);
    if (data.ok) {
      codeEditor.value = data.code || '';
      codeDirty = false;
      updateCodeStats();
    }
  } catch {
    // Non-blocking
  }
}

export async function saveModelCode() {
  const project = currentGetProject();
  if (!project || !codeEditor || !codeDirty) return true;
  saveInFlight = true;
  try {
    const res = await api(`/api/projects/${encodeURIComponent(project)}/code`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ code: codeEditor.value }),
    });
    if (res.ok) {
      codeDirty = false;
      updateCodeStats();
      return true;
    }
    return false;
  } catch (error) {
    currentAddMessage(`Failed to save code: ${error.message}`, 'error');
    return false;
  } finally {
    saveInFlight = false;
  }
}

export function setViewerMode(mode) {
  const isCode = mode === 'code';
  if (viewTab3d) {
    viewTab3d.classList.toggle('active', !isCode);
    viewTab3d.setAttribute('aria-selected', String(!isCode));
  }
  if (viewTabCode) {
    viewTabCode.classList.toggle('active', isCode);
    viewTabCode.setAttribute('aria-selected', String(isCode));
  }
  if (viewerContainer) viewerContainer.hidden = isCode;
  if (codeView) codeView.hidden = !isCode;
  if (!isCode) {
    currentViewer?.resize();
    return Promise.resolve();
  }
  codeEditor?.focus();
  updateCodeGutter();
  return codeDirty ? Promise.resolve() : loadModelCode();
}

export async function triggerBuild(sourceBtn = buildBtn) {
  const project = currentGetProject();
  if (!project) return currentAddMessage('Create or select a project first.', 'error');
  const btn = sourceBtn || buildBtn;
  if (btn?.disabled) return;
  if (btn) btn.disabled = true;
  const label = btn?.querySelector('span');
  const defaultLabel = label ? label.textContent : 'Build';
  if (label) label.textContent = 'Building…';
  try {
    if (!(await saveModelCode())) return;
    const res = await api(`/api/projects/${encodeURIComponent(project)}/build`, {
      method: 'POST',
    });
    if (res.ok) {
      if (res.preview_id && currentOnBuildSuccess) {
        await currentOnBuildSuccess(res.preview_id);
      }
      await loadModelCode();
      if (btn) flashButtonFeedback(btn, 'Built!', { duration: 1200, resetText: defaultLabel });
    } else {
      throw new Error(res.error || 'Build failed.');
    }
  } catch (error) {
    currentAddMessage(`CAD build failed: ${error.message}`, 'error');
  } finally {
    if (btn) btn.disabled = false;
    if (label && label.textContent === 'Building…') {
      label.textContent = defaultLabel;
    }
  }
}

export async function jumpToEditorLine(lineNumber, popoverToClose = null) {
  await setViewerMode('code');
  if (!codeEditor) return;
  const lines = codeEditor.value.split('\n');
  let charPos = 0;
  for (let i = 0; i < Math.min(lineNumber - 1, lines.length); i++) {
    charPos += lines[i].length + 1;
  }
  codeEditor.focus();
  const targetEnd = charPos + (lines[lineNumber - 1] ? lines[lineNumber - 1].length : 0);
  codeEditor.setSelectionRange(charPos, targetEnd);
  const lineHeight = 19.5;
  codeEditor.scrollTop = Math.max(0, (lineNumber - 4) * lineHeight);
  if (codeGutter) codeGutter.scrollTop = codeEditor.scrollTop;
  if (popoverToClose) popoverToClose.hidden = true;
}

export function initEditor({ getProject, viewer, onBuildSuccess, addMessage }) {
  currentGetProject = getProject;
  currentViewer = viewer;
  currentOnBuildSuccess = onBuildSuccess;
  currentAddMessage = addMessage;

  viewTab3d?.addEventListener('click', () => setViewerMode('3d'));
  viewTabCode?.addEventListener('click', () => setViewerMode('code'));

  codeEditor?.addEventListener('scroll', () => {
    if (codeGutter) codeGutter.scrollTop = codeEditor.scrollTop;
  });

  codeEditor?.addEventListener('input', () => {
    codeDirty = true;
    updateCodeStats();
  });

  codeEditor?.addEventListener('keydown', async event => {
    if (event.key === 'Tab') {
      event.preventDefault();
      const start = codeEditor.selectionStart;
      const end = codeEditor.selectionEnd;
      codeEditor.value = codeEditor.value.substring(0, start) + '  ' + codeEditor.value.substring(end);
      codeEditor.selectionStart = codeEditor.selectionEnd = start + 2;
      codeDirty = true;
      updateCodeStats();
    } else if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 's') {
      event.preventDefault();
      await handleSaveCodeAction();
    } else if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
      event.preventDefault();
      await triggerBuild(btnCodeBuild || buildBtn);
    }
  });

  async function handleSaveCodeAction() {
    const project = currentGetProject();
    if (!project || !codeEditor) return;
    if (await saveModelCode()) {
      flashButtonFeedback(saveCodeBtn, 'Saved!');
    } else {
      flashButtonFeedback(saveCodeBtn, 'Error', { className: 'error' });
    }
  }

  saveCodeBtn?.addEventListener('click', handleSaveCodeAction);

  copyCodeBtn?.addEventListener('click', async () => {
    if (!codeEditor?.value) return;
    const ok = await copyTextToClipboard(codeEditor.value);
    if (ok) flashButtonFeedback(copyCodeBtn, 'Copied!');
  });

  buildBtn?.addEventListener('click', () => triggerBuild(buildBtn));
  btnCodeBuild?.addEventListener('click', () => triggerBuild(btnCodeBuild));
}

