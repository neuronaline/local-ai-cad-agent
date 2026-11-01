import { api, escapeHTML } from './utils.js';

const historyDrawer = document.querySelector('#history-drawer');
const revisionsContent = document.querySelector('#revisions-content');
const historyBtn = document.querySelector('#history-btn');
const historyCloseBtn = document.querySelector('#history-close');

let currentGetProject = () => '';
let currentOnRestoreSuccess = async () => {};
let currentAddMessage = () => {};

export function isRevisionsDrawerOpen() {
  return Boolean(historyDrawer && !historyDrawer.hidden);
}

export function closeRevisionsDrawer() {
  if (historyDrawer) historyDrawer.hidden = true;
  historyBtn?.setAttribute('aria-pressed', 'false');
  historyBtn?.setAttribute('aria-expanded', 'false');
}

export function openRevisionsDrawer() {
  if (!historyDrawer) return;
  historyDrawer.hidden = false;
  historyBtn?.setAttribute('aria-pressed', 'true');
  historyBtn?.setAttribute('aria-expanded', 'true');
  loadRevisions();
}

export async function loadRevisions() {
  const project = currentGetProject();
  if (!project || !revisionsContent) return;
  revisionsContent.replaceChildren();
  const loading = document.createElement('div');
  loading.className = 'revision-loading';
  loading.textContent = 'Loading revisions…';
  revisionsContent.appendChild(loading);

  try {
    const data = await api(`/api/projects/${encodeURIComponent(project)}/revisions?limit=25`);
    const revisions = data.revisions || [];
    revisionsContent.replaceChildren();
    if (revisions.length === 0) {
      const empty = document.createElement('div');
      empty.className = 'revision-empty';
      empty.textContent = 'No revisions yet. Revisions are created as the model is generated or edited.';
      revisionsContent.appendChild(empty);
      return;
    }
    const list = document.createElement('ul');
    list.className = 'revision-list';
    for (const rev of revisions) {
      const item = document.createElement('li');
      item.className = 'revision-item';
      if (rev.is_active) item.classList.add('active');

      const header = document.createElement('div');
      header.className = 'revision-header';

      const revId = document.createElement('span');
      revId.className = 'revision-id';
      revId.textContent = (rev.id || '').slice(0, 8);

      const badgesWrap = document.createElement('div');
      badgesWrap.className = 'revision-badges';

      // Origin badge
      const kind = rev.origin?.kind || rev.origin?.source || '';
      let badgeType = 'default';
      let badgeText = kind || 'System';
      if (kind === 'agent' || kind === 'agent_edit') {
        badgeType = 'agent';
        badgeText = 'Agent';
      } else if (kind === 'user_editor' || kind === 'user_edit' || kind === 'user') {
        badgeType = 'user';
        badgeText = 'Manual';
      } else if (kind === 'build_repair' || kind === 'recovery' || kind === 'repair') {
        badgeType = 'repair';
        badgeText = 'Repair';
      } else if (kind === 'import' || kind === 'imported') {
        badgeType = 'import';
        badgeText = 'Import';
      } else if (kind === 'restore') {
        badgeType = 'default';
        badgeText = 'Restore';
      }
      const originBadge = document.createElement('span');
      originBadge.className = `rev-badge badge-origin badge-origin-${badgeType}`;
      originBadge.textContent = badgeText;
      badgesWrap.appendChild(originBadge);

      // LKG badge
      if (rev.is_last_known_good) {
        const lkgBadge = document.createElement('span');
        lkgBadge.className = 'rev-badge badge-lkg';
        lkgBadge.title = 'Last Known Good';
        lkgBadge.textContent = '★ LKG';
        badgesWrap.appendChild(lkgBadge);
      }

      // Status badge
      const revStatus = document.createElement('span');
      revStatus.className = 'revision-status';
      revStatus.dataset.status = rev.build_status || 'not_run';
      revStatus.textContent = rev.build_status || 'not_run';
      badgesWrap.appendChild(revStatus);

      header.append(revId, badgesWrap);
      item.appendChild(header);

      const meta = document.createElement('div');
      meta.className = 'revision-meta';
      meta.textContent = rev.created_at ? new Date(rev.created_at).toLocaleString() : '';
      item.appendChild(meta);

      // Metrics line if available
      if (rev.metrics) {
        const metricsEl = document.createElement('div');
        metricsEl.className = 'revision-metrics';
        let dims = '';
        if (Array.isArray(rev.metrics.dimensions_mm)) {
          dims = `${rev.metrics.dimensions_mm.map(d => Number(d).toFixed(1)).join('×')}mm`;
        } else if (rev.metrics.dimensions_mm && typeof rev.metrics.dimensions_mm === 'object') {
          const { x = 0, y = 0, z = 0 } = rev.metrics.dimensions_mm;
          dims = `${Number(x).toFixed(1)}×${Number(y).toFixed(1)}×${Number(z).toFixed(1)}mm`;
        }
        const vol = rev.metrics.volume_mm3 != null ? `${(rev.metrics.volume_mm3 / 1000).toFixed(1)}cm³` : '';
        const parts = [dims, vol].filter(Boolean);
        if (parts.length) {
          metricsEl.textContent = parts.join(' · ');
          item.appendChild(metricsEl);
        }
      }

      // Error snippet if failed
      if (rev.error) {
        const errEl = document.createElement('div');
        errEl.className = 'revision-error-box';
        errEl.textContent = rev.error;
        item.appendChild(errEl);
      }

      // Actions row
      const actions = document.createElement('div');
      actions.className = 'revision-actions';

      const restore = document.createElement('button');
      restore.type = 'button';
      restore.className = 'quiet';
      restore.textContent = rev.is_active ? 'Active' : 'Restore';
      restore.disabled = Boolean(rev.is_active);
      restore.addEventListener('click', async () => {
        restore.disabled = true;
        try {
          const res = await api(`/api/projects/${encodeURIComponent(project)}/revisions/${rev.id}/restore`, {method: 'POST'});
          if (res && res.ok === false) {
            currentAddMessage(`Restore partial failure: ${res.error}`, 'error');
          }
          closeRevisionsDrawer();
          await currentOnRestoreSuccess();
        } catch (error) {
          currentAddMessage(error.message, 'error');
        } finally {
          restore.disabled = Boolean(rev.is_active);
        }
      });
      actions.appendChild(restore);

      const diffBtn = document.createElement('button');
      diffBtn.type = 'button';
      diffBtn.className = 'quiet';
      diffBtn.textContent = 'Diff';
      actions.appendChild(diffBtn);

      const codeBtn = document.createElement('button');
      codeBtn.type = 'button';
      codeBtn.className = 'quiet';
      codeBtn.textContent = 'Code';
      actions.appendChild(codeBtn);

      item.appendChild(actions);

      // Expandable panel container
      const expandPanel = document.createElement('div');
      expandPanel.className = 'revision-expand-panel';
      expandPanel.hidden = true;
      item.appendChild(expandPanel);

      diffBtn.addEventListener('click', async () => {
        if (!expandPanel.hidden && expandPanel.dataset.mode === 'diff') {
          expandPanel.hidden = true;
          expandPanel.dataset.mode = '';
          return;
        }
        expandPanel.dataset.mode = 'diff';
        expandPanel.hidden = false;
        expandPanel.innerHTML = '<div class="panel-loading">Loading diff…</div>';
        try {
          const diffData = await api(`/api/projects/${encodeURIComponent(project)}/revisions/${rev.id}/diff`);
          const rawDiff = diffData.diff || '';
          if (!rawDiff.trim()) {
            expandPanel.innerHTML = '<div class="panel-empty">No diff (initial revision or identical).</div>';
            return;
          }
          const diffLines = rawDiff.split('\n');
          const pre = document.createElement('pre');
          pre.className = 'revision-diff-view';
          for (const line of diffLines) {
            const div = document.createElement('div');
            div.className = 'diff-line';
            if (line.startsWith('+') && !line.startsWith('+++')) div.classList.add('diff-add');
            else if (line.startsWith('-') && !line.startsWith('---')) div.classList.add('diff-del');
            else if (line.startsWith('@@')) div.classList.add('diff-hunk');
            else div.classList.add('diff-ctx');
            div.textContent = line;
            pre.appendChild(div);
          }
          expandPanel.replaceChildren(pre);
        } catch (err) {
          expandPanel.innerHTML = `<div class="panel-error">Error loading diff: ${escapeHTML(err.message)}</div>`;
        }
      });

      codeBtn.addEventListener('click', async () => {
        if (!expandPanel.hidden && expandPanel.dataset.mode === 'code') {
          expandPanel.hidden = true;
          expandPanel.dataset.mode = '';
          return;
        }
        expandPanel.dataset.mode = 'code';
        expandPanel.hidden = false;
        expandPanel.innerHTML = '<div class="panel-loading">Loading code…</div>';
        try {
          const revDetail = await api(`/api/projects/${encodeURIComponent(project)}/revisions/${rev.id}`);
          const source = revDetail.source || '';
          const pre = document.createElement('pre');
          pre.className = 'revision-code-view';
          const code = document.createElement('code');
          code.className = 'language-openscad';
          if (typeof hljs !== 'undefined' && hljs.getLanguage('openscad')) {
            code.innerHTML = hljs.highlight(source, { language: 'openscad' }).value;
          } else {
            code.textContent = source;
          }
          pre.appendChild(code);
          expandPanel.replaceChildren(pre);
        } catch (err) {
          expandPanel.innerHTML = `<div class="panel-error">Error loading code: ${escapeHTML(err.message)}</div>`;
        }
      });

      list.appendChild(item);
    }
    revisionsContent.appendChild(list);
  } catch (error) {
    revisionsContent.replaceChildren();
    const errorEl = document.createElement('div');
    errorEl.className = 'revision-empty';
    errorEl.textContent = `Failed to load revisions: ${error.message}`;
    revisionsContent.appendChild(errorEl);
  }
}

export function initRevisions({ getProject, onRestoreSuccess, addMessage }) {
  currentGetProject = getProject;
  currentOnRestoreSuccess = onRestoreSuccess;
  currentAddMessage = addMessage;

  historyBtn?.addEventListener('click', () => {
    if (isRevisionsDrawerOpen()) {
      closeRevisionsDrawer();
    } else {
      openRevisionsDrawer();
    }
  });

  historyCloseBtn?.addEventListener('click', closeRevisionsDrawer);
}

