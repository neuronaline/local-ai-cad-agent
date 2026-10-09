const grid = document.querySelector('#projects-grid');

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || 'Request failed');
  return data;
}

function formatDate(iso) {
  if (!iso) return '—';
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return '—';
  const now = new Date();
  const diffMs = now - date;
  const diffDays = Math.floor(diffMs / 86400000);
  if (diffDays <= 0) return 'Today';
  if (diffDays === 1) return 'Yesterday';
  if (diffDays < 7) return `${diffDays} days ago`;
  return date.toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric' });
}

function statusLabel(status) {
  const labels = { none: 'Empty', has_model: 'Has Model', finalized: 'Finalized', stale: 'Stale' };
  return labels[status] || status;
}

function statusClass(status) {
  return `status-${status}`;
}

const searchInput = document.querySelector('#project-search');
const projectsCount = document.querySelector('#projects-count');
let allProjects = [];

function cardTemplate(project) {
  const name = escapeHTML(project.name);
  const encodedName = encodeURIComponent(project.name);
  const status = project.model_status || 'none';
  return `
    <div class="project-card" data-name="${name}">
      <div class="card-header">
        <span class="card-glyph" aria-hidden="true">◈</span>
        <span class="model-badge ${statusClass(status)}">${statusLabel(status)}</span>
      </div>
      <a href="/project/${encodedName}" class="card-body">
        <h3 class="card-name" title="${name}">${name}</h3>
        <div class="card-meta">
          <span class="card-date">Modified ${formatDate(project.modified_at)}</span>
          <span class="card-date card-date-sub">Created ${formatDate(project.created_at)}</span>
        </div>
      </a>
      <div class="card-footer">
        <a href="/project/${encodedName}" class="card-open" tabindex="-1" aria-hidden="true">
          <span>Open</span>
          <svg class="icon" viewBox="0 0 24 24" aria-hidden="true" focusable="false">
            <path d="M5 12h14M13 5l7 7-7 7" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
          </svg>
        </a>
        <div class="card-actions">
          <button type="button" class="icon-btn rename-btn" title="Rename" data-name="${name}" aria-label="Rename ${name}">
            <svg class="icon" viewBox="0 0 24 24" aria-hidden="true" focusable="false">
              <path d="M16.862 4.487l1.687-1.688a1.875 1.875 0 1 1 2.652 2.652L10.582 16.07a4.5 4.5 0 0 1-1.897 1.13L6 18l.8-2.685a4.5 4.5 0 0 1 1.13-1.897l10.932-10.931z" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>
              <path d="M19.5 7.125L16.875 4.5" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
          </button>
          <button type="button" class="icon-btn delete-btn" title="Delete" data-name="${name}" aria-label="Delete ${name}">
            <svg class="icon" viewBox="0 0 24 24" aria-hidden="true" focusable="false">
              <path d="M19 7l-.867 12.142A2 2 0 0 1 16.138 21H7.862a2 2 0 0 1-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 0 0-1-1h-4a1 1 0 0 0-1 1v3M4 7h16" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
          </button>
        </div>
      </div>
    </div>
  `;
}

function escapeHTML(str) {
  return String(str ?? '').replace(/[&<>"']/g, c => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  }[c]));
}

async function loadProjects() {
  try {
    const data = await api('/api/projects');
    allProjects = data.projects || [];
    filterAndRenderProjects();
  } catch (error) {
    grid.innerHTML = `<div class="empty-projects"><p class="error">Failed to load projects: ${escapeHTML(error.message)}</p></div>`;
  }
}

function filterAndRenderProjects() {
  const query = (searchInput?.value || '').trim().toLowerCase();
  const filtered = query
    ? allProjects.filter(p => p.name.toLowerCase().includes(query))
    : allProjects;

  if (projectsCount) {
    if (!allProjects.length) {
      projectsCount.textContent = '';
      projectsCount.hidden = true;
    } else {
      projectsCount.hidden = false;
      projectsCount.textContent = query && filtered.length !== allProjects.length
        ? `${filtered.length} / ${allProjects.length}`
        : String(allProjects.length);
    }
  }

  if (!allProjects.length) {
    grid.innerHTML = `
      <div class="empty-projects">
        <div class="empty-icon">◇</div>
        <h2>No projects yet</h2>
        <p>Create your first CAD project to get started.</p>
        <button id="empty-cta" class="primary">Get Started</button>
      </div>
    `;
    return;
  }

  if (!filtered.length) {
    grid.innerHTML = `
      <div class="empty-projects">
        <div class="empty-icon">🔍</div>
        <h2>No matching projects</h2>
        <p>No project found matching "<strong>${escapeHTML(query)}</strong>"</p>
        <button type="button" id="clear-search-btn" class="quiet">Clear Search</button>
      </div>
    `;
    return;
  }

  grid.innerHTML = filtered.map(cardTemplate).join('');
}

searchInput?.addEventListener('input', () => filterAndRenderProjects());

grid.addEventListener('click', (e) => {
  if (e.target.closest('#clear-search-btn')) {
    if (searchInput) {
      searchInput.value = '';
      searchInput.focus();
    }
    filterAndRenderProjects();
    return;
  }
  const renameBtn = e.target.closest('.rename-btn');
  if (renameBtn) return openRenameModal(renameBtn.dataset.name);
  const deleteBtn = e.target.closest('.delete-btn');
  if (deleteBtn) return openDeleteConfirm(deleteBtn.dataset.name);
  if (e.target.closest('#empty-cta')) return openNewProjectModal();
});

let lastFocusedElement = null;

function clearModalErrors() {
  document.querySelectorAll('.modal-error').forEach(el => {
    el.hidden = true;
    el.textContent = '';
  });
}

function showModalError(selector, message) {
  const el = document.querySelector(selector);
  if (!el) return;
  el.textContent = message;
  el.hidden = false;
}

function closeModal(modal) {
  modal.classList.add('hidden');
  clearModalErrors();
  if (lastFocusedElement && typeof lastFocusedElement.focus === 'function') {
    lastFocusedElement.focus();
    lastFocusedElement = null;
  }
}

document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') {
    const openModal = document.querySelector('.modal:not(.hidden)');
    if (openModal) closeModal(openModal);
  }
});

document.querySelectorAll('.modal').forEach(modal => {
  modal.addEventListener('click', (e) => {
    if (e.target === modal) {
      closeModal(modal);
    }
  });
});

/* ── New Project Modal ── */

const newProjectModal = document.querySelector('#new-project-modal');
const newProjectForm = document.querySelector('#new-project-form');
const newProjectName = document.querySelector('#new-project-name');

function openNewProjectModal() {
  lastFocusedElement = document.activeElement;
  clearModalErrors();
  newProjectModal.classList.remove('hidden');
  newProjectName.value = '';
  newProjectName.focus();
}

document.querySelector('#new-project-btn').addEventListener('click', openNewProjectModal);
document.querySelector('#cancel-new-project').addEventListener('click', () => {
  closeModal(newProjectModal);
});

newProjectForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  clearModalErrors();
  const rawName = newProjectName.value.trim();
  const name = rawName.toLowerCase().replace(/\s+/g, '-');
  if (!name) return;
  const submitBtn = newProjectForm.querySelector('button[type="submit"]');
  if (submitBtn) submitBtn.disabled = true;
  try {
    await api('/api/projects/new', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name }),
    });
    window.location.href = `/project/${encodeURIComponent(name)}`;
  } catch (error) {
    showModalError('#new-project-error', error.message);
  } finally {
    if (submitBtn) submitBtn.disabled = false;
  }
});

/* ── Rename Modal ── */

const renameModal = document.querySelector('#rename-modal');
const renameForm = document.querySelector('#rename-form');
const renameName = document.querySelector('#rename-name');
let renameTarget = '';

function openRenameModal(name) {
  lastFocusedElement = document.activeElement;
  clearModalErrors();
  renameTarget = name;
  renameName.value = name;
  renameModal.classList.remove('hidden');
  renameName.focus();
  renameName.select();
}

document.querySelector('#cancel-rename').addEventListener('click', () => {
  closeModal(renameModal);
});

renameForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  clearModalErrors();
  const rawName = renameName.value.trim();
  const newName = rawName.toLowerCase().replace(/\s+/g, '-');
  if (!newName || newName === renameTarget) {
    closeModal(renameModal);
    return;
  }
  const submitBtn = renameForm.querySelector('button[type="submit"]');
  if (submitBtn) submitBtn.disabled = true;
  try {
    await api(`/api/projects/${encodeURIComponent(renameTarget)}/rename`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: newName }),
    });
    closeModal(renameModal);
    await loadProjects();
  } catch (error) {
    showModalError('#rename-error', error.message);
  } finally {
    if (submitBtn) submitBtn.disabled = false;
  }
});

/* ── Delete Confirm ── */

const deleteModal = document.querySelector('#delete-confirm');
const deleteName = document.querySelector('#delete-project-name');
let deleteTarget = '';

function openDeleteConfirm(name) {
  lastFocusedElement = document.activeElement;
  clearModalErrors();
  deleteTarget = name;
  deleteName.textContent = name;
  deleteModal.classList.remove('hidden');
  document.querySelector('#cancel-delete')?.focus();
}

document.querySelector('#cancel-delete').addEventListener('click', () => {
  closeModal(deleteModal);
});

document.querySelector('#confirm-delete').addEventListener('click', async (e) => {
  clearModalErrors();
  const btn = e.currentTarget;
  if (btn) btn.disabled = true;
  try {
    await api(`/api/projects/${encodeURIComponent(deleteTarget)}`, { method: 'DELETE' });
    closeModal(deleteModal);
    await loadProjects();
  } catch (error) {
    showModalError('#delete-error', error.message);
  } finally {
    if (btn) btn.disabled = false;
  }
});

/* ── Init ── */

loadProjects();
