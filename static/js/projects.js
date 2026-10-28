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

const searchInput = document.querySelector('#project-search');
const projectsCount = document.querySelector('#projects-count');
let allProjects = [];

function cardTemplate(project) {
  const name = escapeHTML(project.name);
  const encodedName = encodeURIComponent(project.name);
  const runningBadge = project.is_running
    ? '<span class="running-badge" title="Agent task running"><span class="running-dot"></span> Running</span>'
    : '';
  return `
    <div class="project-card" data-name="${name}">
      <div class="card-header">
        <span class="card-glyph" aria-hidden="true">◈</span>
        ${runningBadge}
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
          <button type="button" class="icon-btn rename-btn" title="Rename" aria-label="Rename ${name}">
            <svg class="icon" viewBox="0 0 24 24" aria-hidden="true" focusable="false">
              <path d="M21.174 6.812a1 1 0 0 0-3.986-3.987L3.842 16.174a2 2 0 0 0-.5.83l-1.321 4.352a.5.5 0 0 0 .623.622l4.353-1.32a2 2 0 0 0 .83-.497z" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>
              <path d="m15 5 4 4" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
          </button>
          <button type="button" class="icon-btn delete-btn" title="Delete" aria-label="Delete ${name}">
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
  const card = e.target.closest('.project-card');
  const projectName = card?.dataset.name;
  if (e.target.closest('.rename-btn') && projectName) return openRenameModal(projectName);
  if (e.target.closest('.delete-btn') && projectName) return openDeleteConfirm(projectName);
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

function openModal(modal, focusTarget = null) {
  lastFocusedElement = document.activeElement;
  clearModalErrors();
  modal.classList.remove('hidden');
  if (focusTarget) {
    focusTarget.focus();
    if (typeof focusTarget.select === 'function') focusTarget.select();
  }
}

function closeModal(modal) {
  modal.classList.add('hidden');
  clearModalErrors();
  if (lastFocusedElement && typeof lastFocusedElement.focus === 'function') {
    lastFocusedElement.focus();
    lastFocusedElement = null;
  }
}

async function handleModalSubmit(submitBtn, errorSelector, actionAsync) {
  clearModalErrors();
  if (submitBtn) submitBtn.disabled = true;
  try {
    await actionAsync();
  } catch (error) {
    showModalError(errorSelector, error.message);
  } finally {
    if (submitBtn) submitBtn.disabled = false;
  }
}

document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') {
    const open = document.querySelector('.modal:not(.hidden)');
    if (open) closeModal(open);
  }
});

document.querySelectorAll('.modal').forEach(modal => {
  modal.addEventListener('click', (e) => {
    if (e.target === modal) closeModal(modal);
  });
});

document.addEventListener('click', (e) => {
  const cancelBtn = e.target.closest('#cancel-new-project, #cancel-rename, #cancel-delete');
  if (cancelBtn) {
    const modal = cancelBtn.closest('.modal');
    if (modal) closeModal(modal);
  }
});

/* ── New Project Modal ── */

const newProjectModal = document.querySelector('#new-project-modal');
const newProjectForm = document.querySelector('#new-project-form');
const newProjectName = document.querySelector('#new-project-name');

function openNewProjectModal() {
  newProjectName.value = '';
  openModal(newProjectModal, newProjectName);
}

document.querySelector('#new-project-btn')?.addEventListener('click', openNewProjectModal);

newProjectForm?.addEventListener('submit', async (e) => {
  e.preventDefault();
  const name = newProjectName.value.trim().replace(/\s+/g, ' ');
  if (!name) return;
  await handleModalSubmit(newProjectForm.querySelector('button[type="submit"]'), '#new-project-error', async () => {
    await api('/api/projects/new', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name }),
    });
    window.location.href = `/project/${encodeURIComponent(name)}`;
  });
});

/* ── Rename Modal ── */

const renameModal = document.querySelector('#rename-modal');
const renameForm = document.querySelector('#rename-form');
const renameName = document.querySelector('#rename-name');
let renameTarget = '';

function openRenameModal(name) {
  renameTarget = name;
  renameName.value = name;
  openModal(renameModal, renameName);
}

renameForm?.addEventListener('submit', async (e) => {
  e.preventDefault();
  const newName = renameName.value.trim().replace(/\s+/g, ' ');
  if (!newName || newName === renameTarget) {
    closeModal(renameModal);
    return;
  }
  await handleModalSubmit(renameForm.querySelector('button[type="submit"]'), '#rename-error', async () => {
    await api(`/api/projects/${encodeURIComponent(renameTarget)}/rename`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: newName }),
    });
    await loadProjects();
    closeModal(renameModal);
  });
});

/* ── Delete Confirm ── */

const deleteModal = document.querySelector('#delete-confirm');
const deleteName = document.querySelector('#delete-project-name');
let deleteTarget = '';

function openDeleteConfirm(name) {
  deleteTarget = name;
  deleteName.textContent = name;
  openModal(deleteModal, document.querySelector('#cancel-delete'));
}

document.querySelector('#confirm-delete')?.addEventListener('click', async (e) => {
  await handleModalSubmit(e.currentTarget, '#delete-error', async () => {
    await api(`/api/projects/${encodeURIComponent(deleteTarget)}`, { method: 'DELETE' });
    await loadProjects();
    closeModal(deleteModal);
  });
});

/* ── Init ── */

loadProjects();
