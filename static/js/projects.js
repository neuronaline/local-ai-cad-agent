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
  const now = new Date();
  const diffMs = now - date;
  const diffDays = Math.floor(diffMs / 86400000);
  if (diffDays === 0) return 'Today';
  if (diffDays === 1) return 'Yesterday';
  if (diffDays < 7) return `${diffDays} days ago`;
  return date.toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric' });
}

function statusLabel(status) {
  const labels = { none: 'Empty', has_model: 'Has Model', finalized: 'Finalized' };
  return labels[status] || status;
}

function statusClass(status) {
  return `status-${status}`;
}

function cardTemplate(project) {
  return `
    <div class="project-card" data-name="${escapeHTML(project.name)}">
      <a href="/project/${encodeURIComponent(project.name)}" class="card-main">
        <h3 class="card-name">${escapeHTML(project.name)}</h3>
        <div class="card-meta">
          <span class="card-date">Created ${formatDate(project.created_at)}</span>
          <span class="card-date">Modified ${formatDate(project.modified_at)}</span>
        </div>
        <span class="model-badge ${statusClass(project.model_status)}">${statusLabel(project.model_status)}</span>
      </a>
      <div class="card-actions">
        <button class="icon-btn rename-btn" title="Rename" data-name="${escapeHTML(project.name)}" aria-label="Rename ${escapeHTML(project.name)}">✏️</button>
        <button class="icon-btn delete-btn" title="Delete" data-name="${escapeHTML(project.name)}" aria-label="Delete ${escapeHTML(project.name)}">🗑️</button>
      </div>
    </div>
  `;
}

function escapeHTML(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

async function loadProjects() {
  try {
    const data = await api('/api/projects');
    renderProjects(data.projects);
  } catch (error) {
    grid.innerHTML = `<div class="empty-projects"><p class="error">Failed to load projects: ${escapeHTML(error.message)}</p></div>`;
  }
}

function renderProjects(projects) {
  if (!projects.length) {
    grid.innerHTML = `
      <div class="empty-projects">
        <div class="empty-icon">◇</div>
        <h2>No projects yet</h2>
        <p>Create your first CAD project to get started.</p>
        <button id="empty-cta" class="primary">Get Started</button>
      </div>
    `;
    document.querySelector('#empty-cta')?.addEventListener('click', openNewProjectModal);
    return;
  }
  grid.innerHTML = projects.map(cardTemplate).join('');
  grid.querySelectorAll('.rename-btn').forEach(btn => {
    btn.addEventListener('click', () => openRenameModal(btn.dataset.name));
  });
  grid.querySelectorAll('.delete-btn').forEach(btn => {
    btn.addEventListener('click', () => openDeleteConfirm(btn.dataset.name));
  });
}

/* ── New Project Modal ── */

const newProjectModal = document.querySelector('#new-project-modal');
const newProjectForm = document.querySelector('#new-project-form');
const newProjectName = document.querySelector('#new-project-name');

function openNewProjectModal() {
  newProjectModal.classList.remove('hidden');
  newProjectName.value = '';
  newProjectName.focus();
}

document.querySelector('#new-project-btn').addEventListener('click', openNewProjectModal);
document.querySelector('#empty-cta')?.addEventListener('click', openNewProjectModal);
document.querySelector('#cancel-new-project').addEventListener('click', () => {
  newProjectModal.classList.add('hidden');
});

newProjectForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const rawName = newProjectName.value.trim();
  const name = rawName.toLowerCase().replace(/\s+/g, '-');
  if (!name) return;
  try {
    await api('/api/projects/new', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name }),
    });
    window.location.href = `/project/${encodeURIComponent(name)}`;
  } catch (error) {
    alert(error.message);
  }
});

/* ── Rename Modal ── */

const renameModal = document.querySelector('#rename-modal');
const renameForm = document.querySelector('#rename-form');
const renameName = document.querySelector('#rename-name');
let renameTarget = '';

function openRenameModal(name) {
  renameTarget = name;
  renameName.value = name;
  renameModal.classList.remove('hidden');
  renameName.focus();
  renameName.select();
}

document.querySelector('#cancel-rename').addEventListener('click', () => {
  renameModal.classList.add('hidden');
});

renameForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const rawName = renameName.value.trim();
  const newName = rawName.toLowerCase().replace(/\s+/g, '-');
  if (!newName || newName === renameTarget) {
    renameModal.classList.add('hidden');
    return;
  }
  try {
    await api(`/api/projects/${encodeURIComponent(renameTarget)}/rename`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: newName }),
    });
    renameModal.classList.add('hidden');
    await loadProjects();
  } catch (error) {
    alert(error.message);
  }
});

/* ── Delete Confirm ── */

const deleteModal = document.querySelector('#delete-confirm');
const deleteName = document.querySelector('#delete-project-name');
let deleteTarget = '';

function openDeleteConfirm(name) {
  deleteTarget = name;
  deleteName.textContent = name;
  deleteModal.classList.remove('hidden');
}

document.querySelector('#cancel-delete').addEventListener('click', () => {
  deleteModal.classList.add('hidden');
});

document.querySelector('#confirm-delete').addEventListener('click', async () => {
  try {
    await api(`/api/projects/${encodeURIComponent(deleteTarget)}`, { method: 'DELETE' });
    deleteModal.classList.add('hidden');
    await loadProjects();
  } catch (error) {
    alert(error.message);
  }
});

/* ── Init ── */

loadProjects();
