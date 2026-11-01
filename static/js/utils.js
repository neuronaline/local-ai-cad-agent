export function escapeHTML(str) {
  return String(str ?? '').replace(/[&<>"']/g, c => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;'
  }[c]));
}

export function generateUUID() {
  return typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function'
    ? crypto.randomUUID()
    : `${Date.now()}-${Math.random().toString(36).slice(2, 11)}`;
}

export async function api(url, options = {}) {
  const response = await fetch(url, options);
  const isJson = (response.headers.get('content-type') || '').includes('application/json');
  const body = isJson ? await response.json().catch(() => ({})) : await response.text();
  if (!response.ok) {
    const message = (isJson && body && body.error) || `Request failed (${response.status})`;
    throw new Error(message);
  }
  return body;
}

export function triggerBlobDownload(blob, filename) {
  const blobUrl = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = blobUrl;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(blobUrl), 10000);
}

export function flashButtonFeedback(button, tempText, { className = 'success', duration = 1500, resetText = null } = {}) {
  if (!button) return;
  const label = button.querySelector('.copy-label') || button.querySelector('span') || (button.childElementCount === 0 ? button : null);
  if (!label) return;
  const original = resetText ?? button._originalFeedbackText ?? label.textContent;
  button._originalFeedbackText = original;
  if (className) button.classList.add(className);
  label.textContent = tempText;
  if (button._feedbackTimeout) clearTimeout(button._feedbackTimeout);
  button._feedbackTimeout = setTimeout(() => {
    if (className) button.classList.remove(className);
    label.textContent = original;
    button._feedbackTimeout = null;
    button._originalFeedbackText = null;
  }, duration);
}

export async function copyTextToClipboard(text, buttonEl = null) {
  if (!text) return false;
  try {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
    } else {
      const textarea = document.createElement('textarea');
      textarea.value = text;
      textarea.style.position = 'fixed';
      textarea.style.opacity = '0';
      document.body.appendChild(textarea);
      textarea.focus();
      textarea.select();
      document.execCommand('copy');
      textarea.remove();
    }
    if (buttonEl) {
      flashButtonFeedback(buttonEl, 'Copied!', { className: 'copied', duration: 1500 });
    }
    return true;
  } catch (err) {
    console.error('Failed to copy to clipboard:', err);
    return false;
  }
}

export async function checkPreflight(bannerEl = document.querySelector('#preflight-banner')) {
  if (!bannerEl) return;
  try {
    const checks = await api('/api/preflight');
    const issues = [];
    if (!checks.api_key) {
      issues.push(`API key missing for provider <strong>${escapeHTML(checks.provider || 'configured')}</strong>${checks.gcloud_error ? ` (${escapeHTML(checks.gcloud_error)})` : ''}`);
    }
    if (!checks.openscad_installed) {
      issues.push('<strong>OpenSCAD</strong> binary not found on system PATH');
    }
    if (checks.workspace_writable === false) {
      issues.push('CAD workspace directory is not writable');
    }
    if (checks.bwrap_installed === false) {
      issues.push('<strong>bubblewrap (bwrap)</strong> sandbox is not installed');
    }
    if (checks.python_packages === false) {
      issues.push('Missing required Python packages (numpy, Pillow)');
    }
    if (issues.length > 0) {
      bannerEl.innerHTML = `
        <div class="preflight-content">
          <span class="preflight-icon" aria-hidden="true">⚠</span>
          <div class="preflight-text">
            <strong>System Preflight Warning:</strong>
            <ul class="preflight-list">
              ${issues.map(iss => `<li>${iss}</li>`).join('')}
            </ul>
          </div>
        </div>
      `;
      bannerEl.hidden = false;
    } else {
      bannerEl.hidden = true;
    }
  } catch {
    // Non-blocking
  }
}

