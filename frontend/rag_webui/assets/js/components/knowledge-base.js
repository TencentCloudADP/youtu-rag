// Knowledge Base Management Component

// Knowledge Base component state (namespaced to avoid conflicts)
let knowledgeBases = [];
let currentEditingKB = null;
let kbSearchQuery = '';
let kbSelectMode = false;
let selectedKBs = new Set();

// Initialize knowledge base page
function initKnowledgeBase() {
  console.log('[KB] Initializing knowledge base page...');

  // Wait for DOM to be ready
  const checkDOM = () => {
    const kbList = document.getElementById('kb-list');
    if (kbList) {
      console.log('[KB] DOM ready, loading data...');
      loadKnowledgeBases();
    } else {
      console.warn('[KB] DOM not ready, retrying in 50ms...');
      setTimeout(checkDOM, 50);
    }
  };

  checkDOM();

  // Listen for language changes and update dynamic text
  if (typeof i18n !== 'undefined') {
    i18n.onChange(() => {
      // Update select mode button text if in select mode
      if (kbSelectMode) {
        const selectBtn = document.getElementById('kb-select-mode-btn');
        if (selectBtn) {
          selectBtn.innerHTML = `
            <img src="assets/images/close.svg" alt="Cancel" style="width: 16px; height: 16px; margin-right: 6px; vertical-align: middle;">
            <span data-i18n="exit_select_mode">${t('exit_select_mode')}</span>
          `;
        }
      }
      // Update batch delete button text
      updateKnowledgeBaseBatchDeleteButton();
    });
  }
}

// Load knowledge bases
async function loadKnowledgeBases() {
  const container = document.getElementById('kb-list');
  if (!container) {
    console.error('[KB] kb-list container not found');
    return;
  }

  try {
    // Show loading state
    container.innerHTML = `
      <div style="grid-column: 1 / -1; text-align: center; padding: 48px;">
        <div class="spinner spinner-large" style="margin: 0 auto;"></div>
        <p style="margin-top: 16px; color: var(--gray-7);">Loading knowledge base list...</p>
      </div>
    `;

    console.log('[KB] Fetching knowledge bases from API...');
    const data = await API.getKnowledgeBases();
    console.log('[KB] API response:', data);

    // Handle different response formats
    knowledgeBases = Array.isArray(data) ? data : (data.knowledge_bases || []);
    console.log('[KB] Loaded', knowledgeBases.length, 'knowledge bases');

    renderKnowledgeBases();
  } catch (error) {
    console.error('[KB] Load error:', error);
    showToast('Failed to load knowledge base list: ' + error.message, 'error');
    renderKBEmptyState();
  }
}

// Folder color palette
const folderColors = ['blue', 'green', 'purple', 'orange', 'pink', 'cyan', 'yellow'];

// Get consistent color for a knowledge base (based on ID/name hash)
function getKBColor(kbId) {
  // Simple hash function to get consistent color for each KB
  let hash = 0;
  const str = String(kbId);
  for (let i = 0; i < str.length; i++) {
    hash = str.charCodeAt(i) + ((hash << 5) - hash);
  }
  const index = Math.abs(hash) % folderColors.length;
  return folderColors[index];
}

// Truncate text to max length
function truncateText(text, maxLength) {
  if (!text) return '';
  if (text.length <= maxLength) return text;
  return text.substring(0, maxLength);
}

// Update character count for input fields
function updateCharCount(inputId, countId, maxLength) {
  const input = document.getElementById(inputId);
  const counter = document.getElementById(countId);

  if (!input || !counter) return;

  const currentLength = input.value.length;
  counter.textContent = `${currentLength}/${maxLength}`;

  // Change color when approaching limit
  if (currentLength >= maxLength) {
    counter.style.color = '#dc2626'; // Red when at limit
  } else if (currentLength >= maxLength * 0.9) {
    counter.style.color = '#f59e0b'; // Orange when at 90%
  } else {
    counter.style.color = 'var(--primary-blue)'; // Blue otherwise
  }
}

// Render knowledge bases
function renderKnowledgeBases() {
  const container = document.getElementById('kb-list');
  if (!container) return;

  const filteredKBs = knowledgeBases.filter(kb => {
    if (!kbSearchQuery) return true;
    const query = kbSearchQuery.toLowerCase();
    return kb.name.toLowerCase().includes(query) ||
           (kb.description && kb.description.toLowerCase().includes(query));
  });

  if (filteredKBs.length === 0) {
    renderKBEmptyState();
    return;
  }

  container.innerHTML = filteredKBs.map(kb => {
    const kbId = kb.id || kb.name;
    const isSelected = selectedKBs.has(kbId);
    const colorClass = `color-${getKBColor(kbId)}`;

    // Truncate name and description
    const displayName = truncateText(kb.name, 80);
    const displayDesc = truncateText(kb.description || 'No description', 500);

    return `
    <div class="kb-card ${colorClass}" style="position: relative; ${isSelected ? 'border-color: #667eea; box-shadow: 0 4px 16px rgba(102, 126, 234, 0.2);' : ''}">
      ${kbSelectMode ? `
        <div class="kb-checkbox-container">
          <input
            type="checkbox"
            class="kb-checkbox"
            data-kb-id="${escapeHtml(kbId)}"
            ${isSelected ? 'checked' : ''}
            onclick="event.stopPropagation(); event.preventDefault(); toggleKBSelection('${escapeHtml(kbId)}');"
          >
        </div>
      ` : ''}
      <div class="kb-card-content" onclick="${kbSelectMode ? `toggleKBSelection('${escapeHtml(kbId)}')` : `viewKnowledgeBaseDetail('${escapeHtml(kbId)}')`}">
        <div class="kb-card-title">
          ${escapeHtml(displayName)}
        </div>
        <div class="kb-card-desc">
          ${escapeHtml(displayDesc)}
        </div>
        <div class="kb-card-stats">
          <div class="kb-card-stat">
            📄 <span>${kb.document_count || 0} Documents</span>
          </div>
          <div class="kb-card-stat">
            🗄️ <span>${kb.database_count || 0} Databases</span>
          </div>
        </div>
      </div>
      ${!kbSelectMode ? `
        <div class="kb-actions">
          <button class="btn btn-small kb-action-btn" onclick="event.stopPropagation(); editKnowledgeBase('${escapeHtml(kbId)}')">
            <img src="assets/images/edit-02.svg" alt="Edit" style="width: 16px; height: 16px; margin-right: 2px; vertical-align: middle;">
            Edit
          </button>
          <button class="btn btn-small kb-action-btn" onclick="event.stopPropagation(); deleteKnowledgeBase('${escapeHtml(kbId)}')">
            <img src="assets/images/delete-02.svg" alt="Delete" style="width: 16px; height: 16px; margin-right: 2px; vertical-align: middle;">
            Delete
          </button>
        </div>
      ` : ''}
    </div>
  `;
  }).join('');
}

// Render empty state
function renderKBEmptyState() {
  const container = document.getElementById('kb-list');
  if (!container) return;

  container.innerHTML = `
    <div style="grid-column: 1 / -1;">
      <div class="empty-state">
        <div class="icon">📚</div>
        <h3>No Knowledge Bases</h3>
        <p>Click the create button to create your first knowledge base</p>
      </div>
    </div>
  `;
}

// Handle KB search
const handleKBSearch = debounce((query) => {
  kbSearchQuery = query;
  renderKnowledgeBases();
}, 300);

// Create knowledge base
function createKnowledgeBase() {
  currentEditingKB = null;

  // Reset form
  document.getElementById('kb-modal-title').textContent = t('create_knowledge_base');
  document.getElementById('kb-name').value = '';
  document.getElementById('kb-description').value = '';

  // Reset character counters
  updateCharCount('kb-name', 'kb-name-count', 80);
  updateCharCount('kb-description', 'kb-desc-count', 500);

  showModal('kb-modal');
}

// Edit knowledge base
async function editKnowledgeBase(kbId) {
  currentEditingKB = kbId;

  try {
    const kb = await API.getKnowledgeBase(kbId);

    document.getElementById('kb-modal-title').textContent = t('edit_knowledge_base');
    document.getElementById('kb-name').value = kb.name || '';
    document.getElementById('kb-description').value = kb.description || '';

    // Update character counters
    updateCharCount('kb-name', 'kb-name-count', 80);
    updateCharCount('kb-description', 'kb-desc-count', 500);

    showModal('kb-modal');
  } catch (error) {
    showToast('Failed to load knowledge base information: ' + error.message, 'error');
  }
}

// View knowledge base details (old function - opens edit modal)
function viewKnowledgeBase(kbId) {
  editKnowledgeBase(kbId);
}

// View knowledge base detail page
function viewKnowledgeBaseDetail(kbId) {
  console.log('[KB] Navigating to knowledge base detail:', kbId);
  router.navigate(`/knowledge/${encodeURIComponent(kbId)}`);
}

// Save knowledge base
async function saveKnowledgeBase() {
  const name = document.getElementById('kb-name').value.trim();
  if (!name) {
    showToast('Please enter knowledge base name', 'warning');
    return;
  }

  const kbData = {
    name: name,
    description: document.getElementById('kb-description').value.trim()
  };

  try {
    const saveBtn = document.getElementById('save-kb-text');
    saveBtn.textContent = 'Saving...';

    if (currentEditingKB) {
      await API.updateKnowledgeBase(currentEditingKB, kbData);
      showToast('Knowledge base updated successfully', 'success');
    } else {
      await API.createKnowledgeBase(kbData);
      showToast('Knowledge base created successfully', 'success');
    }

    hideModal('kb-modal');
    loadKnowledgeBases();
  } catch (error) {
    showToast('Failed to save knowledge base: ' + error.message, 'error');
  } finally {
    const saveBtn = document.getElementById('save-kb-text');
    saveBtn.textContent = 'Save';
  }
}

// Build knowledge base
async function buildKnowledgeBase(kbId) {
  try {
    showModal('build-progress-modal');
    document.getElementById('build-status-text').textContent = 'Building knowledge base, please wait...';
    document.getElementById('build-progress-bar').style.width = '0%';
    document.getElementById('build-progress-text').textContent = '0%';

    // Simulate progress (replace with actual progress tracking if API supports it)
    let progress = 0;
    const progressInterval = setInterval(() => {
      progress += 10;
      if (progress <= 90) {
        document.getElementById('build-progress-bar').style.width = progress + '%';
        document.getElementById('build-progress-text').textContent = progress + '%';
      }
    }, 500);

    const result = await API.buildKnowledgeBase(kbId);

    clearInterval(progressInterval);
    document.getElementById('build-progress-bar').style.width = '100%';
    document.getElementById('build-progress-text').textContent = '100%';
    document.getElementById('build-status-text').textContent = 'Knowledge base build completed!';

    setTimeout(() => {
      hideModal('build-progress-modal');
    }, 1500);

    // Show detailed summary after a short delay 
    setTimeout(() => {
      showBuildSummary(result);
    }, 1600);

    showToast('Knowledge base built successfully', 'success');
  } catch (error) {
    hideModal('build-progress-modal');
    showToast('Failed to build knowledge base: ' + error.message, 'error');
  }
}

// Show build summary dialog
function showBuildSummary(result) {
  let message = `📊 Knowledge base build ${result.status === 'completed' ? 'completed' : result.status}!\n\n`;
  message += `📁 Total data sources: ${result.total_files || 0}\n`;
  message += `✅ Processed: ${result.processed_files || 0}\n`;

  // Show skipped files if any
  const skippedFiles = result.skipped_files || 0;
  if (skippedFiles > 0) {
    message += `⚡ Skipped build: ${skippedFiles} (unchanged)\n`;
  }

  message += `📝 Total chunks created: ${result.total_chunks || 0}\n`;

  if (result.errors && result.errors.length > 0) {
    message += `\n❌ Errors:\n${result.errors.join('\n')}`;
  }

  alert(message);
}

// Delete knowledge base
async function deleteKnowledgeBase(kbId) {
  const confirmed = await confirmDialog('Are you sure you want to delete this knowledge base? This action cannot be undone.');
  if (!confirmed) return;

  try {
    await API.deleteKnowledgeBase(kbId);
    showToast('Knowledge base deleted successfully', 'success');
    loadKnowledgeBases();
  } catch (error) {
    showToast('Failed to delete knowledge base: ' + error.message, 'error');
  }
}

// Toggle KB selection mode
function toggleKBSelectMode() {
  kbSelectMode = !kbSelectMode;
  selectedKBs.clear();

  const selectBtn = document.getElementById('kb-select-mode-btn');
  const batchDeleteBtn = document.getElementById('kb-batch-delete-btn');

  if (kbSelectMode) {
    // Entering select mode
    selectBtn.innerHTML = `
      <img src="assets/images/close.svg" alt="Cancel" style="width: 16px; height: 16px; margin-right: 6px; vertical-align: middle;">
      <span data-i18n="exit_select_mode">${t('exit_select_mode')}</span>
    `;
    batchDeleteBtn.style.display = 'inline-flex';
  } else {
    // Exiting select mode
    selectBtn.innerHTML = `
      <img src="assets/images/select.svg" alt="Select" style="width: 16px; height: 16px; margin-right: 6px; vertical-align: middle;">
      <span data-i18n="select">${t('select')}</span>
    `;
    batchDeleteBtn.style.display = 'none';
  }

  updateKnowledgeBaseBatchDeleteButton();
  renderKnowledgeBases();
}

// Toggle KB selection
function toggleKBSelection(kbId) {
  if (selectedKBs.has(kbId)) {
    selectedKBs.delete(kbId);
  } else {
    selectedKBs.add(kbId);
  }

  updateKnowledgeBaseBatchDeleteButton();

  // Find and update the specific checkbox using data attribute
  const checkbox = document.querySelector(`.kb-checkbox[data-kb-id="${kbId}"]`);
  if (checkbox) {
    const isSelected = selectedKBs.has(kbId);

    // Update checkbox state and visual class
    if (isSelected) {
      checkbox.setAttribute('checked', 'checked');
      checkbox.checked = true;
      checkbox.classList.add('is-checked');
    } else {
      checkbox.removeAttribute('checked');
      checkbox.checked = false;
      checkbox.classList.remove('is-checked');
    }

    // Force browser to recalculate styles
    void checkbox.offsetHeight;

    // Update card styling
    const card = checkbox.closest('.kb-card');
    if (card) {
      if (isSelected) {
        card.style.borderColor = '#667eea';
        card.style.boxShadow = '0 4px 16px rgba(102, 126, 234, 0.2)';
      } else {
        card.style.borderColor = '';
        card.style.boxShadow = '';
      }
    }
  }
}

// Update batch delete button (knowledge base)
function updateKnowledgeBaseBatchDeleteButton() {
  const batchDeleteText = document.getElementById('kb-batch-delete-text');
  if (batchDeleteText) {
    batchDeleteText.textContent = t('delete_selected', { count: selectedKBs.size });
  }
}

// Batch delete knowledge bases
async function batchDeleteKnowledgeBases() {
  if (selectedKBs.size === 0) {
    showToast('Please select knowledge bases to delete first', 'warning');
    return;
  }

  const confirmed = await confirmDialog(`Are you sure you want to delete the selected ${selectedKBs.size} knowledge bases? This action cannot be undone.`);
  if (!confirmed) return;

  try {
    const deletePromises = Array.from(selectedKBs).map(kbId =>
      API.deleteKnowledgeBase(kbId)
    );

    await Promise.all(deletePromises);

    showToast(`Successfully deleted ${selectedKBs.size} knowledge bases`, 'success');
    selectedKBs.clear();
    kbSelectMode = false;

    // Reset UI
    const selectBtn = document.getElementById('kb-select-mode-btn');
    const batchDeleteBtn = document.getElementById('kb-batch-delete-btn');
    selectBtn.innerHTML = `
      <img src="assets/images/select.svg" alt="Select" style="width: 16px; height: 16px; margin-right: 6px; vertical-align: middle;">
      Select
    `;
    batchDeleteBtn.style.display = 'none';

    loadKnowledgeBases();
  } catch (error) {
    showToast('Failed to batch delete knowledge bases: ' + error.message, 'error');
  }
}
