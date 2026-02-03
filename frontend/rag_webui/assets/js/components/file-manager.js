// File Management Component

let currentPage = 1;
let pageSize = 10;
let searchQuery = '';
let currentEditingFile = null;
let isSelectMode = false;
let selectedFiles = new Set();
let updateTimeSortOrder = null; // null: no sort, 'asc': ascending, 'desc': descending
let fileNameSortOrder = null; // null: no sort, 'asc': ascending, 'desc': descending
let cachedFiles = []; // Cache files for client-side sorting
let currentDeleteData = null; // Store current file deletion data

// ============================================
// Upload Progress Persistence (sessionStorage)
// ============================================

const STORAGE_KEY = 'upload_progress_tasks';

// Save upload task to sessionStorage
function saveUploadTask(taskId, filename) {
  try {
    const tasks = getStoredUploadTasks();
    tasks[taskId] = {
      filename: filename,
      timestamp: Date.now(),
      progressId: 'progress-' + Date.now() + '-' + Math.random().toString(36).slice(2, 11)
    };
    sessionStorage.setItem(STORAGE_KEY, JSON.stringify(tasks));
    console.log('[Storage] Saved task:', taskId, filename);
  } catch (error) {
    console.error('[Storage] Failed to save task:', error);
  }
}

// Remove task from sessionStorage
function removeUploadTask(taskId) {
  try {
    const tasks = getStoredUploadTasks();
    delete tasks[taskId];
    sessionStorage.setItem(STORAGE_KEY, JSON.stringify(tasks));
    console.log('[Storage] Removed task:', taskId);
  } catch (error) {
    console.error('[Storage] Failed to remove task:', error);
  }
}

// Get all stored upload tasks
function getStoredUploadTasks() {
  try {
    const stored = sessionStorage.getItem(STORAGE_KEY);
    return stored ? JSON.parse(stored) : {};
  } catch (error) {
    console.error('[Storage] Failed to parse stored tasks:', error);
    return {};
  }
}

// ============================================

// Get API base URL from config
function getApiBase() {
  return window.APP_CONFIG?.API_BASE || '';
}

// Initialize file management page
function initFileManagement() {
  // Set default sort to update time descending (newest first)
  updateTimeSortOrder = 'desc';
  updateSortIndicators();

  loadFileList();

  // Add metadata import input change listener
  const metadataImportInput = document.getElementById('metadata-import-input');
  if (metadataImportInput) {
    metadataImportInput.addEventListener('change', async (e) => {
      const file = e.target.files[0];
      if (!file) return;

      // Validate file type
      if (!file.name.endsWith('.xlsx') && !file.name.endsWith('.xls')) {
        showToast(t('toast_select_excel_file'), 'error');
        e.target.value = ''; // Reset input
        return;
      }

      await importMetadataFromExcel(file);
      e.target.value = ''; // Reset input for next import
    });
  }

  // Listen for language changes and update dynamic text
  if (typeof i18n !== 'undefined') {
    i18n.onChange(() => {
      // Update select mode button text if in select mode
      if (isSelectMode) {
        const selectModeBtn = document.getElementById('select-mode-btn');
        if (selectModeBtn) {
          selectModeBtn.innerHTML = `
            <img src="assets/images/close.svg" alt="Cancel" style="width: 16px; height: 16px; margin-right: 6px; vertical-align: middle;">
            <span data-i18n="exit_select_mode">${t('exit_select_mode')}</span>
          `;
        }
      }
      // Update batch delete button text
      updateFileManagerBatchDeleteButton();
    });
  }

  // Recover any in-progress uploads from previous session
  recoverUploadProgress();
}

// ============================================
// Upload Progress Recovery (on page load)
// ============================================

async function recoverUploadProgress() {
  const tasks = getStoredUploadTasks();
  const taskIds = Object.keys(tasks);

  if (taskIds.length === 0) {
    console.log('[Recovery] No tasks to recover');
    return;
  }

  console.log(`[Recovery] Found ${taskIds.length} task(s) to recover:`, taskIds);

  const apiBase = getApiBase();
  const now = Date.now();
  const MAX_AGE_MS = 2 * 60 * 60 * 1000; // 2 hours

  for (const taskId of taskIds) {
    const taskData = tasks[taskId];
    const age = now - taskData.timestamp;

    // Skip tasks older than 2 hours (likely stale)
    if (age > MAX_AGE_MS) {
      console.log(`[Recovery] Skipping stale task (${Math.round(age / 60000)} min old):`, taskId);
      removeUploadTask(taskId);
      continue;
    }

    try {
      // Check if task still exists in backend
      const response = await fetch(`${apiBase}/api/files/upload-progress/${taskId}`);

      if (!response.ok) {
        if (response.status === 404) {
          console.log('[Recovery] Task not found in backend (completed or expired):', taskId);
          removeUploadTask(taskId);
          continue;
        }
        throw new Error(`HTTP ${response.status}`);
      }

      const progress = await response.json();

      // If task already completed/failed, clean up
      if (progress.status === 'completed' || progress.status === 'failed') {
        console.log(`[Recovery] Task already ${progress.status}:`, taskId);
        removeUploadTask(taskId);

        // If completed, refresh file list to show new file
        if (progress.status === 'completed') {
          currentPage = 1;
          loadFileList();
        }
        continue;
      }

      // Task is still in progress - recreate progress bar and resume polling
      console.log(`[Recovery] Resuming task (${progress.progress}%):`, taskId, taskData.filename);

      // Create progress bar with stored progressId
      const progressId = taskData.progressId;
      createProgressBar(progressId, taskData.filename);

      // Update to current progress
      updateProgressBar(progressId, progress.progress, progress.message);

      // Resume polling
      await pollProgress(taskId, progressId);

    } catch (error) {
      console.error('[Recovery] Error recovering task:', taskId, error);
      // Keep task in storage for next refresh attempt
    }
  }
}

// Load file list
async function loadFileList() {
  try {
    const data = await API.getFiles(currentPage, pageSize, searchQuery);
    cachedFiles = data.files || [];

    // Apply sorting if enabled (only for user-initiated sorts)
    // Note: API.getFiles() already sorts by last_modified desc by default
    if (fileNameSortOrder) {
      sortFilesByName();
    } else if (updateTimeSortOrder && updateTimeSortOrder !== 'desc') {
      // Only re-sort if user wants ascending order (desc is already default)
      sortFilesByUpdateTime();
    }

    renderFileList(cachedFiles);
    renderPagination(data.total || 0, data.page || 1, data.page_size || 10);
  } catch (error) {
    showToast('Failed to load file list: ' + error.message, 'error');
    renderEmptyState();
  }
}

// Render file list
function renderFileList(files) {
  const tbody = document.getElementById('file-list-body');
  if (!tbody) return;

  if (files.length === 0) {
    renderEmptyState();
    return;
  }

  console.log('[renderFileList] Rendering', files.length, 'files, isSelectMode:', isSelectMode);

  tbody.innerHTML = files.map(file => {
    const metadata = file.metadata || {};
    const filename = file.filename || file.name;
    const isChecked = selectedFiles.has(filename);

    // Build metadata badges
    let badges = [];
    if (metadata.char_length) {
      badges.push(`<span class="badge" style="background: #f0f0f0; color: #666; box-shadow: 0 1px 2px rgba(0,0,0,0.1);">📏 ${escapeHtml(String(metadata.char_length))} chars</span>`);
    }
    if (metadata.publish_date) {
      badges.push(`<span class="badge" style="background: #f0f0f0; color: #666; box-shadow: 0 1px 2px rgba(0,0,0,0.1);">publish_date: ${escapeHtml(metadata.publish_date)}</span>`);
    }
    if (metadata.key_timepoints) {
      badges.push(`<span class="badge" style="background: #f0f0f0; color: #666; box-shadow: 0 1px 2px rgba(0,0,0,0.1);">key_timepoints: ${escapeHtml(metadata.key_timepoints.replace(/;/g, ', '))}</span>`);
    }
    if (file.size) {
      badges.push(`<span class="badge" style="background: #f0f0f0; color: #666; box-shadow: 0 1px 2px rgba(0,0,0,0.1);">💾 ${formatFileSize(file.size)}</span>`);
    }

    // Format update time for display in separate column
    let updateTimeText = '-';
    if (file.last_modified || file.created_at || file.upload_time) {
      const date = new Date(file.last_modified || file.created_at || file.upload_time);
      updateTimeText = date.toLocaleString('zh-CN', {
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit'
      });
    }

    // Check if we need to show viewer dropdown
    const hasOcr = metadata.ocr_processed === 'ocr_success';
    const hasChunk = metadata.chunk_processed === 'chunk_success';

    // Build dropdown menu items (always show Details)
    const iconStyle = "width: 16px; height: 16px; margin-right: 8px; vertical-align: middle; opacity: 0.7;";
    let dropdownItems = `<button class="dropdown-menu-item" onclick="handleViewerClick(event, 'details', '${escapeHtml(filename)}')"><img src="assets/images/details.svg" alt="Details" style="${iconStyle}"><span style="font-weight: 600;">Details</span></button>`;

    if (hasOcr) {
      dropdownItems += `<button class="dropdown-menu-item" onclick="handleViewerClick(event, 'ocr', '${escapeHtml(filename)}')"><img src="assets/images/ocr.svg" alt="OCR" style="${iconStyle}"><span style="font-weight: 600;">View OCR</span></button>`;
    }
    if (hasChunk) {
      dropdownItems += `<button class="dropdown-menu-item" onclick="handleViewerClick(event, 'chunk', '${escapeHtml(filename)}')"><img src="assets/images/chunklevel.svg" alt="Chunk" style="${iconStyle}"><span style="font-weight: 600;">View Chunk</span></button>`;
    }

    return `
    <tr>
      ${isSelectMode ? `<td style="vertical-align: middle; text-align: center;">
        <input type="checkbox" class="file-checkbox" data-filename="${escapeHtml(filename)}" ${isChecked ? 'checked' : ''} style="cursor: pointer; width: 18px; height: 18px;">
      </td>` : ''}
      <td>
        <div>
          <strong style="cursor: pointer; color: var(--primary-blue); transition: color var(--transition-fast);"
                  onmouseover="this.style.textDecoration='underline'"
                  onmouseout="this.style.textDecoration='none'"
                  onclick="openFileContentSidebar('${escapeHtml(filename)}')">${escapeHtml(filename)}</strong>
          ${metadata.summary ? `<div style="color: var(--gray-8); font-size: 13px; margin-top: 6px; line-height: 1.5;">📝 ${escapeHtml(metadata.summary)}</div>` : ''}
          ${badges.length > 0 ? `<div style="margin-top: 6px; display: flex; gap: 6px; flex-wrap: wrap;">${badges.join('')}</div>` : ''}
        </div>
      </td>
      <td style="vertical-align: top; color: var(--gray-7); font-size: 14px;">
        ${updateTimeText}
      </td>
      <td style="vertical-align: top;">
        <div style="display: flex; gap: 6px; flex-wrap: wrap;">
          <button class="btn btn-small btn-secondary" onclick="downloadFile('${escapeHtml(filename)}', ${hasOcr}, ${hasChunk})" title="Download">
            <img src="assets/images/download.svg" alt="Download" style="width: 16px; height: 16px; filter: grayscale(100%) brightness(0.6);">
          </button>
          <button class="btn btn-small btn-secondary" onclick="deleteFile('${escapeHtml(filename)}')" title="Delete">
            <img src="assets/images/delete.svg" alt="Delete" style="width: 16px; height: 16px; filter: grayscale(100%) brightness(0.6);">
          </button>
          <div class="dropdown">
            <button class="btn btn-small btn-secondary" onclick="toggleDropdown(event, this)" title="More Options">
              <img src="assets/images/more-vertical.svg" alt="More" style="width: 16px; height: 16px;">
            </button>
            <div class="dropdown-menu">
              ${dropdownItems}
            </div>
          </div>
        </div>
      </td>
    </tr>
  `;
  }).join('');

  // Attach event listeners to all checkboxes
  setTimeout(() => {
    const checkboxes = document.querySelectorAll('.file-checkbox');
    checkboxes.forEach((checkbox) => {
      // Remove inline onchange attribute to prevent conflicts
      checkbox.removeAttribute('onchange');

      // Add event listener
      checkbox.addEventListener('change', function() {
        toggleFileCheckbox(this);
      });
    });
  }, 0);
}

// Render empty state
function renderEmptyState() {
  const tbody = document.getElementById('file-list-body');
  if (!tbody) return;

  const colspan = isSelectMode ? 4 : 3;
  tbody.innerHTML = `
    <tr>
      <td colspan="${colspan}">
        <div class="empty-state">
          <div class="icon">📁</div>
          <h3>No files yet</h3>
          <p>Click upload button to start</p>
        </div>
      </td>
    </tr>
  `;
}

// Get status badge HTML
function getStatusBadge(status) {
  const statusMap = {
    'pending': { text: 'Pending', class: 'badge-warning' },
    'processing': { text: 'Processing', class: 'badge-info' },
    'completed': { text: 'Completed', class: 'badge-success' },
    'failed': { text: 'Failed', class: 'badge-error' }
  };

  const statusInfo = statusMap[status] || statusMap['pending'];
  return `<span class="badge ${statusInfo.class}">${statusInfo.text}</span>`;
}

// Render pagination
function renderPagination(total, page, size) {
  const paginationEl = document.getElementById('pagination');
  const paginationContainer = document.getElementById('pagination-container');
  const paginationInfo = document.getElementById('pagination-info');

  if (!paginationEl || !paginationContainer) return;

  const totalPages = Math.ceil(total / size);

  // Show/hide pagination container based on total items
  if (total <= 0) {
    paginationContainer.style.display = 'none';
    return;
  } else {
    paginationContainer.style.display = 'flex';
  }

  // Update pagination info
  const startItem = total > 0 ? (page - 1) * size + 1 : 0;
  const endItem = Math.min(page * size, total);
  if (paginationInfo) {
    paginationInfo.textContent = `Showing ${startItem}-${endItem} of ${total} items`;
  }

  // Update page size select to match current size
  const pageSizeSelect = document.getElementById('page-size-select');
  if (pageSizeSelect && pageSizeSelect.value !== String(size)) {
    pageSizeSelect.value = String(size);
  }

  // If only one page, show simplified pagination
  if (totalPages <= 1) {
    paginationEl.innerHTML = '<span style="color: var(--gray-6); font-size: 13px;">Page 1 of 1</span>';
    return;
  }

  let html = `
    <button ${page <= 1 ? 'disabled' : ''} onclick="goToPage(${page - 1})">Previous</button>
  `;

  // Show page numbers
  for (let i = 1; i <= totalPages; i++) {
    if (i === 1 || i === totalPages || (i >= page - 2 && i <= page + 2)) {
      html += `
        <button class="${i === page ? 'active' : ''}" onclick="goToPage(${i})">${i}</button>
      `;
    } else if (i === page - 3 || i === page + 3) {
      html += '<button disabled>...</button>';
    }
  }

  html += `
    <button ${page >= totalPages ? 'disabled' : ''} onclick="goToPage(${page + 1})">Next</button>
  `;

  paginationEl.innerHTML = html;
}

// Go to page
function goToPage(page) {
  currentPage = page;
  loadFileList();
}

// Change page size
function changePageSize() {
  const select = document.getElementById('page-size-select');
  if (!select) return;

  pageSize = parseInt(select.value);
  currentPage = 1; // Reset to first page when changing page size
  loadFileList();
}

// Handle file search
const handleFileSearch = debounce((query) => {
  searchQuery = query;
  currentPage = 1;
  // Keep sort state when searching
  loadFileList();
}, 500);

// Handle file upload
async function handleFileUpload(files) {
  if (!files || files.length === 0) return;

  // Check for duplicate files first
  const filesToUpload = [];
  const apiBase = getApiBase();

  for (let file of files) {
    try {
      // Check if file already exists in MinIO
      const response = await fetch(`${apiBase}/api/files/check-exists/${encodeURIComponent(file.name)}`);
      const result = await response.json();

      if (result.exists) {
        // File exists, ask user for confirmation
        const message =
          `${t('file_duplicate_warning')}\n\n` +
          `${t('file_exists_message', { filename: file.name })}\n\n` +
          `${t('file_size_label')}: ${formatFileSize(result.size || 0)}\n` +
          `${t('last_modified_label')}: ${result.last_modified ? new Date(result.last_modified).toLocaleString() : t('loading')}\n\n` +
          `${t('overwrite_confirm')}`;

        const shouldOverwrite = await confirmDialog(message);

        if (shouldOverwrite) {
          filesToUpload.push(file);
        } else {
          showToast(t('file_skipped', { filename: file.name }), 'info');
        }
      } else {
        // File doesn't exist, add to upload list
        filesToUpload.push(file);
      }
    } catch (error) {
      console.error(`Error checking file existence for ${file.name}:`, error);
      // If check fails, proceed with upload anyway
      filesToUpload.push(file);
    }
  }

  if (filesToUpload.length === 0) {
    showToast(t('no_files_to_upload'), 'info');
    return;
  }

  // Upload all files concurrently (each with independent progress bar)
  const uploadPromises = [];
  for (let file of filesToUpload) {
    uploadPromises.push(fileManagerUploadFile(file));
  }

  await Promise.allSettled(uploadPromises);
  // Note: File list is automatically refreshed after each file completes
}

// Upload single file with progress tracking
async function fileManagerUploadFile(file) {
  // Generate unique progress bar ID
  const progressId = 'progress-' + Date.now() + '-' + Math.random().toString(36).slice(2, 11);

  try {
    // Create progress bar UI
    createProgressBar(progressId, file.name);

    // Create FormData and upload
    const formData = new FormData();
    formData.append('file', file);

    const apiBase = getApiBase();
    const response = await fetch(`${apiBase}/api/files/upload-with-progress`, {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error('[Upload] Error response:', errorText);
      throw new Error(`Upload request failed: ${response.status}`);
    }

    const result = await response.json();

    // Save task to sessionStorage for recovery after page refresh
    saveUploadTask(result.task_id, file.name);

    // Immediately update progress bar to avoid showing static "Preparing..."
    updateProgressBar(progressId, 5, 'Task created, processing...');

    // Start polling progress
    await pollProgress(result.task_id, progressId);

  } catch (error) {
    console.error('[Upload] Error:', error);
    removeProgressBar(progressId);
    showToast(`Failed to upload "${file.name}": ${error.message}`, 'error');
  }
}

// Drag and drop handlers
function handleDragOver(e) {
  e.preventDefault();
  e.stopPropagation();
  e.currentTarget.classList.add('dragover');
}

function handleDragLeave(e) {
  e.preventDefault();
  e.stopPropagation();
  e.currentTarget.classList.remove('dragover');
}

function handleDrop(e) {
  e.preventDefault();
  e.stopPropagation();
  e.currentTarget.classList.remove('dragover');

  const files = e.dataTransfer.files;
  handleFileUpload(files);
}

// Edit metadata
async function editMetadata(filename) {
  currentEditingFile = filename;

  try {
    const metadata = await API.getFileMetadata(filename);

    document.getElementById('metadata-filename').value = filename;
    document.getElementById('metadata-title').value = metadata.title || '';
    document.getElementById('metadata-description').value = metadata.description || '';
    document.getElementById('metadata-tags').value = (metadata.tags || []).join(', ');

    showModal('edit-metadata-modal');
  } catch (error) {
    showToast('Failed to load metadata: ' + error.message, 'error');
  }
}

// Save metadata
async function saveMetadata() {
  if (!currentEditingFile) return;

  try {
    const metadata = {
      title: document.getElementById('metadata-title').value,
      description: document.getElementById('metadata-description').value,
      tags: document.getElementById('metadata-tags').value
        .split(',')
        .map(tag => tag.trim())
        .filter(tag => tag.length > 0)
    };

    await API.updateFileMetadata(currentEditingFile, metadata);
    showToast('Metadata saved successfully', 'success');
    hideModal('edit-metadata-modal');
    loadFileList();
  } catch (error) {
    showToast('Failed to save metadata: ' + error.message, 'error');
  }
}

// Open OCR Viewer in new tab
function openOcrViewer(filename) {
  // Use relative path, supports any port
  const url = `/ocr-viewer?file=${encodeURIComponent(filename)}`;
  window.open(url, '_blank');
}

// Open Chunk Viewer in new tab
function openChunkViewer(filename) {
  // Use relative path, supports any port
  const url = `/chunk-viewer?file=${encodeURIComponent(filename)}`;
  window.open(url, '_blank');
}

// Download file
// Intelligent download based on file processing status:
// 1. If chunk_processed=true or ocr_processed=true: returns ZIP with original + OCR files + chunk file
// 2. Otherwise: returns original file only
function downloadFile(filename, hasOcr = false, hasChunk = false) {
  const apiBase = getApiBase();
  let url;

  if (hasChunk || hasOcr) {
    // Use download-with-derivatives endpoint for files with OCR/Chunk processing
    // This returns a ZIP file containing original + processed files
    url = `${apiBase}/api/files/download-with-derivatives/${encodeURIComponent(filename)}`;
  } else {
    // Simple download for original files only
    url = `${apiBase}/api/files/download/${encodeURIComponent(filename)}`;
  }

  // Open in new tab to trigger download
  window.open(url, '_blank');
}

// Show file details modal
async function showFileDetails(filename) {
  try {
    const metadata = await API.getFileMetadata(filename);

    const modalBody = document.querySelector('#file-details-modal .modal-body');
    if (!modalBody) {
      console.error('File details modal not found');
      return;
    }

    // Separate fields into categories
    const fileInfo = {};
    const systemMetadata = {};
    const customTags = {};

    const knownSystemFields = ['char_length', 'publish_date', 'key_timepoints', 'summary', 'ocr_processed', 'chunk_processed'];

    // Helper function to check if value is empty/null
    const isEmptyValue = (value) => {
      if (value === null || value === undefined || value === '') {
        return true;
      }
      if (Array.isArray(value) && value.length === 0) {
        return true;
      }
      return false;
    };

    Object.entries(metadata).forEach(([key, value]) => {
      // Skip empty values
      if (isEmptyValue(value)) {
        return;
      }

      if (key.startsWith('_')) {
        // File information (e.g., _file_size, _etag)
        fileInfo[key] = value;
      } else if (knownSystemFields.includes(key)) {
        // Known system metadata
        systemMetadata[key] = value;
      } else {
        // Custom user-defined tags
        customTags[key] = value;
      }
    });

    // Helper function to format value
    const formatValue = (value) => {
      if (Array.isArray(value)) {
        return value.join(', ');
      }
      // Convert semicolon-separated strings to comma-separated for display
      const strValue = String(value);
      if (strValue.includes(';')) {
        return escapeHtml(strValue.replace(/;/g, ', '));
      }
      return escapeHtml(strValue);
    };

    // Helper function to format label
    const formatLabel = (key) => {
      return key.split('_').map(word =>
        word.charAt(0).toUpperCase() + word.slice(1)
      ).join(' ');
    };

    // Helper function to render a section
    const renderSection = (title, data, bgColor) => {
      if (Object.keys(data).length === 0) return '';
      return `
        <div style="margin-bottom: 20px;">
          <h4 style="color: #495057; margin-bottom: 10px; padding-bottom: 8px; border-bottom: 2px solid ${bgColor}; font-size: 16px;">${title}</h4>
          ${Object.entries(data).map(([key, value]) => `
            <div style="display: flex; padding: 8px 0; border-bottom: 1px solid #e9ecef;">
              <div style="font-weight: 600; color: #495057; width: 180px; flex-shrink: 0;">${formatLabel(key)}:</div>
              <div style="color: #212529; flex: 1;">${formatValue(value)}</div>
            </div>
          `).join('')}
        </div>
      `;
    };

    // Build the complete HTML
    modalBody.innerHTML = `
      ${renderSection('📁 File Information', fileInfo, '#007bff')}
      ${renderSection('📋 System Metadata', systemMetadata, '#28a745')}
      ${renderSection('🏷️ Custom Tags', customTags, '#ffc107')}
    `;

    showModal('file-details-modal');
  } catch (error) {
    showToast('Failed to load file details: ' + error.message, 'error');
  }
}

// Delete file with knowledge base reference check
async function deleteFile(filename) {
  try {
    // Check if file is referenced by any knowledge base
    const apiBase = getApiBase();
    const checkResponse = await fetch(`${apiBase}/api/files/check-references/${encodeURIComponent(filename)}`);

    if (!checkResponse.ok) {
      throw new Error('Failed to check file references');
    }

    const refData = await checkResponse.json();

    // Show custom confirmation modal
    showDeleteConfirmation(filename, refData);

  } catch (error) {
    console.error('Delete error:', error);
    showToast('Failed to check file references: ' + error.message, 'error');
  }
}

// Show delete confirmation modal
function showDeleteConfirmation(filename, refData) {
  currentDeleteData = { filename, refData };
  const body = document.getElementById('delete-confirm-body');

  if (refData.is_referenced) {
    // File is referenced - show detailed warning
    const kbList = refData.knowledge_bases.map(kb =>
      `<div style="padding: 5px 0;">• ${escapeHtml(kb.name)} (${kb.chunks_created} chunks)</div>`
    ).join('');

    const totalChunks = refData.knowledge_bases.reduce((sum, kb) => sum + kb.chunks_created, 0);

    body.innerHTML = `
      <div style="background-color: #fef2f2; border: 2px solid #fca5a5; border-radius: 8px; padding: 20px; margin-bottom: 20px;">
        <div style="color: #dc2626; font-weight: 700; font-size: 18px; margin-bottom: 16px;">
          ⚠️ Warning: This file is currently in use!
        </div>
        <div style="color: #991b1b; line-height: 1.8; white-space: pre-line;">
This file is currently used by ${refData.total_references} knowledge base(s):
<div style="background-color: white; border-left: 3px solid #dc2626; padding: 10px; margin: 10px 0; font-family: monospace;">${kbList}</div>

Deleting this file will permanently remove:
✗ File from storage
✗ All ${totalChunks} vector embeddings
✗ All knowledge base references
✗ All file-knowledge base mappings
✗ All related database tables (if Excel file)

<strong style="color: #dc2626; font-size: 16px;">⚠️ This action cannot be undone!</strong>
        </div>
      </div>

      <div style="margin-top: 20px;">
        <label style="display: block; font-weight: 600; margin-bottom: 10px; color: #374151;">
          To confirm deletion, copy and paste the filename below:
        </label>
        <div style="display: flex; gap: 10px; align-items: center; margin-bottom: 10px;">
          <div style="flex: 1; background-color: #f3f4f6; border: 2px solid #d1d5db; border-radius: 6px; padding: 12px; font-family: monospace; word-break: break-all;">
            ${escapeHtml(filename)}
          </div>
          <button class="btn btn-secondary" onclick="copyToClipboard('${filename.replace(/'/g, "\\'")}')">
            📋 Copy
          </button>
        </div>
        <input type="text"
               class="input"
               id="delete-confirmation-input"
               placeholder="Paste filename here to confirm deletion"
               autocomplete="off"
               style="width: 100%; font-family: monospace;">
      </div>
    `;

    // Disable delete button initially
    document.getElementById('delete-confirm-btn').disabled = false;
  } else {
    // File is not referenced - simple confirmation
    body.innerHTML = `
      <div style="padding: 20px; text-align: center;">
        <p style="font-size: 16px; margin-bottom: 16px; color: #374151;">
          Are you sure you want to delete this file?
        </p>
        <div style="background-color: #f3f4f6; border: 2px solid #d1d5db; border-radius: 6px; padding: 12px; display: inline-block; font-family: monospace; word-break: break-all; max-width: 100%;">
          ${escapeHtml(filename)}
        </div>
        <p style="color: #6b7280; margin-top: 16px; font-size: 14px;">
          This file is not currently used by any knowledge base.
        </p>
      </div>
    `;

    document.getElementById('delete-confirm-btn').disabled = false;
  }

  showModal('delete-confirm-modal');
}

// Confirm and execute file deletion
async function confirmFileDeletion() {
  if (!currentDeleteData) return;

  const { filename, refData } = currentDeleteData;

  // If file is referenced, validate the input
  if (refData.is_referenced) {
    const input = document.getElementById('delete-confirmation-input');
    if (!input || input.value !== filename) {
      showToast('Filename does not match, please copy and paste the exact filename', 'error');
      return;
    }
  }

  // Close modal and perform deletion
  hideModal('delete-confirm-modal');

  try {
    const apiBase = getApiBase();
    const response = await fetch(`${apiBase}/api/files/delete/${encodeURIComponent(filename)}`, {
      method: 'DELETE'
    });

    if (response.ok) {
      const result = await response.json();

      if (result.knowledge_bases_affected > 0) {
        const details = `File deleted successfully!\n\nDetails:\n` +
          `- Vector chunks deleted: ${result.vector_chunks_deleted}\n` +
          `- Knowledge bases affected: ${result.knowledge_bases_affected}\n` +
          `- Config entries deleted: ${result.config_entries_deleted}\n` +
          `- File mappings deleted: ${result.file_mappings_deleted}\n` +
          `- Excel tables deleted: ${result.excel_tables_deleted || 0}`;
        showToast(details, 'success');
      } else {
        showToast(`File "${filename}" deleted successfully!`, 'success');
      }

      loadFileList();
    } else {
      const error = await response.json();
      throw new Error(error.detail || 'Delete failed');
    }
  } catch (error) {
    console.error('Delete error:', error);
    showToast(`Failed to delete file: ${error.message}`, 'error');
  }

  currentDeleteData = null;
}

// Copy filename to clipboard
async function copyToClipboard(text) {
  try {
    await navigator.clipboard.writeText(text);
    showToast('Filename copied to clipboard!', 'success');
  } catch (err) {
    console.error('Failed to copy:', err);
    // Fallback: select the text
    const textArea = document.createElement('textarea');
    textArea.value = text;
    textArea.style.position = 'fixed';
    textArea.style.left = '-9999px';
    document.body.appendChild(textArea);
    textArea.select();
    try {
      document.execCommand('copy');
      showToast('Filename copied to clipboard!', 'success');
    } catch (err2) {
      showToast('Failed to copy to clipboard', 'error');
    }
    document.body.removeChild(textArea);
  }
}

// Import metadata from Excel
async function importMetadataFromExcel(file) {
  try {
    // Show loading toast
    showToast('Importing metadata...', 'info');

    // Create FormData and upload
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(window.APP_CONFIG.API_BASE + '/api/files/import-metadata', {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Import failed');
    }

    const result = await response.json();

    // Display results in modal
    showImportResults(result);

  } catch (error) {
    console.error('Import error:', error);
    showToast('Import failed: ' + error.message, 'error');
  }
}

// Show import results
function showImportResults(result) {
  const summaryDiv = document.getElementById('import-result-summary');
  const errorsDiv = document.getElementById('import-result-errors');

  if (!summaryDiv || !errorsDiv) {
    console.error('Import result elements not found');
    return;
  }

  // Build summary
  const successRate = result.total_rows > 0
    ? Math.round((result.successful / result.total_rows) * 100)
    : 0;

  let summaryColor = 'var(--success)';
  let summaryBgColor = 'var(--success-light, #d4edda)';
  let summaryIcon = '✅';

  if (result.failed > 0 && result.successful === 0) {
    summaryColor = 'var(--danger)';
    summaryBgColor = 'var(--danger-light, #f8d7da)';
    summaryIcon = '❌';
  } else if (result.failed > 0) {
    summaryColor = 'var(--warning)';
    summaryBgColor = 'var(--warning-light, #fff3cd)';
    summaryIcon = '⚠️';
  }

  summaryDiv.innerHTML = `
    <div style="background: ${summaryBgColor}; padding: 20px; border-radius: 8px; border: 1px solid ${summaryColor};">
      <h3 style="margin: 0 0 16px 0; font-size: 18px; color: ${summaryColor};">${summaryIcon} Import Summary</h3>
      <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px;">
        <div style="text-align: center;">
          <div style="font-size: 32px; font-weight: bold; color: var(--gray-8);">${result.total_rows}</div>
          <div style="font-size: 14px; color: var(--gray-7); margin-top: 4px;">Total Rows</div>
        </div>
        <div style="text-align: center;">
          <div style="font-size: 32px; font-weight: bold; color: var(--success);">${result.successful}</div>
          <div style="font-size: 14px; color: var(--gray-7); margin-top: 4px;">Successful</div>
        </div>
        <div style="text-align: center;">
          <div style="font-size: 32px; font-weight: bold; color: var(--danger);">${result.failed}</div>
          <div style="font-size: 14px; color: var(--gray-7); margin-top: 4px;">Failed</div>
        </div>
      </div>
      <div style="margin-top: 16px; text-align: center; font-size: 16px; font-weight: 600; color: ${summaryColor};">
        Success Rate: ${successRate}%
      </div>
    </div>
  `;

  // Build error list
  if (result.errors && result.errors.length > 0) {
    errorsDiv.innerHTML = `
      <div style="margin-top: 20px;">
        <h4 style="color: var(--danger); margin-bottom: 16px; font-size: 16px;">❌ Error List (${result.errors.length})</h4>
        <div style="background: var(--gray-1); border-radius: 8px; padding: 16px;">
          ${result.errors.map((err, idx) => `
            <div style="padding: 12px; margin-bottom: 10px; background: white; border-left: 4px solid var(--danger); border-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
              <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 6px;">
                <span style="background: var(--danger); color: white; padding: 2px 8px; border-radius: 3px; font-size: 12px; font-weight: bold;">
                  Excel Row ${err.row}
                </span>
                <span style="font-weight: 600; color: var(--gray-8); flex: 1;">
                  📄 ${escapeHtml(err.filename || 'N/A')}
                </span>
              </div>
              <div style="color: var(--danger); font-size: 14px; line-height: 1.5; padding-left: 4px;">
                ❌ <strong>Error:</strong> ${escapeHtml(err.error)}
              </div>
            </div>
          `).join('')}
        </div>
      </div>
    `;
  } else {
    errorsDiv.innerHTML = `
      <div style="text-align: center; padding: 40px; color: var(--success);">
        <div style="font-size: 48px;">🎉</div>
        <div style="font-size: 18px; font-weight: 600; margin-top: 10px;">All metadata imported successfully!</div>
      </div>
    `;
  }

  // Show modal
  showModal('import-result-modal');
}

// Export files to Excel
async function exportFilesToExcel() {
  console.log('exportFilesToExcel() called');
  try {
    // Fetch all files from the API using the API class
    const response = await fetch(window.APP_CONFIG.API_BASE + '/api/files/list');
    console.log('Response status:', response.status);

    const files = await response.json();
    console.log('Files received:', files.length);

    if (files.length === 0) {
      showToast('No files to export', 'warning');
      return;
    }

    // Collect all unique metadata keys across all files
    const metadataKeys = new Set();
    // Fields to exclude from export (auto-generated internal fields)
    const excludeFields = [
      'key_timepoints_min_stamp',
      'key_timepoints_max_stamp',
      'publish_date_min_stamp',
      'publish_date_max_stamp'
    ];

    files.forEach(file => {
      if (file.metadata) {
        Object.keys(file.metadata).forEach(key => {
          // Skip excluded fields
          if (!excludeFields.includes(key)) {
            metadataKeys.add(key);
          }
        });
      }
    });

    // Convert to sorted array for consistent column order
    const knownFields = ['char_length', 'publish_date', 'key_timepoints', 'summary', 'ocr_processed', 'chunk_processed'];
    const sortedMetadataKeys = [
      ...knownFields.filter(k => metadataKeys.has(k)),
      ...Array.from(metadataKeys).filter(k => !knownFields.includes(k)).sort()
    ];

    // Build header row: File Name, ETAG, then all metadata fields
    const headers = ['File Name', 'ETAG', ...sortedMetadataKeys];

    // Build data rows
    const dataRows = files.map(file => {
      const row = [
        file.name,
        file.etag || ''
      ];

      // Add metadata values in the same order as headers
      sortedMetadataKeys.forEach(key => {
        const value = file.metadata?.[key];

        // Handle array values (convert to semicolon-separated string)
        if (Array.isArray(value)) {
          row.push(value.join(';'));
        } else if (value === null || value === undefined) {
          row.push('');
        } else {
          row.push(String(value));
        }
      });

      return row;
    });

    // Combine headers and data
    const data = [headers, ...dataRows];

    // Create a new workbook and worksheet
    const wb = XLSX.utils.book_new();
    const ws = XLSX.utils.aoa_to_sheet(data);

    // Set column widths for better readability
    const colWidths = headers.map((h, i) => {
      if (i === 0) return { wch: 40 }; // File Name
      if (i === 1) return { wch: 35 }; // ETAG
      return { wch: 20 }; // Metadata fields
    });
    ws['!cols'] = colWidths;

    // Add the worksheet to the workbook
    XLSX.utils.book_append_sheet(wb, ws, "Files");

    // Create instructions sheet
    const instructionsData = [
      ['元数据导入使用说明', '', '', ''],
      ['', '', '', ''],
      ['重要提示', '', '', ''],
      ['1. 请勿修改 A列(File Name) 和 B列(ETAG)，否则导入会失败', '', '', ''],
      ['2. ETAG用于校验文件完整性，如果文件被修改，ETAG会不匹配', '', '', ''],
      ['3. 如遇ETAG不匹配错误，请重新导出Excel后再编辑', '', '', ''],
      ['', '', '', ''],
      ['字段限制', '', '', ''],
      ['限制项', '说明', '最大值', '备注'],
      ['自定义字段数量', '除File Name和ETAG外的列数', '50个', '超过会导入失败'],
      ['字段名长度', '列名（表头）的字符长度', '100字符', '支持中英文、数字、下划线、连字符'],
      ['字段值长度', '单元格内容的字符长度', '500字符', '超过会自动截断'],
      ['', '', '', ''],
      ['编辑规则', '', '', ''],
      ['规则', '示例', '说明', ''],
      ['多值用分号分隔', 'key_timepoints: 2024-01-01;2024-12-31', '系统会自动将分号替换为逗号前端显示', ''],
      ['空值表示删除字段', '某个单元格留空', '该文件的对应字段会被删除', ''],
      ['可添加新列', '在尾列后添加文章类型、UV等', '新列会作为自定义标签存储', ''],
      ['', '', '', ''],
      ['系统字段（可编辑）', '', '', ''],
      ['字段名', '说明', '格式示例', ''],
      ['char_length', '字符长度', '1234', ''],
      ['publish_date', '发布日期', '2024-01-01 或 2024-01-01 12:00:00', ''],
      ['key_timepoints', '关键时间点（多值）', '2024-01-01;2024-Q1;2024年', ''],
      ['summary', '摘要', '这是一段摘要文字...', ''],
      ['', '', '', ''],
      ['常用自定义字段示例', '', '', ''],
      ['字段名', '类型', '示例值', '说明'],
      ['内容类型', '枚举', '文章/财报/报告', '内容分类'],
      ['创建来源', '枚举', 'OGC-来自外部采购/PUGC-合作创作', '来源标识'],
      ['有效学习UV', '数值', '1234', '学习用户数'],
      ['内容评分', '数值', '0.875', '质量评分'],
      ['内容标签', '多值', '拼多多;电商;Q2财报', '标签，用分号分隔'],
      ['', '', '', ''],
      ['导入流程', '', '', ''],
      ['1. 点击 Import Metadata 按钮', '', '', ''],
      ['2. 选择编辑好的Excel文件（.xlsx）', '', '', ''],
      ['3. 系统会验证ETAG，跳过不匹配的行', '', '', ''],
      ['4. 查看导入结果报告，确认成功和失败数量', '', '', ''],
      ['5. 如有错误，根据错误提示修正后重新导入', '', '', ''],
      ['', '', '', ''],
      ['常见问题', '', '', ''],
      ['Q: ETAG校验失败怎么办？', 'A: 说明文件在导出后被修改了，请重新导出Excel再编辑', '', ''],
      ['Q: 可以删除某个文件的元数据吗？', 'A: 可以，将该文件对应行的所有元数据单元格清空即可', '', ''],
      ['Q: 支持批量添加新字段吗？', 'A: 支持，直接在表头添加新列即可，所有文件都会添加该字段', '', ''],
      ['Q: 导入会覆盖现有数据吗？', 'A: 是的，采用完全覆盖模式，Excel中的数据会替换MinIO中的元数据', '', ''],
      ['', '', '', ''],
      ['', '', '', ''],
      ['=== ENGLISH VERSION ===', '', '', ''],
      ['', '', '', ''],
      ['Metadata Import Instructions', '', '', ''],
      ['', '', '', ''],
      ['Important Notes', '', '', ''],
      ['1. Do not modify Column A (File Name) and Column B (ETAG), otherwise import will fail', '', '', ''],
      ['2. ETAG is used to verify file integrity. If the file is modified, ETAG will not match', '', '', ''],
      ['3. If you encounter ETAG mismatch error, please re-export Excel and then edit', '', '', ''],
      ['', '', '', ''],
      ['Field Limitations', '', '', ''],
      ['Limitation', 'Description', 'Maximum', 'Notes'],
      ['Custom field count', 'Number of columns excluding File Name and ETAG', '50 fields', 'Exceeding will cause import failure'],
      ['Field name length', 'Character length of column name (header)', '100 characters', 'Supports letters, numbers, underscores, hyphens'],
      ['Field value length', 'Character length of cell content', '500 characters', 'Exceeding will be truncated'],
      ['', '', '', ''],
      ['Editing Rules', '', '', ''],
      ['Rule', 'Example', 'Description', ''],
      ['Multiple values separated by semicolon', 'key_timepoints: 2024-01-01;2024-12-31', 'System will automatically replace semicolons with commas for frontend display', ''],
      ['Empty value means delete field', 'Leave a cell empty', 'The corresponding field of the file will be deleted', ''],
      ['Can add new columns', 'Add article type, UV, etc. after the last column', 'New columns will be stored as custom tags', ''],
      ['', '', '', ''],
      ['System Fields (Editable)', '', '', ''],
      ['Field Name', 'Description', 'Format Example', ''],
      ['char_length', 'Character length', '1234', ''],
      ['publish_date', 'Publish date', '2024-01-01 or 2024-01-01 12:00:00', ''],
      ['key_timepoints', 'Key timepoints (multiple values)', '2024-01-01;2024-Q1;2024', ''],
      ['summary', 'Summary', 'This is a summary text...', ''],
      ['', '', '', ''],
      ['Common Custom Field Examples', '', '', ''],
      ['Field Name', 'Type', 'Example Value', 'Description'],
      ['Content Type', 'Enum', 'Article/Financial Report/Report', 'Content classification'],
      ['Creation Source', 'Enum', 'OGC-External Purchase/PUGC-Collaborative Creation', 'Source identifier'],
      ['Effective Learning UV', 'Number', '1234', 'Learning user count'],
      ['Content Rating', 'Number', '0.875', 'Quality rating'],
      ['Content Tags', 'Multiple', 'Pinduoduo;E-commerce;Q2 Report', 'Tags, separated by semicolons'],
      ['', '', '', ''],
      ['Import Process', '', '', ''],
      ['1. Click Import Metadata button', '', '', ''],
      ['2. Select the edited Excel file (.xlsx)', '', '', ''],
      ['3. System will verify ETAG and skip mismatched rows', '', '', ''],
      ['4. View import result report to confirm success and failure counts', '', '', ''],
      ['5. If there are errors, correct them according to error prompts and re-import', '', '', ''],
      ['', '', '', ''],
      ['FAQ', '', '', ''],
      ['Q: What if ETAG verification fails?', 'A: It means the file was modified after export. Please re-export Excel and edit again', '', ''],
      ['Q: Can I delete metadata for a file?', 'A: Yes, just clear all metadata cells in the corresponding row', '', ''],
      ['Q: Can I batch add new fields?', 'A: Yes, just add new columns in the header. All files will have this field added', '', ''],
      ['Q: Will import overwrite existing data?', 'A: Yes, it uses full overwrite mode. Data in Excel will replace metadata in MinIO', '', ''],
    ];

    const wsInstructions = XLSX.utils.aoa_to_sheet(instructionsData);
    wsInstructions['!cols'] = [
      { wch: 40 },
      { wch: 35 },
      { wch: 30 },
      { wch: 25 }
    ];

    // Merge cells for title
    if (!wsInstructions['!merges']) wsInstructions['!merges'] = [];
    wsInstructions['!merges'].push(
      // Chinese version
      { s: { r: 0, c: 0 }, e: { r: 0, c: 3 } },  // Title row "元数据导入使用说明"
      { s: { r: 2, c: 0 }, e: { r: 2, c: 3 } },  // "重要提示"
      { s: { r: 7, c: 0 }, e: { r: 7, c: 3 } },  // "字段限制"
      { s: { r: 13, c: 0 }, e: { r: 13, c: 3 } }, // "编辑规则"
      { s: { r: 18, c: 0 }, e: { r: 18, c: 3 } }, // "系统字段"
      { s: { r: 26, c: 0 }, e: { r: 26, c: 3 } }, // "常用自定义字段"
      { s: { r: 34, c: 0 }, e: { r: 34, c: 3 } }, // "导入流程"
      { s: { r: 41, c: 0 }, e: { r: 41, c: 3 } }, // "常见问题"
      // English version (starts at row 48)
      { s: { r: 48, c: 0 }, e: { r: 48, c: 3 } }, // "=== ENGLISH VERSION ==="
      { s: { r: 50, c: 0 }, e: { r: 50, c: 3 } }, // "Metadata Import Instructions"
      { s: { r: 52, c: 0 }, e: { r: 52, c: 3 } }, // "Important Notes"
      { s: { r: 57, c: 0 }, e: { r: 57, c: 3 } }, // "Field Limitations"
      { s: { r: 63, c: 0 }, e: { r: 63, c: 3 } }, // "Editing Rules"
      { s: { r: 69, c: 0 }, e: { r: 69, c: 3 } }, // "System Fields (Editable)"
      { s: { r: 76, c: 0 }, e: { r: 76, c: 3 } }, // "Common Custom Field Examples"
      { s: { r: 84, c: 0 }, e: { r: 84, c: 3 } }, // "Import Process"
      { s: { r: 91, c: 0 }, e: { r: 91, c: 3 } }, // "FAQ"
    );

    // Add instructions sheet
    XLSX.utils.book_append_sheet(wb, wsInstructions, "Instructions");

    // Generate Excel file and trigger download
    const now = new Date();
    const timestamp = `${now.getFullYear()}${String(now.getMonth() + 1).padStart(2, '0')}${String(now.getDate()).padStart(2, '0')}_${String(now.getHours()).padStart(2, '0')}${String(now.getMinutes()).padStart(2, '0')}${String(now.getSeconds()).padStart(2, '0')}`;
    const filename = `file_metadata_${timestamp}.xlsx`;
    XLSX.writeFile(wb, filename);

    showToast(`Successfully exported metadata for ${files.length} files`, 'success');
  } catch (error) {
    console.error('Export error:', error);
    showToast('Export failed: ' + error.message, 'error');
  }
}

// Dropdown Menu Functions
function toggleDropdown(event, button) {
  event.stopPropagation();

  const dropdown = button.closest('.dropdown');
  const menu = dropdown.querySelector('.dropdown-menu');
  const isCurrentlyOpen = menu.classList.contains('show');

  // Close all other dropdowns first
  closeAllDropdowns();

  // Toggle the current dropdown
  if (!isCurrentlyOpen) {
    menu.classList.add('show');
  }
}

function closeAllDropdowns() {
  document.querySelectorAll('.dropdown-menu.show').forEach(menu => {
    menu.classList.remove('show');
  });
}

// Handle viewer menu item click
function handleViewerClick(event, type, filename) {
  event.stopPropagation();

  // Close the dropdown
  closeAllDropdowns();

  // Open the appropriate viewer or show details
  if (type === 'details') {
    showFileDetails(filename);
  } else if (type === 'ocr') {
    openOcrViewer(filename);
  } else if (type === 'chunk') {
    openChunkViewer(filename);
  }
}

// Close dropdowns when clicking outside
document.addEventListener('click', (event) => {
  if (!event.target.closest('.dropdown')) {
    closeAllDropdowns();
  }
});

// Toggle select mode
async function toggleSelectMode() {
  isSelectMode = !isSelectMode;
  const selectModeBtn = document.getElementById('select-mode-btn');
  const batchDeleteBtn = document.getElementById('batch-delete-btn');
  const checkboxHeader = document.getElementById('checkbox-header');
  const selectAllCheckbox = document.getElementById('select-all-checkbox');

  if (isSelectMode) {
    // Entering select mode
    selectModeBtn.innerHTML = `
      <img src="assets/images/close.svg" alt="Cancel" style="width: 16px; height: 16px; margin-right: 6px; vertical-align: middle;">
      <span data-i18n="exit_select_mode">${t('exit_select_mode')}</span>
    `;
    checkboxHeader.style.display = 'table-cell';
    batchDeleteBtn.style.display = 'inline-flex';
  } else {
    // Exiting select mode
    selectModeBtn.innerHTML = `
      <img src="assets/images/select.svg" alt="Select" style="width: 16px; height: 16px; margin-right: 6px; vertical-align: middle;">
      <span data-i18n="batch_select">${t('batch_select')}</span>
    `;
    checkboxHeader.style.display = 'none';
    batchDeleteBtn.style.display = 'none';
    selectedFiles.clear();
    selectAllCheckbox.checked = false;
  }

  // Reload file list to show/hide checkboxes and wait for completion
  await loadFileList();
  updateFileManagerBatchDeleteButton();
}

// Toggle select all
function toggleSelectAll(checked) {
  const checkboxes = document.querySelectorAll('.file-checkbox');
  checkboxes.forEach(checkbox => {
    const filename = checkbox.dataset.filename;
    checkbox.checked = checked;
    if (checked) {
      selectedFiles.add(filename);
    } else {
      selectedFiles.delete(filename);
    }
  });
  updateFileManagerBatchDeleteButton();
}

// Toggle file selection (file manager specific)
function toggleFileCheckbox(checkbox) {
  // Get filename from checkbox's data attribute
  const filename = checkbox.dataset.filename;

  console.log('[toggleFileCheckbox] Checkbox clicked:', filename, 'checked:', checkbox.checked);

  // Update selectedFiles based on checkbox's actual checked state
  if (checkbox.checked) {
    selectedFiles.add(filename);
  } else {
    selectedFiles.delete(filename);
  }

  console.log('[toggleFileCheckbox] selectedFiles size:', selectedFiles.size);

  // Update select all checkbox state
  const selectAllCheckbox = document.getElementById('select-all-checkbox');
  const checkboxes = document.querySelectorAll('.file-checkbox');
  const allChecked = Array.from(checkboxes).every(cb => cb.checked);
  const noneChecked = Array.from(checkboxes).every(cb => !cb.checked);

  selectAllCheckbox.checked = allChecked;
  selectAllCheckbox.indeterminate = !allChecked && !noneChecked;

  updateFileManagerBatchDeleteButton();
}

// Update batch delete button text and state (file manager)
function updateFileManagerBatchDeleteButton() {
  const batchDeleteBtn = document.getElementById('batch-delete-btn');
  const batchDeleteText = document.getElementById('batch-delete-text');
  const count = selectedFiles.size;

  console.log('[updateFileManagerBatchDeleteButton] Count:', count, 'batchDeleteText exists:', !!batchDeleteText);

  if (batchDeleteText) {
    batchDeleteText.textContent = t('delete_selected', { count });
    console.log('[updateFileManagerBatchDeleteButton] Updated text to:', batchDeleteText.textContent);
  }

  // Disable button if no files selected
  if (batchDeleteBtn) {
    batchDeleteBtn.disabled = count === 0;
    if (count === 0) {
      // Disabled state: lighter colors
      batchDeleteBtn.style.backgroundColor = '#fef2f2';
      batchDeleteBtn.style.color = '#fca5a5';
      batchDeleteBtn.style.borderColor = '#fecaca';
      batchDeleteBtn.style.cursor = 'not-allowed';
    } else {
      // Active state: restore original colors
      batchDeleteBtn.style.backgroundColor = '#fee2e2';
      batchDeleteBtn.style.color = '#dc2626';
      batchDeleteBtn.style.borderColor = '#fca5a5';
      batchDeleteBtn.style.cursor = 'pointer';
    }
  }
}

// Batch delete files
async function batchDeleteFiles() {
  const count = selectedFiles.size;
  if (count === 0) {
    showToast('Please select files to delete first', 'warning');
    return;
  }

  const confirmed = await confirmDialog(
    `Are you sure you want to delete ${count} selected file(s)? This action cannot be undone.`
  );
  if (!confirmed) return;

  const filesToDelete = Array.from(selectedFiles);
  let successCount = 0;
  let failCount = 0;

  showToast(`Deleting ${count} file(s)...`, 'info');

  for (const filename of filesToDelete) {
    try {
      await API.deleteFile(filename);
      successCount++;
      selectedFiles.delete(filename);
    } catch (error) {
      console.error(`Failed to delete file: ${filename}`, error);
      failCount++;
    }
  }

  // Show result
  if (failCount === 0) {
    showToast(`Successfully deleted ${successCount} file(s)`, 'success');
  } else {
    showToast(`Deletion complete: ${successCount} succeeded, ${failCount} failed`, 'warning');
  }

  // Clear selection and reload list
  selectedFiles.clear();
  updateFileManagerBatchDeleteButton();

  // Update select all checkbox
  const selectAllCheckbox = document.getElementById('select-all-checkbox');
  if (selectAllCheckbox) {
    selectAllCheckbox.checked = false;
    selectAllCheckbox.indeterminate = false;
  }

  // Reload file list
  loadFileList();
}

// File Content Sidebar Functions
function openFileContentSidebar(filename) {
  const sidebar = document.getElementById('file-content-sidebar');
  const overlay = document.getElementById('sidebar-overlay');
  const titleElement = document.getElementById('sidebar-filename');
  const loadingElement = document.getElementById('sidebar-content-loading');
  const displayElement = document.getElementById('sidebar-content-display');

  // Set filename in title
  if (titleElement) {
    titleElement.textContent = filename;
  }

  // Show sidebar and overlay
  if (sidebar) {
    sidebar.style.display = 'flex';
    setTimeout(() => sidebar.classList.add('show'), 10);
  }
  if (overlay) {
    overlay.style.display = 'block';
    setTimeout(() => overlay.classList.add('show'), 10);
  }

  // Show loading state
  if (loadingElement) loadingElement.style.display = 'block';
  if (displayElement) displayElement.style.display = 'none';

  // Load file content
  loadFileContent(filename);
}

function closeFileContentSidebar() {
  const sidebar = document.getElementById('file-content-sidebar');
  const overlay = document.getElementById('sidebar-overlay');

  if (sidebar) {
    sidebar.classList.remove('show');
    setTimeout(() => sidebar.style.display = 'none', 300);
  }
  if (overlay) {
    overlay.classList.remove('show');
    setTimeout(() => overlay.style.display = 'none', 300);
  }

  // Clean up workbook data
  if (window._currentWorkbook) {
    delete window._currentWorkbook;
  }
}

async function loadFileContent(filename) {
  const apiBase = getApiBase();
  const loadingElement = document.getElementById('sidebar-content-loading');
  const displayElement = document.getElementById('sidebar-content-display');

  try {
    const response = await fetch(`${apiBase}/api/files/download/${encodeURIComponent(filename)}`);

    if (!response.ok) {
      throw new Error('Failed to load file');
    }

    const contentType = response.headers.get('content-type') || '';
    const fileExt = filename.split('.').pop().toLowerCase();

    // Get file content as blob
    const blob = await response.blob();

    // Hide loading, show content
    if (loadingElement) loadingElement.style.display = 'none';
    if (displayElement) displayElement.style.display = 'block';

    // Render content based on file type
    if (contentType.includes('image/') || ['png', 'jpg', 'jpeg', 'bmp', 'webp'].includes(fileExt)) {
      // Display image
      const imageUrl = URL.createObjectURL(blob);
      displayElement.innerHTML = `
        <div style="text-align: center;">
          <img src="${imageUrl}" alt="${escapeHtml(filename)}" style="max-width: 100%; height: auto; border-radius: var(--radius-md);">
        </div>
      `;
    } else if (fileExt === 'pdf') {
      // Display PDF using PDF.js - render pages as images
      try {
        console.log('Starting PDF rendering for:', filename);
        console.log('Blob size:', blob.size, 'bytes');
        console.log('Blob type:', blob.type);
        console.log('Content-Type header:', contentType);

        const arrayBuffer = await blob.arrayBuffer();
        console.log('ArrayBuffer size:', arrayBuffer.byteLength, 'bytes');

        // Verify we have PDF data
        const uint8Array = new Uint8Array(arrayBuffer);
        const pdfHeader = String.fromCharCode(...uint8Array.slice(0, 10));
        console.log('First 10 bytes as string:', pdfHeader);
        console.log('First 10 bytes as hex:', Array.from(uint8Array.slice(0, 10)).map(b => b.toString(16).padStart(2, '0')).join(' '));

        // Try to read as text to see what we actually got (for error debugging)
        if (blob.size < 10000) {
          const text = await blob.text();
          console.log('Blob content (first 1000 chars):', text.substring(0, 1000));
          console.error('ERROR: Backend did not return PDF data. Response content:', text);
        }

        if (!pdfHeader.startsWith('%PDF')) {
          // Get full error message if it's text
          const errorText = blob.size < 10000 ? await blob.text() : `First ${blob.size} bytes are not PDF format`;

          // Check if it looks like markdown/text (likely OCR output mistakenly stored)
          if (errorText.startsWith('**') || errorText.includes('# ') || blob.size < 50000) {
            throw new Error(`Stored file is corrupted\n\nThe file stored in MinIO is text content instead of a PDF file (size: ${blob.size} bytes).\n\nThis may be caused by:\n• Original PDF was incorrectly overwritten during upload or OCR processing\n• File was corrupted during processing\n\nSuggested solutions:\n1. Delete this file\n2. Re-upload the original PDF file\n\nIf you need to save the current text content, please use the download button.`);
          }

          throw new Error(`Backend did not return a valid PDF file.\n\nPossible reasons:\n1. File does not exist in MinIO\n2. MinIO connection failed\n3. File permission issue\n\nBackend response (first 100 chars):\n${errorText.substring(0, 100)}`);
        }

        // Configure PDF.js worker
        if (typeof pdfjsLib !== 'undefined') {
          pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';
        } else {
          throw new Error('PDF.js library not loaded');
        }

        displayElement.innerHTML = `
          <div style="margin-bottom: var(--spacing-sm);">
            <span id="pdf-page-info" style="color: var(--gray-7); font-size: var(--font-size-sm);">Loading...</span>
          </div>
          <div id="pdf-pages-container" style="display: flex; flex-direction: column; gap: var(--spacing-md);">
            <!-- PDF pages will be rendered here -->
          </div>
        `;

        console.log('Loading PDF document...');
        // Load and render PDF
        const loadingTask = pdfjsLib.getDocument({ data: arrayBuffer });
        const pdf = await loadingTask.promise;
        console.log('PDF loaded successfully, pages:', pdf.numPages);

        const totalPages = pdf.numPages;
        document.getElementById('pdf-page-info').textContent = `Total ${totalPages} page(s)`;

        const container = document.getElementById('pdf-pages-container');

        // Render each page
        for (let pageNum = 1; pageNum <= totalPages; pageNum++) {
          console.log(`Rendering page ${pageNum}/${totalPages}`);
          const page = await pdf.getPage(pageNum);

          // Calculate scale to fit width (sidebar is 600px, minus padding)
          const viewport = page.getViewport({ scale: 1 });
          const scale = Math.min(550 / viewport.width, 2.0); // Max scale 2.0 for quality
          const scaledViewport = page.getViewport({ scale: scale });

          // Create canvas for this page
          const canvas = document.createElement('canvas');
          const context = canvas.getContext('2d');
          canvas.width = scaledViewport.width;
          canvas.height = scaledViewport.height;
          canvas.style.width = '100%';
          canvas.style.height = 'auto';
          canvas.style.borderRadius = 'var(--radius-md)';
          canvas.style.boxShadow = '0 2px 8px rgba(0, 0, 0, 0.1)';

          // Create page wrapper
          const pageWrapper = document.createElement('div');
          pageWrapper.style.position = 'relative';

          const pageLabel = document.createElement('div');
          pageLabel.textContent = `Page ${pageNum}`;
          pageLabel.style.cssText = 'font-size: 12px; color: var(--gray-7); margin-bottom: 4px; font-weight: 500;';

          pageWrapper.appendChild(pageLabel);
          pageWrapper.appendChild(canvas);
          container.appendChild(pageWrapper);

          // Render page
          await page.render({
            canvasContext: context,
            viewport: scaledViewport
          }).promise;
        }

        console.log('All pages rendered successfully');

      } catch (error) {
        console.error('PDF rendering error:', error);
        console.error('Error stack:', error.stack);

        // Check if this is a corrupted file error
        const isCorruptedFile = error.message.includes('Stored file is corrupted');

        displayElement.innerHTML = `
          <div class="file-type-notice">
            <div class="icon">❌</div>
            <h3>PDF ${isCorruptedFile ? 'File Corrupted' : 'Rendering Failed'}</h3>
            <p style="margin-top: var(--spacing-md); color: var(--error); white-space: pre-line; text-align: left; max-width: 500px; margin-left: auto; margin-right: auto;">${escapeHtml(error.message)}</p>
            ${isCorruptedFile ? '' : '<p style="color: var(--gray-6); margin-top: var(--spacing-sm); font-size: 12px;">Please check browser console for detailed error information</p>'}
            <button class="btn btn-${isCorruptedFile ? 'secondary' : 'primary'}" style="margin-top: var(--spacing-lg);" onclick="downloadFile('${escapeHtml(filename)}')">
              <img src="assets/images/download.svg" alt="Download" style="width: 16px; height: 16px; filter: invert(50%) sepia(0%) saturate(0%) brightness(90%);">
              Download ${isCorruptedFile ? 'Text ' : ''}File
            </button>
            ${isCorruptedFile ? `<button class="btn btn-danger" style="margin-top: var(--spacing-sm); margin-left: var(--spacing-sm);" onclick="if(confirm('Are you sure you want to delete this corrupted file?')) { deleteFile('${escapeHtml(filename)}'); closeFileContentSidebar(); }">
              <img src="assets/images/delete.svg" alt="Delete" style="width: 16px; height: 16px; filter: invert(50%) sepia(0%) saturate(0%) brightness(90%);">
              Delete Corrupted File
            </button>` : ''}
          </div>
        `;
      }
    } else if (fileExt === 'xlsx' || fileExt === 'xls') {
      // Display Excel file as table
      try {
        const arrayBuffer = await blob.arrayBuffer();
        const workbook = XLSX.read(arrayBuffer, { type: 'array' });

        // Get first sheet
        const firstSheetName = workbook.SheetNames[0];
        const worksheet = workbook.Sheets[firstSheetName];

        // Limit rows to prevent performance issues
        const MAX_ROWS = 3000;
        let rowLimitWarning = '';
        if (worksheet['!ref']) {
          const range = XLSX.utils.decode_range(worksheet['!ref']);
          const totalRows = range.e.r + 1;
          if (totalRows > MAX_ROWS) {
            range.e.r = MAX_ROWS - 1;
            worksheet['!ref'] = XLSX.utils.encode_range(range);
            rowLimitWarning = `
              <div class="excel-row-limit-warning" style="margin-bottom: var(--spacing-md); padding: var(--spacing-sm); background-color: var(--warning-bg, #fff3cd); border-left: 3px solid var(--warning, #ffc107); border-radius: var(--radius-md);">
                <span style="color: var(--warning-dark, #856404); font-size: var(--font-size-sm);">⚠️ Too many rows, showing only first ${MAX_ROWS} rows (total ${totalRows} rows)</span>
              </div>
            `;
          }
        }

        // Convert to HTML table
        const htmlTable = XLSX.utils.sheet_to_html(worksheet);

        // Show sheet selector if multiple sheets
        let sheetSelector = '';
        if (workbook.SheetNames.length > 1) {
          sheetSelector = `
            <div style="margin-bottom: var(--spacing-md); padding: var(--spacing-sm); background-color: var(--gray-2); border-radius: var(--radius-md);">
              <label style="font-weight: 600; margin-right: var(--spacing-sm); color: var(--gray-9);">Worksheet:</label>
              <select id="sheet-selector" onchange="switchExcelSheet(this.value)" style="padding: 6px 12px; border: 1px solid var(--gray-4); border-radius: var(--radius-sm); background-color: white; cursor: pointer; font-size: var(--font-size-sm);">
                ${workbook.SheetNames.map((name, idx) =>
                  `<option value="${idx}" ${idx === 0 ? 'selected' : ''}>${escapeHtml(name)}</option>`
                ).join('')}
              </select>
              <span style="margin-left: var(--spacing-sm); color: var(--gray-7); font-size: var(--font-size-sm);">Total ${workbook.SheetNames.length} worksheet(s)</span>
            </div>
          `;
        } else {
          // Show single sheet name
          sheetSelector = `
            <div style="margin-bottom: var(--spacing-md); padding: var(--spacing-sm); background-color: var(--gray-2); border-radius: var(--radius-md);">
              <span style="font-weight: 600; color: var(--gray-9);">Worksheet: </span>
              <span style="color: var(--gray-8);">${escapeHtml(firstSheetName)}</span>
            </div>
          `;
        }

        // Store workbook data for sheet switching
        window._currentWorkbook = workbook;

        displayElement.innerHTML = `
          ${sheetSelector}
          ${rowLimitWarning}
          <div class="excel-table-container">
            ${htmlTable.replace('<table', '<table class="excel-table"')}
          </div>
        `;
      } catch (error) {
        console.error('Excel parsing error:', error);
        displayElement.innerHTML = `
          <div class="file-type-notice">
            <div class="icon">❌</div>
            <h3>Excel File Parsing Failed</h3>
            <p style="margin-top: var(--spacing-md); color: var(--error);">${error.message}</p>
          </div>
        `;
      }
    } else if (fileExt === 'docx') {
      // Display DOCX file
      try {
        const arrayBuffer = await blob.arrayBuffer();
        const result = await mammoth.convertToHtml({ arrayBuffer: arrayBuffer });

        displayElement.innerHTML = `
          <div class="docx-content">
            ${result.value}
          </div>
        `;

        // Log any conversion messages/warnings
        if (result.messages.length > 0) {
          console.warn('DOCX conversion messages:', result.messages);
        }
      } catch (error) {
        console.error('DOCX parsing error:', error);
        displayElement.innerHTML = `
          <div class="file-type-notice">
            <div class="icon">❌</div>
            <h3>Word Document Parsing Failed</h3>
            <p style="margin-top: var(--spacing-md); color: var(--error);">${error.message}</p>
          </div>
        `;
      }
    } else if (fileExt === 'doc') {
      // DOC format (old Word format) cannot be parsed in browser
      displayElement.innerHTML = `
        <div class="file-type-notice">
          <div class="icon">📄</div>
          <h3>Legacy Word Document Format</h3>
          <p style="margin-top: var(--spacing-md); color: var(--gray-7);">File type: DOC</p>
          <p style="color: var(--gray-6);">Only .docx format supports online preview, please download .doc files to view</p>
          <button class="btn btn-primary" style="margin-top: var(--spacing-lg);" onclick="downloadFile('${escapeHtml(filename)}')">
            <img src="assets/images/download.svg" alt="Download" style="width: 16px; height: 16px;">
            Download File
          </button>
        </div>
      `;
    } else if (['txt', 'md', 'json', 'xml', 'csv', 'log', 'py', 'js', 'html', 'css'].includes(fileExt)) {
      // Display text content
      const text = await blob.text();
      displayElement.innerHTML = `<pre style="background-color: var(--gray-2); padding: var(--spacing-md); border-radius: var(--radius-md); overflow-x: auto; font-size: var(--font-size-sm); line-height: 1.6; white-space: pre-wrap; word-wrap: break-word;">${escapeHtml(text)}</pre>`;
    } else {
      // Unknown file type
      displayElement.innerHTML = `
        <div class="file-type-notice">
          <div class="icon">📦</div>
          <h3>Unknown File Type</h3>
          <p style="margin-top: var(--spacing-md); color: var(--gray-7);">File type: ${fileExt.toUpperCase()}</p>
          <p style="color: var(--gray-6);">Cannot preview this file</p>
          <button class="btn btn-primary" style="margin-top: var(--spacing-lg);" onclick="downloadFile('${escapeHtml(filename)}')">
            <img src="assets/images/download.svg" alt="Download" style="width: 16px; height: 16px;">
            Download File
          </button>
        </div>
      `;
    }
  } catch (error) {
    console.error('Load file content error:', error);
    if (loadingElement) loadingElement.style.display = 'none';
    if (displayElement) {
      displayElement.style.display = 'block';
      displayElement.innerHTML = `
        <div class="file-type-notice">
          <div class="icon">❌</div>
          <h3>Loading Failed</h3>
          <p style="margin-top: var(--spacing-md); color: var(--error);">${error.message}</p>
        </div>
      `;
    }
  }
}

// Switch Excel sheet viewer
function switchExcelSheet(sheetIndex) {
  const displayElement = document.getElementById('sidebar-content-display');

  if (!window._currentWorkbook) {
    console.error('No workbook data available');
    return;
  }

  try {
    const workbook = window._currentWorkbook;
    const sheetName = workbook.SheetNames[sheetIndex];
    const worksheet = workbook.Sheets[sheetName];

    // Limit rows to prevent performance issues
    const MAX_ROWS = 3000;
    let rowLimitWarning = '';
    if (worksheet['!ref']) {
      const range = XLSX.utils.decode_range(worksheet['!ref']);
      const totalRows = range.e.r + 1;
      if (totalRows > MAX_ROWS) {
        range.e.r = MAX_ROWS - 1;
        worksheet['!ref'] = XLSX.utils.encode_range(range);
        rowLimitWarning = `
          <div class="excel-row-limit-warning" style="margin-bottom: var(--spacing-md); padding: var(--spacing-sm); background-color: var(--warning-bg, #fff3cd); border-left: 3px solid var(--warning, #ffc107); border-radius: var(--radius-md);">
            <span style="color: var(--warning-dark, #856404); font-size: var(--font-size-sm);">⚠️ 数据行数过多，仅显示前 ${MAX_ROWS} 行（共 ${totalRows} 行）</span>
          </div>
        `;
      }
    }

    // Convert to HTML table
    const htmlTable = XLSX.utils.sheet_to_html(worksheet);

    // Update the warning message
    const existingWarning = displayElement.querySelector('.excel-row-limit-warning');
    if (existingWarning) {
      if (rowLimitWarning) {
        existingWarning.outerHTML = rowLimitWarning;
      } else {
        existingWarning.remove();
      }
    } else if (rowLimitWarning) {
      const sheetSelector = displayElement.querySelector('.excel-table-container');
      if (sheetSelector) {
        sheetSelector.insertAdjacentHTML('beforebegin', rowLimitWarning);
      }
    }

    // Update only the table container
    const tableContainer = displayElement.querySelector('.excel-table-container');
    if (tableContainer) {
      tableContainer.innerHTML = htmlTable.replace('<table', '<table class="excel-table"');
    }
  } catch (error) {
    console.error('Sheet switching error:', error);
    showToast('Failed to switch worksheet: ' + error.message, 'error');
  }
}

// Toggle update time sort order
function toggleUpdateTimeSort() {
  // Clear filename sort (mutual exclusivity)
  fileNameSortOrder = null;

  // Cycle through: null -> desc -> asc -> null
  if (updateTimeSortOrder === null) {
    updateTimeSortOrder = 'desc'; // First click: newest first
  } else if (updateTimeSortOrder === 'desc') {
    updateTimeSortOrder = 'asc'; // Second click: oldest first
  } else {
    updateTimeSortOrder = null; // Third click: clear sort
  }

  // Update sort indicators
  updateSortIndicators();

  // Apply sort and re-render
  if (updateTimeSortOrder) {
    sortFilesByUpdateTime();
  } else {
    // Restore original order from last API call
    // (cachedFiles is already loaded)
  }

  renderFileList(cachedFiles);
}

// Toggle filename sort order
function toggleFileNameSort() {
  // Clear update time sort (mutual exclusivity)
  updateTimeSortOrder = null;

  // Cycle through: null -> asc -> desc -> null
  if (fileNameSortOrder === null) {
    fileNameSortOrder = 'asc'; // First click: A-Z
  } else if (fileNameSortOrder === 'asc') {
    fileNameSortOrder = 'desc'; // Second click: Z-A
  } else {
    fileNameSortOrder = null; // Third click: clear sort
  }

  // Update sort indicators
  updateSortIndicators();

  // Apply sort and re-render
  if (fileNameSortOrder) {
    sortFilesByName();
  } else {
    // Restore original order from last API call
    // (cachedFiles is already loaded)
  }

  renderFileList(cachedFiles);
}

// Update the sort indicator icons
function updateSortIndicators() {
  // Update filename sort indicator
  const filenameIndicator = document.getElementById('filename-sort-indicator');
  if (filenameIndicator) {
    if (fileNameSortOrder === 'asc') {
      filenameIndicator.textContent = '↑';
      filenameIndicator.style.color = 'var(--primary-blue, #3b82f6)';
    } else if (fileNameSortOrder === 'desc') {
      filenameIndicator.textContent = '↓';
      filenameIndicator.style.color = 'var(--primary-blue, #3b82f6)';
    } else {
      filenameIndicator.textContent = '⇅';
      filenameIndicator.style.color = 'var(--gray-6)';
    }
  }

  // Update update time sort indicator
  const updateTimeIndicator = document.getElementById('updatetime-sort-indicator');
  if (updateTimeIndicator) {
    if (updateTimeSortOrder === 'desc') {
      updateTimeIndicator.textContent = '↓';
      updateTimeIndicator.style.color = 'var(--primary-blue, #3b82f6)';
    } else if (updateTimeSortOrder === 'asc') {
      updateTimeIndicator.textContent = '↑';
      updateTimeIndicator.style.color = 'var(--primary-blue, #3b82f6)';
    } else {
      updateTimeIndicator.textContent = '⇅';
      updateTimeIndicator.style.color = 'var(--gray-6)';
    }
  }
}

// Sort files by update time
function sortFilesByUpdateTime() {
  cachedFiles.sort((a, b) => {
    // Get timestamps
    const timeA = a.last_modified || a.created_at || a.upload_time;
    const timeB = b.last_modified || b.created_at || b.upload_time;

    // Handle missing timestamps
    if (!timeA && !timeB) return 0;
    if (!timeA) return 1; // Put files without timestamp at the end
    if (!timeB) return -1;

    // Convert to Date objects for comparison
    const dateA = new Date(timeA);
    const dateB = new Date(timeB);

    if (updateTimeSortOrder === 'desc') {
      return dateB - dateA; // Newest first
    } else {
      return dateA - dateB; // Oldest first
    }
  });
}

// Sort files by name
function sortFilesByName() {
  cachedFiles.sort((a, b) => {
    // Get filenames (try multiple possible properties)
    const nameA = (a.filename || a.name || '').toLowerCase();
    const nameB = (b.filename || b.name || '').toLowerCase();

    if (fileNameSortOrder === 'asc') {
      return nameA.localeCompare(nameB); // A-Z
    } else {
      return nameB.localeCompare(nameA); // Z-A
    }
  });
}

// ============================================
// Upload Progress Bar Helper Functions
// ============================================

// Create progress bar UI
function createProgressBar(progressId, filename) {
  const progressContainer = document.getElementById('progress-container');
  if (!progressContainer) {
    console.error('[UI] progress-container not found!');
    return;
  }

  const progressHTML = `
    <div id="${progressId}" class="upload-progress">
      <div class="progress-header">
        <span class="filename">${escapeHtml(filename)}</span>
        <span class="progress-percent">0%</span>
      </div>
      <div class="progress-bar-container">
        <div class="progress-bar" style="width: 0%"></div>
      </div>
      <div class="progress-message">Preparing upload...</div>
    </div>
  `;
  progressContainer.insertAdjacentHTML('beforeend', progressHTML);
}

// Update progress bar
function updateProgressBar(progressId, percent, message) {
  const progressElement = document.getElementById(progressId);
  if (!progressElement) return;

  const bar = progressElement.querySelector('.progress-bar');
  const percentElement = progressElement.querySelector('.progress-percent');
  const messageElement = progressElement.querySelector('.progress-message');

  if (bar) bar.style.width = `${percent}%`;
  if (percentElement) percentElement.textContent = `${percent}%`;
  if (messageElement) messageElement.textContent = message;
}

// Remove progress bar
function removeProgressBar(progressId) {
  const progressElement = document.getElementById(progressId);
  if (progressElement) {
    progressElement.remove();
  }
}

// Poll upload progress
async function pollProgress(taskId, progressId) {
  const progressElement = document.getElementById(progressId);
  if (!progressElement) {
    console.error(`[Poll] Progress element not found: ${progressId}`);
    return;
  }

  const progressBar = progressElement.querySelector('.progress-bar');
  const progressPercent = progressElement.querySelector('.progress-percent');
  const progressMessage = progressElement.querySelector('.progress-message');

  if (!progressBar || !progressPercent || !progressMessage) {
    console.error('[Poll] Failed to find UI sub-elements in progress element!');
    return;
  }

  let pollInterval = null;

  // Extract polling logic as independent function
  const doPoll = async () => {
    try {
      const apiBase = getApiBase();
      const url = `${apiBase}/api/files/upload-progress/${taskId}`;
      const response = await fetch(url);

      if (!response.ok) {
        const errorText = await response.text();
        console.error(`[Poll] Error response: ${errorText}`);
        throw new Error(`Failed to fetch progress: ${response.status}`);
      }

      const progress = await response.json();

      // Check if UI elements still exist
      if (!progressBar || !progressPercent || !progressMessage) {
        console.error('[Poll] UI elements not found!');
        throw new Error('Progress UI elements not found');
      }

      // Update progress UI
      progressBar.style.width = `${progress.progress}%`;
      progressPercent.textContent = `${progress.progress}%`;
      progressMessage.textContent = progress.message;

      // Handle completed status
      if (progress.status === 'completed') {
        progressBar.classList.add('success');
        progressPercent.style.color = '#56ab2f';
        progressMessage.textContent = '✓ Upload complete!';
        if (pollInterval) clearInterval(pollInterval);

        // Remove from sessionStorage
        removeUploadTask(taskId);

        // Remove progress bar and refresh file list after 2 seconds
        setTimeout(() => {
          removeProgressBar(progressId);
          // Reset to first page to show newly uploaded file
          currentPage = 1;
          loadFileList();
        }, 2000);
      }
      // Handle failed status
      else if (progress.status === 'failed') {
        progressBar.classList.add('error');
        progressPercent.style.color = '#eb3349';
        progressMessage.textContent = `✗ Failed: ${progress.error || 'Unknown error'}`;
        progressMessage.style.color = '#eb3349';
        if (pollInterval) clearInterval(pollInterval);

        // Remove from sessionStorage
        removeUploadTask(taskId);

        // Remove progress bar after 5 seconds
        setTimeout(() => {
          removeProgressBar(progressId);
        }, 5000);
      }

    } catch (error) {
      console.error('[Poll] Error:', error);
      progressMessage.textContent = `✗ Error: ${error.message}`;
      progressMessage.style.color = '#eb3349';
      progressBar.classList.add('error');
      if (pollInterval) clearInterval(pollInterval);

      // Remove progress bar after 5 seconds
      setTimeout(() => {
        removeProgressBar(progressId);
      }, 5000);
    }
  };

  // Start polling immediately
  await doPoll();

  // Continue polling every 500ms if not completed or failed
  pollInterval = setInterval(async () => {
    const elem = document.getElementById(progressId);
    if (!elem) {
      clearInterval(pollInterval);
      return;
    }
    await doPoll();
  }, 500);
}
