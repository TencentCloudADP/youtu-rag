// Knowledge Base Detail Component

// Knowledge Base Detail state
let currentKBDetail = {
  id: null,
  name: '',
  description: '',
  files: [],
  configuration: {
    tools: {},
    selectedFiles: [],
    selectedQAFiles: [],
    dbConnections: []
  }
};

// File Selection Module State
let fileModalPagination = {
  allFiles: [],
  filteredFiles: [],
  currentPage: 1,
  itemsPerPage: 20
};

let selectedFilesPagination = {
  currentPage: 1,
  itemsPerPage: 10
};

// Selected files search and select mode state
let selectedFilesSearchQuery = '';
let isSelectedFilesSelectMode = false;
let selectedFilesForBatchDelete = new Set();

// File sorting state
let kbDetailFileNameSortOrder = null; // null: no sort, 'asc': ascending, 'desc': descending
let kbDetailFileStatusSortOrder = null; // null: no sort, 'asc': ascending, 'desc': descending

// Database Configuration Module State
let dbConnectionsPagination = {
  currentPage: 1,
  itemsPerPage: 10
};

let availableTables = [];
let currentDBPassword = '';

// Database tables search and select mode state
let dbTablesSearchQuery = '';
let isDBTablesSelectMode = false;
let selectedDBTablesForBatchDelete = new Set();

// Database tables sorting state
let dbTableNameSortOrder = null; // null: no sort, 'asc': ascending, 'desc': descending
let dbTypeSortOrder = null; // null: no sort, 'asc': ascending, 'desc': descending
let dbDatabaseNameSortOrder = null; // null: no sort, 'asc': ascending, 'desc': descending
let dbTableStatusSortOrder = null; // null: no sort, 'asc': ascending, 'desc': descending

// Q&A File Selection Module State
let selectedQAFilesPagination = {
  currentPage: 1,
  itemsPerPage: 10
};

// Selected Q&A files search and select mode state
let selectedQAFilesSearchQuery = '';
let isSelectedQAFilesSelectMode = false;
let selectedQAFilesForBatchDelete = new Set();

// Q&A file sorting state
let qaFileNameSortOrder = null; // null: no sort, 'asc': ascending, 'desc': descending
let qaFileStatusSortOrder = null; // null: no sort, 'asc': ascending, 'desc': descending

// Current active data source tab
let currentActiveTab = 'file'; // default to file tab

// ==================== Utility Functions ====================

// Update batch delete button state and text
function updateBatchDeleteButton(btnId, textId, selectionSize) {
  const batchDeleteText = document.getElementById(textId);
  const batchDeleteBtn = document.getElementById(btnId);

  if (batchDeleteText) {
    batchDeleteText.textContent = t('delete_selected', { count: selectionSize });
  }

  if (batchDeleteBtn) {
    if (selectionSize === 0) {
      batchDeleteBtn.disabled = true;
      batchDeleteBtn.style.backgroundColor = '#fef2f2';
      batchDeleteBtn.style.color = '#fca5a5';
      batchDeleteBtn.style.borderColor = '#fecaca';
      batchDeleteBtn.style.cursor = 'not-allowed';
    } else {
      batchDeleteBtn.disabled = false;
      batchDeleteBtn.style.backgroundColor = '#fee2e2';
      batchDeleteBtn.style.color = '#dc2626';
      batchDeleteBtn.style.borderColor = '#fca5a5';
      batchDeleteBtn.style.cursor = 'pointer';
    }
  }
}

// Render pagination HTML
function renderPaginationHTML(config) {
  const { currentPage, totalPages, itemsPerPage, pageOptions, startIndex, endIndex, totalItems, callbacks } = config;
  
  return `
    <div style="display: flex; align-items: center; gap: 10px;">
      <span style="font-size: 12px; color: var(--gray-7);">Items per page:</span>
      <select onchange="${callbacks.onPageSizeChange}(this.value)" style="padding: 4px 8px; border: 1px solid var(--gray-4); border-radius: 4px; font-size: 12px;">
        ${pageOptions.map(size => `<option value="${size}" ${itemsPerPage === size ? 'selected' : ''}>${size}</option>`).join('')}
      </select>
      <span style="font-size: 12px; color: var(--gray-7);">
        ${startIndex + 1}-${endIndex} / ${totalItems}
      </span>
    </div>
    <div style="display: flex; gap: 4px; ${totalPages <= 1 ? 'visibility: hidden;' : ''}">
      <button class="btn-pagination" onclick="${callbacks.onFirstPage}()" ${currentPage === 1 ? 'disabled' : ''}>«</button>
      <button class="btn-pagination" onclick="${callbacks.onPrevPage}()" ${currentPage === 1 ? 'disabled' : ''}>‹</button>
      <input type="number" min="1" max="${totalPages}" value="${currentPage}" onchange="${callbacks.onPageNumber}(this.value)" style="width: 50px; text-align: center; padding: 4px; border: 1px solid var(--gray-4); border-radius: 4px; font-size: 12px;">
      <button class="btn-pagination" onclick="${callbacks.onNextPage}()" ${currentPage === totalPages ? 'disabled' : ''}>›</button>
      <button class="btn-pagination" onclick="${callbacks.onLastPage}()" ${currentPage === totalPages ? 'disabled' : ''}>»</button>
    </div>
  `;
}

// Update sort indicator
function updateSortIndicator(indicatorId, sortOrder) {
  const indicator = document.getElementById(indicatorId);
  if (!indicator) return;

  if (sortOrder === 'asc') {
    indicator.textContent = '↑';
    indicator.style.color = 'var(--primary-blue, #3b82f6)';
  } else if (sortOrder === 'desc') {
    indicator.textContent = '↓';
    indicator.style.color = 'var(--primary-blue, #3b82f6)';
  } else {
    indicator.textContent = '⇅';
    indicator.style.color = 'var(--gray-6)';
  }
}

// Generic sort by name
function sortByName(items, order, nameExtractor = item => item) {
  return [...items].sort((a, b) => {
    const nameA = nameExtractor(a).toLowerCase();
    const nameB = nameExtractor(b).toLowerCase();
    return order === 'asc' ? nameA.localeCompare(nameB) : nameB.localeCompare(nameA);
  });
}

// Generic sort by status
function sortByStatus(items, statusMap, order, nameExtractor = item => item) {
  const statusPriority = {
    'pending': 1,
    'processing': 2,
    'completed': 3,
    'failed': 4,
    'unknown': 5
  };

  return [...items].sort((a, b) => {
    const statusA = statusMap[nameExtractor(a)]?.status || 'unknown';
    const statusB = statusMap[nameExtractor(b)]?.status || 'unknown';
    const priorityA = statusPriority[statusA] || 5;
    const priorityB = statusPriority[statusB] || 5;
    return order === 'asc' ? priorityA - priorityB : priorityB - priorityA;
  });
}

// Toggle select mode UI
function toggleSelectModeUI(config) {
  const { isSelectMode, selectBtnId, batchDeleteBtnId, batchDeleteTextId } = config;
  const selectBtn = document.getElementById(selectBtnId);
  const batchDeleteBtn = document.getElementById(batchDeleteBtnId);

  if (isSelectMode) {
    selectBtn.innerHTML = `
      <img src="assets/images/close.svg" alt="Cancel" style="width: 16px; height: 16px; margin-right: 6px; vertical-align: middle;">
      Cancel Selection
    `;
    batchDeleteBtn.style.display = 'inline-flex';
    updateBatchDeleteButton(batchDeleteBtnId, batchDeleteTextId, 0);
  } else {
    selectBtn.innerHTML = `
      <img src="assets/images/select.svg" alt="Select" style="width: 16px; height: 16px; margin-right: 6px; vertical-align: middle;">
      Batch Select
    `;
    batchDeleteBtn.style.display = 'none';
  }
}

// ==================== End Utility Functions ====================

// Initialize knowledge base detail page
window.initKnowledgeBaseDetail = async function() {
  console.log('[KB Detail] Initializing knowledge base detail page...');

  // Get KB ID from URL hash
  const hash = window.location.hash;
  const match = hash.match(/\/knowledge\/(.+)/);

  if (!match) {
    console.error('[KB Detail] No KB ID in URL');
    showToast(t('toast_kb_id_not_found'), 'error');
    router.navigate('/knowledge');
    return;
  }

  const kbId = decodeURIComponent(match[1]);
  console.log('[KB Detail] Loading KB:', kbId);

  await loadKBDetail(kbId);

  // Add language switch listener
  i18n.onChange(() => {
    // Re-render all dynamic text
    if (currentKBDetail.id) {
      renderSelectedFiles();
      renderDBConnectionsList();
      renderSelectedQAFiles();
    }
  });
};

// Load knowledge base detail
async function loadKBDetail(kbId) {
  try {
    // Show loading state
    document.getElementById('kb-detail-name').textContent = t('loading_kb_detail');
    document.getElementById('kb-detail-description').textContent = t('getting_kb_info');

    console.log('[KB Detail] Fetching KB details for:', kbId);
    const data = await API.getKnowledgeBase(kbId);
    console.log('[KB Detail] Received data:', data);

    currentKBDetail = {
      id: kbId,
      name: data.name || kbId,
      description: data.description || 'No description',
      files: data.files || [],
      configuration: data.configuration || {
        tools: {},
        selectedFiles: [],
        selectedQAFiles: [],
        dbConnections: []
      }
    };

    // Update UI
    document.getElementById('kb-detail-name').textContent = currentKBDetail.name;
    document.getElementById('kb-detail-description').textContent = currentKBDetail.description;

    // Render selected files
    renderSelectedFiles();

    // Render database connections (async - waits for status API)
    await renderDBConnectionsList();

    // Render selected Q&A files
    renderSelectedQAFiles();

    // Restore the last active tab
    restoreActiveTab();
  } catch (error) {
    console.error('[KB Detail] Load error:', error);
    showToast(t('toast_load_kb_failed', { error: error.message }), 'error');
  }
}

// Build knowledge base from detail page
async function buildKnowledgeBaseFromDetail() {
  if (!currentKBDetail.id) return;
  await buildKnowledgeBase(currentKBDetail.id);
  // Reload detail after build
  await loadKBDetail(currentKBDetail.id);
}

// Edit knowledge base from detail page
function editKnowledgeBaseFromDetail() {
  if (!currentKBDetail.id) return;
  router.navigate('/knowledge');
  setTimeout(() => {
    editKnowledgeBase(currentKBDetail.id);
  }, 500);
}

// Delete knowledge base from detail page
async function deleteKnowledgeBaseFromDetail() {
  if (!currentKBDetail.id) return;

  const confirmed = await confirmDialog(`Are you sure you want to delete the knowledge base "${currentKBDetail.name}"? This action cannot be undone.`);
  if (!confirmed) return;

  try {
    await API.deleteKnowledgeBase(currentKBDetail.id);
    showToast(t('toast_kb_deleted'), 'success');
    router.navigate('/knowledge');
  } catch (error) {
    showToast(t('toast_kb_delete_failed', { error: error.message }), 'error');
  }
}

// Data Source Tab Switching Function

// Switch data source tab (file, database, qa)
window.switchDataSourceTab = function(tab, event) {
  // Update tab buttons
  const tabButtons = document.querySelectorAll('.config-tabs .config-tab');
  tabButtons.forEach(btn => {
    btn.classList.remove('active');
  });
  if (event && event.target) {
    event.target.classList.add('active');
  }

  // Update tab content
  const allContents = document.querySelectorAll('#datasource-tab-file, #datasource-tab-database, #datasource-tab-qa');
  allContents.forEach(content => {
    content.classList.remove('active');
  });

  const targetContent = document.getElementById(`datasource-tab-${tab}`);
  if (targetContent) {
    targetContent.classList.add('active');
  }

  // Save the current active tab to localStorage
  currentActiveTab = tab;
  localStorage.setItem('kb-detail-active-tab', tab);
};

// Restore the last active tab from localStorage
function restoreActiveTab() {
  // Try to get the saved tab from localStorage
  const savedTab = localStorage.getItem('kb-detail-active-tab');
  const tabToActivate = savedTab || 'file'; // default to 'file' if no saved tab

  // Update the current active tab
  currentActiveTab = tabToActivate;

  // Find and click the corresponding tab button to activate it
  const tabButtons = document.querySelectorAll('.config-tabs .config-tab');
  tabButtons.forEach((btn, index) => {
    const tabMap = ['file', 'database', 'qa'];
    if (tabMap[index] === tabToActivate) {
      btn.click();
    }
  });
}

// File Selection Module Functions

// Show file selection modal
async function showFileSelectionModal() {
  try {
    showModal('file-selection-modal');

    const response = await API.request('/api/files/list');
    fileModalPagination.allFiles = response;
    fileModalPagination.filteredFiles = response;
    fileModalPagination.currentPage = 1;

    await renderFileModalList();
  } catch (error) {
    showToast(t('toast_load_files_failed_kb', { error: error.message }), 'error');
  }
}

// Render file modal list
async function renderFileModalList() {
  const container = document.getElementById('file-modal-list');
  const { filteredFiles, currentPage, itemsPerPage } = fileModalPagination;

  if (filteredFiles.length === 0) {
    container.innerHTML = `<div class="empty-state"><p>${t('no_files_found')}</p></div>`;
    document.getElementById('file-modal-pagination').style.display = 'none';
    return;
  }

  // Get file status (optional)
  let fileStatusMap = {};
  if (currentKBDetail.id) {
    try {
      fileStatusMap = await API.getKnowledgeBaseFileStatus(currentKBDetail.id);
    } catch (error) {
      console.warn('Failed to load file status:', error);
    }
  }

  // Paginate files
  const startIndex = (currentPage - 1) * itemsPerPage;
  const endIndex = startIndex + itemsPerPage;
  const pageFiles = filteredFiles.slice(startIndex, endIndex);

  container.innerHTML = pageFiles.map(file => {
    const fileName = file.name;
    const isSelected = currentKBDetail.configuration.selectedFiles.includes(fileName);
    const status = fileStatusMap[fileName];

    return `
      <div style="padding: 12px; border-bottom: 1px solid var(--gray-4); display: flex; align-items: center; gap: 12px;">
        <input type="checkbox" class="file-checkbox" value="${escapeHtml(fileName)}" ${isSelected ? 'checked' : ''}>
        <div style="flex: 1;">
          <div style="font-weight: 500;">${escapeHtml(fileName)}</div>
          <div style="font-size: 12px; color: var(--gray-7);">
            ${formatFileSize(file.size)}
          </div>
        </div>
        ${status ? renderStatusBadge(status.status) : ''}
      </div>
    `;
  }).join('');

  // Add event listeners to checkboxes to update selection count
  setTimeout(() => {
    document.querySelectorAll('#file-modal-list .file-checkbox').forEach(cb => {
      cb.addEventListener('change', function() {
        const fileName = this.value;
        const existingSelection = new Set(currentKBDetail.configuration.selectedFiles);

        if (this.checked) {
          existingSelection.add(fileName);
        } else {
          existingSelection.delete(fileName);
        }

        currentKBDetail.configuration.selectedFiles = Array.from(existingSelection);
        updateFileSelectionCount();
      });
    });
    updateFileSelectionCount();
  }, 0);

  // Always show pagination when there are files
  if (filteredFiles.length > 0) {
    renderFileModalPagination();
    document.getElementById('file-modal-pagination').style.display = 'flex';
  }
}

// Render status badge
function renderStatusBadge(status) {
  const config = {
    'completed': { class: 'badge badge-success', text: '✓ Completed' },
    'processing': { class: 'badge badge-warning', text: '⟳ Processing' },
    'pending': { class: 'badge badge-info', text: '○ Pending' },
    'failed': { class: 'badge badge-error', text: '✕ Failed' }
  }[status] || { class: 'badge', text: '− Not Added' };

  return `<span class="badge ${config.class}">${config.text}</span>`;
}

// Filter files in modal
function filterFilesInModal(query) {
  const lowerQuery = query.toLowerCase();
  fileModalPagination.filteredFiles = fileModalPagination.allFiles.filter(file =>
    file.name.toLowerCase().includes(lowerQuery)
  );
  fileModalPagination.currentPage = 1; // Reset to first page
  renderFileModalList();
}

// Render file modal pagination
function renderFileModalPagination() {
  const container = document.getElementById('file-modal-pagination');
  const { filteredFiles, currentPage, itemsPerPage } = fileModalPagination;
  const totalPages = Math.ceil(filteredFiles.length / itemsPerPage);
  const startIndex = (currentPage - 1) * itemsPerPage;
  const endIndex = Math.min(startIndex + itemsPerPage, filteredFiles.length);

  container.innerHTML = `
    <div style="display: flex; align-items: center; gap: 10px;">
      <span style="font-size: 12px; color: var(--gray-7);">Items per page:</span>
      <select onchange="changeFileModalPageSize(this.value)" style="padding: 4px 8px; border: 1px solid var(--gray-4); border-radius: 4px; font-size: 12px;">
        <option value="20" ${itemsPerPage === 20 ? 'selected' : ''}>20</option>
        <option value="50" ${itemsPerPage === 50 ? 'selected' : ''}>50</option>
        <option value="100" ${itemsPerPage === 100 ? 'selected' : ''}>100</option>
      </select>
      <span style="font-size: 12px; color: var(--gray-7);">
        ${startIndex + 1}-${endIndex} / ${filteredFiles.length}
      </span>
    </div>
    <div style="display: flex; gap: 4px; ${totalPages <= 1 ? 'visibility: hidden;' : ''}">
      <button class="btn-pagination" onclick="goToFirstFileModalPage()" ${currentPage === 1 ? 'disabled' : ''}>«</button>
      <button class="btn-pagination" onclick="changeFileModalPage(-1)" ${currentPage === 1 ? 'disabled' : ''}>‹</button>
      <input type="number" min="1" max="${totalPages}" value="${currentPage}" onchange="goToFileModalPageNumber(this.value)" style="width: 50px; text-align: center; padding: 4px; border: 1px solid var(--gray-4); border-radius: 4px; font-size: 12px;">
      <button class="btn-pagination" onclick="changeFileModalPage(1)" ${currentPage === totalPages ? 'disabled' : ''}>›</button>
      <button class="btn-pagination" onclick="goToLastFileModalPage()" ${currentPage === totalPages ? 'disabled' : ''}>»</button>
    </div>
  `;
}

// Go to first file modal page
function goToFirstFileModalPage() {
  fileModalPagination.currentPage = 1;
  renderFileModalList();
}

// Go to last file modal page
function goToLastFileModalPage() {
  const totalPages = Math.ceil(fileModalPagination.filteredFiles.length / fileModalPagination.itemsPerPage);
  fileModalPagination.currentPage = totalPages;
  renderFileModalList();
}

// Go to file modal page number
function goToFileModalPageNumber(pageNum) {
  const totalPages = Math.ceil(fileModalPagination.filteredFiles.length / fileModalPagination.itemsPerPage);
  const page = parseInt(pageNum);

  if (page >= 1 && page <= totalPages) {
    fileModalPagination.currentPage = page;
    renderFileModalList();
  }
}

// Change file modal page
function changeFileModalPage(delta) {
  const { currentPage, filteredFiles, itemsPerPage } = fileModalPagination;
  const totalPages = Math.ceil(filteredFiles.length / itemsPerPage);

  const newPage = currentPage + delta;
  if (newPage >= 1 && newPage <= totalPages) {
    fileModalPagination.currentPage = newPage;
    renderFileModalList();
  }
}

// Change file modal page size
function changeFileModalPageSize(size) {
  fileModalPagination.itemsPerPage = parseInt(size);
  fileModalPagination.currentPage = 1;
  renderFileModalList();
}

// Confirm file selection
function confirmFileSelection() {
  // Selection is already tracked in real-time, just close modal and update UI
  hideModal('file-selection-modal');
  renderSelectedFiles();
  showToast(t('toast_files_selected', { count: currentKBDetail.configuration.selectedFiles.length }), 'success');
}

// Render selected files list
async function renderSelectedFiles() {
  const container = document.getElementById('selected-files-container');
  const headerElement = document.getElementById('selected-files-header');
  const { currentPage, itemsPerPage } = selectedFilesPagination;

  // Get sorted and filtered files
  const sortedFiles = await getSortedAndFilteredFiles();

  if (sortedFiles.length === 0) {
    container.innerHTML = `<div class="empty-state"><p>${t('no_files_selected_yet')}</p></div>`;
    document.getElementById('selected-files-pagination').style.display = 'none';
    if (headerElement) headerElement.style.display = 'none';
    return;
  }

  // Show header when there are files
  if (headerElement) {
    headerElement.style.display = 'flex';
    // Update header content based on select mode
    updateSelectedFilesHeader();
  }

  // Get file status
  let fileStatusMap = {};
  if (currentKBDetail.id) {
    try {
      fileStatusMap = await API.getKnowledgeBaseFileStatus(currentKBDetail.id);
    } catch (error) {
      console.warn('Failed to load file status:', error);
    }
  }

  const startIndex = (currentPage - 1) * itemsPerPage;
  const endIndex = startIndex + itemsPerPage;
  const pageFiles = sortedFiles.slice(startIndex, endIndex);

  container.innerHTML = pageFiles.map((fileName) => {
    const status = fileStatusMap[fileName];
    const statusBadge = status ? renderStatusBadge(status.status) : '<span class="badge" style="background-color: var(--gray-3); color: var(--gray-7);">− Not Added</span>';
    const isChecked = selectedFilesForBatchDelete.has(fileName);

    return `
      <div class="selected-item">
        <div style="display: flex; align-items: center; gap: 8px; flex: 1; min-width: 0;">
          ${isSelectedFilesSelectMode ? `
            <input type="checkbox" class="selected-file-checkbox" value="${escapeHtml(fileName)}" ${isChecked ? 'checked' : ''} onchange="toggleSelectedFileForBatchDelete('${escapeHtml(fileName)}', this.checked)">
          ` : ''}
          <img src="/assets/images/document-text.svg" alt="Document-text" style="width: 16px; height: 16px;">
          <span style="overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">${escapeHtml(fileName)}</span>
        </div>
        <div style="width: 120px; display: flex; justify-content: center; align-items: center;">
          ${statusBadge}
        </div>
        <div style="width: 100px; display: flex; justify-content: center; align-items: center;">
          <button class="btn btn-small btn-secondary" onclick="removeSelectedFileByName('${escapeHtml(fileName)}')">
            <img src="assets/images/delete.svg" alt="Delete" style="width: 16px; height: 16px; filter: grayscale(100%) brightness(0.6);">
          </button>
        </div>
      </div>
    `;
  }).join('');

  // Always show pagination when there are files
  renderSelectedFilesPagination(sortedFiles);
  document.getElementById('selected-files-pagination').style.display = 'flex';

  // Update select all checkbox state after rendering file list
  if (isSelectedFilesSelectMode) {
    updateSelectAllCheckboxState();
  }
}

// Update selected files header based on select mode
function updateSelectedFilesHeader() {
  const headerElement = document.getElementById('selected-files-header');
  if (!headerElement) return;

  if (isSelectedFilesSelectMode) {
    // In select mode, show checkbox in header
    headerElement.innerHTML = `
      <div style="display: flex; align-items: center; gap: 8px; flex: 1; min-width: 0;">
        <input type="checkbox" id="select-all-current-page-checkbox" onchange="toggleSelectAllCurrentPage(this.checked)">
        <span class="sortable" style="cursor: pointer; user-select: none; display: flex; align-items: center; gap: 6px;" onclick="toggleFileNameSort()">
          File Name
          <span id="filename-sort-indicator" style="font-size: 12px; color: var(--gray-6);">⇅</span>
        </span>
      </div>
      <div style="width: 120px; text-align: center;">
        <span class="sortable" style="cursor: pointer; user-select: none; display: inline-flex; align-items: center; gap: 6px;" onclick="toggleFileStatusSort()">
          Processing Status
          <span id="status-sort-indicator" style="font-size: 12px; color: var(--gray-6);">⇅</span>
        </span>
      </div>
      <div style="width: 100px; text-align: center;">Actions</div>
    `;
    // Update checkbox state
    updateSelectAllCheckboxState();
  } else {
    // Normal mode, no checkbox
    headerElement.innerHTML = `
      <div style="flex: 1; min-width: 0;">
        <span class="sortable" style="cursor: pointer; user-select: none; display: inline-flex; align-items: center; gap: 6px;" onclick="toggleFileNameSort()">
          File Name
          <span id="filename-sort-indicator" style="font-size: 12px; color: var(--gray-6);">⇅</span>
        </span>
      </div>
      <div style="width: 120px; text-align: center;">
        <span class="sortable" style="cursor: pointer; user-select: none; display: inline-flex; align-items: center; gap: 6px;" onclick="toggleFileStatusSort()">
          Processing Status
          <span id="status-sort-indicator" style="font-size: 12px; color: var(--gray-6);">⇅</span>
        </span>
      </div>
      <div style="width: 100px; text-align: center;">Actions</div>
    `;
  }

  // Update sort indicators
  updateFileNameSortIndicator();
  updateFileStatusSortIndicator();
}

// Toggle select all files on current page
async function toggleSelectAllCurrentPage(checked) {
  const { currentPage, itemsPerPage } = selectedFilesPagination;

  // Get sorted and filtered files
  const sortedFiles = await getSortedAndFilteredFiles();

  const startIndex = (currentPage - 1) * itemsPerPage;
  const endIndex = startIndex + itemsPerPage;
  const pageFiles = sortedFiles.slice(startIndex, endIndex);

  if (checked) {
    // Select all files on current page
    pageFiles.forEach(fileName => selectedFilesForBatchDelete.add(fileName));
  } else {
    // Deselect all files on current page
    pageFiles.forEach(fileName => selectedFilesForBatchDelete.delete(fileName));
  }

  // Update batch delete button text and state
  updateBatchDeleteButton('selected-files-batch-delete-btn', 'selected-files-batch-delete-text', selectedFilesForBatchDelete.size);

  // Re-render to update checkboxes
  renderSelectedFiles();
}

// Update the "select all" checkbox state based on current selections
function updateSelectAllCheckboxState() {
  const checkbox = document.getElementById('select-all-current-page-checkbox');
  if (!checkbox) return;

  // Get all file checkboxes on the current page
  const fileCheckboxes = document.querySelectorAll('.selected-file-checkbox');

  if (fileCheckboxes.length === 0) {
    checkbox.checked = false;
    return;
  }

  // Check if all checkboxes are checked
  const allChecked = Array.from(fileCheckboxes).every(cb => cb.checked);
  checkbox.checked = allChecked;
}

// Remove selected file by name
function removeSelectedFileByName(fileName) {
  currentKBDetail.configuration.selectedFiles = currentKBDetail.configuration.selectedFiles.filter(f => f !== fileName);
  renderSelectedFiles();
  showToast(t('toast_file_removed_kb'), 'info');
}

// Handle selected files search
function handleSelectedFilesSearch(query) {
  selectedFilesSearchQuery = query;
  selectedFilesPagination.currentPage = 1;
  renderSelectedFiles();
}

// Toggle selected files select mode
function toggleSelectedFilesSelectMode() {
  isSelectedFilesSelectMode = !isSelectedFilesSelectMode;
  
  toggleSelectModeUI({
    isSelectMode: isSelectedFilesSelectMode,
    selectBtnId: 'selected-files-select-btn',
    batchDeleteBtnId: 'selected-files-batch-delete-btn',
    batchDeleteTextId: 'selected-files-batch-delete-text'
  });

  if (!isSelectedFilesSelectMode) {
    selectedFilesForBatchDelete.clear();
  }

  renderSelectedFiles();
}

// Toggle selected file for batch delete
function toggleSelectedFileForBatchDelete(fileName, checked) {
  if (checked) {
    selectedFilesForBatchDelete.add(fileName);
  } else {
    selectedFilesForBatchDelete.delete(fileName);
  }

  updateBatchDeleteButton('selected-files-batch-delete-btn', 'selected-files-batch-delete-text', selectedFilesForBatchDelete.size);
  updateSelectAllCheckboxState();
}

// Batch delete selected files
async function batchDeleteSelectedFiles() {
  const count = selectedFilesForBatchDelete.size;
  if (count === 0) {
    showToast(t('toast_select_files_first'), 'warning');
    return;
  }

  const confirmed = await confirmDialog(`Are you sure you want to delete the selected ${count} file(s)?`);
  if (!confirmed) return;

  // Remove selected files from configuration
  const filesToDelete = Array.from(selectedFilesForBatchDelete);
  currentKBDetail.configuration.selectedFiles = currentKBDetail.configuration.selectedFiles.filter(
    f => !filesToDelete.includes(f)
  );

  // Clear selection and exit select mode
  selectedFilesForBatchDelete.clear();
  isSelectedFilesSelectMode = false;
  const selectBtn = document.getElementById('selected-files-select-btn');
  const batchDeleteBtn = document.getElementById('selected-files-batch-delete-btn');
  selectBtn.innerHTML = `
    <img src="assets/images/select.svg" alt="Select" style="width: 16px; height: 16px; margin-right: 6px; vertical-align: middle;">
    Batch Select
  `;
  batchDeleteBtn.style.display = 'none';

  renderSelectedFiles();
  showToast(t('toast_files_deleted', { count }), 'success');
}

// Render selected files pagination
function renderSelectedFilesPagination(filteredFiles) {
  const container = document.getElementById('selected-files-pagination');
  const { currentPage, itemsPerPage } = selectedFilesPagination;
  const totalPages = Math.ceil(filteredFiles.length / itemsPerPage);
  const startIndex = (currentPage - 1) * itemsPerPage;
  const endIndex = Math.min(startIndex + itemsPerPage, filteredFiles.length);

  container.innerHTML = renderPaginationHTML({
    currentPage,
    totalPages,
    itemsPerPage,
    pageOptions: [5, 10, 20, 50],
    startIndex,
    endIndex,
    totalItems: filteredFiles.length,
    callbacks: {
      onPageSizeChange: 'changeSelectedFilesPerPage',
      onFirstPage: 'goToFirstSelectedFilesPage',
      onPrevPage: 'changeSelectedFilesPage.bind(null, -1)',
      onPageNumber: 'goToSelectedFilesPageNumber',
      onNextPage: 'changeSelectedFilesPage.bind(null, 1)',
      onLastPage: 'goToLastSelectedFilesPage'
    }
  });
}

// Change selected files per page
function changeSelectedFilesPerPage(size) {
  selectedFilesPagination.itemsPerPage = parseInt(size);
  selectedFilesPagination.currentPage = 1;
  renderSelectedFiles();
}

// Go to first selected files page
function goToFirstSelectedFilesPage() {
  selectedFilesPagination.currentPage = 1;
  renderSelectedFiles();
}

// Go to last selected files page
async function goToLastSelectedFilesPage() {
  const sortedFiles = await getSortedAndFilteredFiles();
  const totalPages = Math.ceil(sortedFiles.length / selectedFilesPagination.itemsPerPage);
  selectedFilesPagination.currentPage = totalPages;
  renderSelectedFiles();
}

// Go to selected files page number
async function goToSelectedFilesPageNumber(pageNum) {
  const sortedFiles = await getSortedAndFilteredFiles();
  const totalPages = Math.ceil(sortedFiles.length / selectedFilesPagination.itemsPerPage);
  const page = parseInt(pageNum);

  if (page >= 1 && page <= totalPages) {
    selectedFilesPagination.currentPage = page;
    renderSelectedFiles();
  }
}

// Change selected files page
async function changeSelectedFilesPage(delta) {
  const { currentPage } = selectedFilesPagination;
  const sortedFiles = await getSortedAndFilteredFiles();
  const totalPages = Math.ceil(sortedFiles.length / selectedFilesPagination.itemsPerPage);

  const newPage = currentPage + delta;
  if (newPage >= 1 && newPage <= totalPages) {
    selectedFilesPagination.currentPage = newPage;
    renderSelectedFiles();
  }
}

// Database Configuration Module Functions

// Handle database type change
function handleDBTypeChange() {
  const dbType = document.getElementById('db-type').value;
  const mysqlFields = document.getElementById('mysql-fields');
  const sqliteFields = document.getElementById('sqlite-fields');
  const tableArea = document.getElementById('table-selection-area');
  const testStatus = document.getElementById('db-test-status');

  if (dbType === 'mysql') {
    mysqlFields.style.display = 'block';
    sqliteFields.style.display = 'none';
  } else {
    mysqlFields.style.display = 'none';
    sqliteFields.style.display = 'block';
  }

  tableArea.style.display = 'none';
  testStatus.style.display = 'none';
}

// Test database connection
async function testDatabaseConnection() {
  const dbType = document.getElementById('db-type').value;
  const testStatus = document.getElementById('db-test-status');

  testStatus.style.display = 'block';
  testStatus.innerHTML = `<div class="spinner"></div> ${t('testing_connection')}`;

  try {
    let dbConfig = { db_type: dbType };

    if (dbType === 'mysql') {
      dbConfig.host = document.getElementById('db-host').value;
      dbConfig.port = parseInt(document.getElementById('db-port').value);
      dbConfig.database = document.getElementById('db-database').value;
      dbConfig.username = document.getElementById('db-username').value;
      dbConfig.password = document.getElementById('db-password').value;
      currentDBPassword = dbConfig.password;
    } else {
      dbConfig.file_path = document.getElementById('db-file-path').value;
      dbConfig.database = 'sqlite_db';
    }

    const result = await API.testDatabaseConnection(dbConfig);

    if (result.success) {
      availableTables = result.tables || [];
      testStatus.innerHTML = `<span style="color: var(--success);">${t('connection_success', { count: availableTables.length })}</span>`;

      renderTableList(availableTables);
      document.getElementById('table-selection-area').style.display = 'block';
    } else {
      testStatus.innerHTML = `<span style="color: var(--error);">${t('connection_failed', { error: result.message })}</span>`;
    }
  } catch (error) {
    testStatus.innerHTML = `<span style="color: var(--error);">${t('connection_failed', { error: error.message })}</span>`;
  }
}

// Render table list
function renderTableList(tables) {
  const container = document.getElementById('table-list');

  container.innerHTML = tables.map(table => `
    <label style="display: flex; align-items: center; gap: 8px; cursor: pointer;">
      <input type="checkbox" class="table-checkbox" value="${escapeHtml(table)}">
      <span>${escapeHtml(table)}</span>
    </label>
  `).join('');
}

// Toggle all tables selection
function toggleAllTables(checked) {
  document.querySelectorAll('.table-checkbox').forEach(cb => {
    cb.checked = checked;
  });
}

// Add database connection
async function addDatabaseConnection() {
  const dbType = document.getElementById('db-type').value;
  const selectedTables = Array.from(document.querySelectorAll('.table-checkbox:checked'))
    .map(cb => cb.value);

  if (selectedTables.length === 0) {
    showToast(t('toast_select_at_least_one_table'), 'warning');
    return;
  }

  let connection = {
    id: Date.now(),
    type: dbType,
    tables: selectedTables
  };

  if (dbType === 'mysql') {
    connection.host = document.getElementById('db-host').value;
    connection.port = parseInt(document.getElementById('db-port').value);
    connection.database = document.getElementById('db-database').value;
    connection.username = document.getElementById('db-username').value;
    connection.password = currentDBPassword;
    connection.connectionString = `mysql://${connection.username}@${connection.host}:${connection.port}/${connection.database}`;
  } else {
    connection.file_path = document.getElementById('db-file-path').value;
    connection.database = connection.file_path.split(/[/\\]/).pop();
    connection.connectionString = `sqlite:///${connection.file_path}`;
  }

  currentKBDetail.configuration.dbConnections.push(connection);
  await renderDBConnectionsList();

  // Clear table selection
  document.querySelectorAll('.table-checkbox').forEach(cb => cb.checked = false);
  document.getElementById('select-all-tables').checked = false;

  showToast(t('toast_db_connection_added', { count: selectedTables.length }), 'success');
}

// Render database connections list
async function renderDBConnectionsList() {
  const container = document.getElementById('db-connections-list');
  const headerElement = document.getElementById('db-tables-header');
  const { currentPage, itemsPerPage } = dbConnectionsPagination;

  // Get file/database status from API
  let dbStatusMap = {};
  if (currentKBDetail.id) {
    try {
      dbStatusMap = await API.getKnowledgeBaseFileStatus(currentKBDetail.id);
      console.log('[DB Connections] Loaded status map:', dbStatusMap);
    } catch (error) {
      console.warn('Failed to load database status:', error);
    }
  }

  // Get sorted and filtered tables with actual status
  const sortedTables = getSortedAndFilteredDBTables(dbStatusMap);

  if (sortedTables.length === 0) {
    container.innerHTML = '<div class="empty-state"><p>No database tables added yet</p></div>';
    document.getElementById('db-connections-pagination').style.display = 'none';
    if (headerElement) headerElement.style.display = 'none';
    return;
  }

  // Show header when there are tables
  if (headerElement) {
    headerElement.style.display = 'flex';
    updateDBTablesHeader();
  }

  const startIndex = (currentPage - 1) * itemsPerPage;
  const endIndex = startIndex + itemsPerPage;
  const pageTables = sortedTables.slice(startIndex, endIndex);

  container.innerHTML = pageTables.map((table, pageIndex) => {
    const globalIndex = startIndex + pageIndex;
    const statusBadge = renderStatusBadge(table.status);
    const isChecked = selectedDBTablesForBatchDelete.has(globalIndex);
    const dbTypeDisplay = table.type === 'mysql' ? 'MySQL' : 'SQLite';
    const dbTypeColor = table.type === 'mysql' ? '#f97316' : '#0891b2'; // MySQL: orange, SQLite: cyan

    return `
      <div class="selected-item" style="min-width: 620px;">
        <div style="display: flex; align-items: center; gap: 8px; flex: 1; min-width: 0;">
          ${isDBTablesSelectMode ? `
            <input type="checkbox" class="db-table-checkbox" value="${globalIndex}" ${isChecked ? 'checked' : ''} onchange="toggleDBTableForBatchDelete(${globalIndex}, this.checked)">
          ` : ''}
          <img src="/assets/images/table.svg" alt="Database" style="width: 16px; height: 16px;">
          <span style="overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" title="${escapeHtml(table.tableName)}">${escapeHtml(table.tableName)}</span>
        </div>
        <div style="width: 100px; display: flex; justify-content: center; align-items: center; flex-shrink: 0;">
          <span class="badge" style="background-color: ${dbTypeColor}20; color: ${dbTypeColor}; border: 1px solid ${dbTypeColor}40; font-size: 12px;">${dbTypeDisplay}</span>
        </div>
        <div style="width: 200px; display: flex; justify-content: center; align-items: center; flex-shrink: 0;">
          <span style="font-size: 13px; color: var(--gray-7); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 190px;" title="${escapeHtml(table.databaseName)}">${escapeHtml(table.databaseName)}</span>
        </div>
        <div style="width: 120px; display: flex; justify-content: center; align-items: center; flex-shrink: 0;">
          ${statusBadge}
        </div>
        <div style="width: 100px; display: flex; justify-content: center; align-items: center; flex-shrink: 0;">
          <button class="btn btn-small btn-secondary" onclick="removeDBTableByIndex(${globalIndex})">
            <img src="assets/images/delete.svg" alt="Delete" style="width: 16px; height: 16px; filter: grayscale(100%) brightness(0.6);">
          </button>
        </div>
      </div>
    `;
  }).join('');

  // Always show pagination when there are tables
  if (sortedTables.length > 0) {
    renderDBConnectionsPagination(sortedTables);
    document.getElementById('db-connections-pagination').style.display = 'flex';
  }

  // Update select all checkbox state after rendering table list
  if (isDBTablesSelectMode) {
    updateDBTablesSelectAllCheckboxState();
  }
}

// Remove database connection
async function removeDBConnection(index) {
  currentKBDetail.configuration.dbConnections.splice(index, 1);
  await renderDBConnectionsList();
  showToast('Database connection removed', 'info');
}

// Remove database table by global index
async function removeDBTableByIndex(globalIndex) {
  const sortedTables = getSortedAndFilteredDBTables();

  if (globalIndex < 0 || globalIndex >= sortedTables.length) {
    showToast('Invalid table index', 'error');
    return;
  }

  const table = sortedTables[globalIndex];
  const conn = currentKBDetail.configuration.dbConnections[table.connectionIndex];

  if (!conn) {
    showToast('Database connection not found', 'error');
    return;
  }

  // Remove table from connection
  conn.tables = conn.tables.filter(t => t !== table.tableName);

  // If connection has no more tables, remove the connection entirely
  if (conn.tables.length === 0) {
    currentKBDetail.configuration.dbConnections.splice(table.connectionIndex, 1);
  }

  await renderDBConnectionsList();
  showToast('Table removed', 'info');
}

// Render database connections pagination
function renderDBConnectionsPagination(sortedTables) {
  const container = document.getElementById('db-connections-pagination');
  const { currentPage, itemsPerPage } = dbConnectionsPagination;
  const totalPages = Math.ceil(sortedTables.length / itemsPerPage);
  const startIndex = (currentPage - 1) * itemsPerPage;
  const endIndex = Math.min(startIndex + itemsPerPage, sortedTables.length);

  container.innerHTML = renderPaginationHTML({
    currentPage,
    totalPages,
    itemsPerPage,
    pageOptions: [5, 10, 20, 50],
    startIndex,
    endIndex,
    totalItems: sortedTables.length,
    callbacks: {
      onPageSizeChange: 'changeDBTablesPerPage',
      onFirstPage: 'goToFirstDBTablesPage',
      onPrevPage: 'changeDBConnectionsPage.bind(null, -1)',
      onPageNumber: 'goToDBTablesPageNumber',
      onNextPage: 'changeDBConnectionsPage.bind(null, 1)',
      onLastPage: 'goToLastDBTablesPage'
    }
  });
}

// Change database connections page
async function changeDBConnectionsPage(delta) {
  const { currentPage } = dbConnectionsPagination;
  const sortedTables = getSortedAndFilteredDBTables();
  const totalPages = Math.ceil(sortedTables.length / dbConnectionsPagination.itemsPerPage);

  const newPage = currentPage + delta;
  if (newPage >= 1 && newPage <= totalPages) {
    dbConnectionsPagination.currentPage = newPage;
    await renderDBConnectionsList();
  }
}

// Go to first database tables page
async function goToFirstDBTablesPage() {
  dbConnectionsPagination.currentPage = 1;
  await renderDBConnectionsList();
}

// Go to last database tables page
async function goToLastDBTablesPage() {
  const sortedTables = getSortedAndFilteredDBTables();
  const totalPages = Math.ceil(sortedTables.length / dbConnectionsPagination.itemsPerPage);
  dbConnectionsPagination.currentPage = totalPages;
  await renderDBConnectionsList();
}

// Go to database tables page number
async function goToDBTablesPageNumber(pageNum) {
  const sortedTables = getSortedAndFilteredDBTables();
  const totalPages = Math.ceil(sortedTables.length / dbConnectionsPagination.itemsPerPage);
  const page = parseInt(pageNum);

  if (page >= 1 && page <= totalPages) {
    dbConnectionsPagination.currentPage = page;
    await renderDBConnectionsList();
  }
}

// Change database tables per page
async function changeDBTablesPerPage(size) {
  dbConnectionsPagination.itemsPerPage = parseInt(size);
  dbConnectionsPagination.currentPage = 1;
  await renderDBConnectionsList();
}

// Handle database tables search
async function handleDBTablesSearch(query) {
  dbTablesSearchQuery = query;
  dbConnectionsPagination.currentPage = 1;
  await renderDBConnectionsList();
}

// Toggle database tables select mode
async function toggleDBTablesSelectMode() {
  isDBTablesSelectMode = !isDBTablesSelectMode;
  
  toggleSelectModeUI({
    isSelectMode: isDBTablesSelectMode,
    selectBtnId: 'db-tables-select-btn',
    batchDeleteBtnId: 'db-tables-batch-delete-btn',
    batchDeleteTextId: 'db-tables-batch-delete-text'
  });

  if (!isDBTablesSelectMode) {
    selectedDBTablesForBatchDelete.clear();
  }

  await renderDBConnectionsList();
}

// Toggle selected database table for batch delete
function toggleDBTableForBatchDelete(tableIndex, checked) {
  if (checked) {
    selectedDBTablesForBatchDelete.add(tableIndex);
  } else {
    selectedDBTablesForBatchDelete.delete(tableIndex);
  }

  updateBatchDeleteButton('db-tables-batch-delete-btn', 'db-tables-batch-delete-text', selectedDBTablesForBatchDelete.size);
  updateDBTablesSelectAllCheckboxState();
}

// Batch delete selected database tables
async function batchDeleteDBTables() {
  const count = selectedDBTablesForBatchDelete.size;
  if (count === 0) {
    showToast('Please select tables to delete first', 'warning');
    return;
  }

  const confirmed = await confirmDialog(`Are you sure you want to delete the selected ${count} table(s)?`);
  if (!confirmed) return;

  // Get the sorted tables first
  const sortedTables = getSortedAndFilteredDBTables();

  // Get all tables to delete
  const tablesToDelete = Array.from(selectedDBTablesForBatchDelete)
    .sort((a, b) => b - a) // Sort in descending order
    .map(index => sortedTables[index])
    .filter(table => table !== undefined);

  // Group tables by connection index for efficient deletion
  const tablesByConnection = new Map();
  tablesToDelete.forEach(table => {
    if (!tablesByConnection.has(table.connectionIndex)) {
      tablesByConnection.set(table.connectionIndex, []);
    }
    tablesByConnection.get(table.connectionIndex).push(table.tableName);
  });

  // Remove tables from their connections
  tablesByConnection.forEach((tableNames, connIndex) => {
    const conn = currentKBDetail.configuration.dbConnections[connIndex];
    if (conn) {
      conn.tables = conn.tables.filter(t => !tableNames.includes(t));
    }
  });

  // Remove connections that no longer have any tables
  currentKBDetail.configuration.dbConnections = currentKBDetail.configuration.dbConnections.filter(
    conn => conn.tables && conn.tables.length > 0
  );

  // Clear selection and exit select mode
  selectedDBTablesForBatchDelete.clear();
  isDBTablesSelectMode = false;
  const selectBtn = document.getElementById('db-tables-select-btn');
  const batchDeleteBtn = document.getElementById('db-tables-batch-delete-btn');
  selectBtn.innerHTML = `
    <img src="assets/images/select.svg" alt="Select" style="width: 16px; height: 16px; margin-right: 6px; vertical-align: middle;">
    Batch Select
  `;
  batchDeleteBtn.style.display = 'none';

  await renderDBConnectionsList();
  showToast(`Deleted ${count} table(s)`, 'success');
}

// Update the "select all" checkbox state for database tables
function updateDBTablesSelectAllCheckboxState() {
  const checkbox = document.getElementById('select-all-db-tables-checkbox');
  if (!checkbox) return;

  const tableCheckboxes = document.querySelectorAll('.db-table-checkbox');
  if (tableCheckboxes.length === 0) {
    checkbox.checked = false;
    return;
  }

  const allChecked = Array.from(tableCheckboxes).every(cb => cb.checked);
  checkbox.checked = allChecked;
}

// Toggle select all database tables on current page
function toggleSelectAllDBTablesCurrentPage(checked) {
  const tableCheckboxes = document.querySelectorAll('.db-table-checkbox');

  tableCheckboxes.forEach(cb => {
    const tableIndex = parseInt(cb.value);
    if (checked) {
      selectedDBTablesForBatchDelete.add(tableIndex);
      cb.checked = true;
    } else {
      selectedDBTablesForBatchDelete.delete(tableIndex);
      cb.checked = false;
    }
  });

  updateBatchDeleteButton('db-tables-batch-delete-btn', 'db-tables-batch-delete-text', selectedDBTablesForBatchDelete.size);
}

// Get sorted and filtered database tables list
function getSortedAndFilteredDBTables(statusMap = {}) {
  const allTables = [];

  // Check if dbConnections exists and is an array
  if (!currentKBDetail.configuration.dbConnections || !Array.isArray(currentKBDetail.configuration.dbConnections)) {
    return [];
  }

  // Process each connection
  currentKBDetail.configuration.dbConnections.forEach((conn, connIndex) => {
    // Extract database type
    const dbType = conn.type || '';

    // Extract database name based on database type
    let databaseName = '';
    if (dbType === 'mysql') {
      // For MySQL, use the database field
      databaseName = conn.database || '';
    } else if (dbType === 'sqlite') {
      // For SQLite, extract the filename from file_path
      const filePath = conn.file_path || '';
      if (filePath) {
        const pathParts = filePath.split(/[/\\]/); // Split by / or \
        databaseName = pathParts[pathParts.length - 1]; // Get last part (filename)
      }
    }

    // Process each table in the connection's tables array
    const tables = conn.tables || [];
    tables.forEach(tableName => {
      if (tableName) {
        // Get actual status from statusMap (key format: "database:tableName")
        const statusKey = `database:${tableName}`;
        const tableStatus = statusMap[statusKey];
        const status = tableStatus?.status || 'pending';

        allTables.push({
          tableName,
          databaseName,
          connectionIndex: connIndex,
          type: dbType,
          status: status,
          chunksCreated: tableStatus?.chunks_created || 0
        });
      }
    });
  });

  // Filter tables based on search query
  const filteredTables = allTables.filter(table =>
    table.tableName.toLowerCase().includes(dbTablesSearchQuery.toLowerCase())
  );

  // Apply sorting if needed
  let sortedTables = [...filteredTables];
  if (dbTableNameSortOrder) {
    sortedTables = sortDBTablesByName(sortedTables, dbTableNameSortOrder);
  } else if (dbTypeSortOrder) {
    sortedTables = sortDBTablesByType(sortedTables, dbTypeSortOrder);
  } else if (dbDatabaseNameSortOrder) {
    sortedTables = sortDBTablesByDatabaseName(sortedTables, dbDatabaseNameSortOrder);
  } else if (dbTableStatusSortOrder) {
    sortedTables = sortDBTablesByStatus(sortedTables, dbTableStatusSortOrder);
  }

  return sortedTables;
}

// Toggle table name sort order
async function toggleTableNameSort() {
  // Clear other sorts
  dbTypeSortOrder = null;
  dbDatabaseNameSortOrder = null;
  dbTableStatusSortOrder = null;

  // Cycle through: null -> asc -> desc -> null
  if (dbTableNameSortOrder === null) {
    dbTableNameSortOrder = 'asc';
  } else if (dbTableNameSortOrder === 'asc') {
    dbTableNameSortOrder = 'desc';
  } else {
    dbTableNameSortOrder = null;
  }

  await renderDBConnectionsList();
}

// Toggle database type sort order
async function toggleDBTypeSort() {
  // Clear other sorts
  dbTableNameSortOrder = null;
  dbDatabaseNameSortOrder = null;
  dbTableStatusSortOrder = null;

  // Cycle through: null -> asc -> desc -> null
  if (dbTypeSortOrder === null) {
    dbTypeSortOrder = 'asc';
  } else if (dbTypeSortOrder === 'asc') {
    dbTypeSortOrder = 'desc';
  } else {
    dbTypeSortOrder = null;
  }

  await renderDBConnectionsList();
}

// Toggle database name sort order
async function toggleDatabaseNameSort() {
  // Clear other sorts
  dbTableNameSortOrder = null;
  dbTypeSortOrder = null;
  dbTableStatusSortOrder = null;

  // Cycle through: null -> asc -> desc -> null
  if (dbDatabaseNameSortOrder === null) {
    dbDatabaseNameSortOrder = 'asc';
  } else if (dbDatabaseNameSortOrder === 'asc') {
    dbDatabaseNameSortOrder = 'desc';
  } else {
    dbDatabaseNameSortOrder = null;
  }

  await renderDBConnectionsList();
}

// Toggle table status sort order
async function toggleTableStatusSort() {
  // Clear other sorts
  dbTableNameSortOrder = null;
  dbTypeSortOrder = null;
  dbDatabaseNameSortOrder = null;

  // Cycle through: null -> asc -> desc -> null
  if (dbTableStatusSortOrder === null) {
    dbTableStatusSortOrder = 'asc';
  } else if (dbTableStatusSortOrder === 'asc') {
    dbTableStatusSortOrder = 'desc';
  } else {
    dbTableStatusSortOrder = null;
  }

  await renderDBConnectionsList();
}

// Update table name sort indicator
function updateTableNameSortIndicator() {
  updateSortIndicator('table-name-sort-indicator', dbTableNameSortOrder);
}

// Update database type sort indicator
function updateDBTypeSortIndicator() {
  updateSortIndicator('db-type-sort-indicator', dbTypeSortOrder);
}

// Update database name sort indicator
function updateDatabaseNameSortIndicator() {
  updateSortIndicator('database-name-sort-indicator', dbDatabaseNameSortOrder);
}

// Update table status sort indicator
function updateTableStatusSortIndicator() {
  updateSortIndicator('table-status-sort-indicator', dbTableStatusSortOrder);
}

// Sort database tables by name
function sortDBTablesByName(tables, order) {
  return sortByName(tables, order, table => table.tableName);
}

// Sort database tables by type
function sortDBTablesByType(tables, order) {
  return sortByName(tables, order, table => table.type);
}

// Sort database tables by database name
function sortDBTablesByDatabaseName(tables, order) {
  return sortByName(tables, order, table => table.databaseName);
}

// Sort database tables by status
function sortDBTablesByStatus(tables, order) {
  return sortByStatus(tables, {}, order, table => ({ status: table.status }));
}

// Update database tables header based on select mode
function updateDBTablesHeader() {
  const headerElement = document.getElementById('db-tables-header');
  if (!headerElement) return;

  if (isDBTablesSelectMode) {
    // In select mode, show checkbox in header
    headerElement.innerHTML = `
      <div style="display: flex; align-items: center; gap: 8px; flex: 1; min-width: 0;">
        <input type="checkbox" id="select-all-db-tables-checkbox" onchange="toggleSelectAllDBTablesCurrentPage(this.checked)">
        <span class="sortable" style="cursor: pointer; user-select: none; display: flex; align-items: center; gap: 6px;" onclick="toggleTableNameSort()">
          Table Name
          <span id="table-name-sort-indicator" style="font-size: 12px; color: var(--gray-6);">⇅</span>
        </span>
      </div>
      <div style="width: 100px; text-align: center; flex-shrink: 0;">
        <span class="sortable" style="cursor: pointer; user-select: none; display: inline-flex; align-items: center; gap: 6px;" onclick="toggleDBTypeSort()">
          Database Type
          <span id="db-type-sort-indicator" style="font-size: 12px; color: var(--gray-6);">⇅</span>
        </span>
      </div>
      <div style="width: 200px; text-align: center; flex-shrink: 0;">
        <span class="sortable" style="cursor: pointer; user-select: none; display: inline-flex; align-items: center; gap: 6px;" onclick="toggleDatabaseNameSort()">
          Database Name
          <span id="database-name-sort-indicator" style="font-size: 12px; color: var(--gray-6);">⇅</span>
        </span>
      </div>
      <div style="width: 120px; text-align: center; flex-shrink: 0;">
        <span class="sortable" style="cursor: pointer; user-select: none; display: inline-flex; align-items: center; gap: 6px;" onclick="toggleTableStatusSort()">
          Processing Status
          <span id="table-status-sort-indicator" style="font-size: 12px; color: var(--gray-6);">⇅</span>
        </span>
      </div>
      <div style="width: 100px; text-align: center; flex-shrink: 0;">Actions</div>
    `;
    // Update checkbox state
    updateDBTablesSelectAllCheckboxState();
  } else {
    // Normal mode, no checkbox
    headerElement.innerHTML = `
      <div style="flex: 1; min-width: 0;">
        <span class="sortable" style="cursor: pointer; user-select: none; display: inline-flex; align-items: center; gap: 6px;" onclick="toggleTableNameSort()">
          Table Name
          <span id="table-name-sort-indicator" style="font-size: 12px; color: var(--gray-6);">⇅</span>
        </span>
      </div>
      <div style="width: 100px; text-align: center; flex-shrink: 0;">
        <span class="sortable" style="cursor: pointer; user-select: none; display: inline-flex; align-items: center; gap: 6px;" onclick="toggleDBTypeSort()">
          Database Type
          <span id="db-type-sort-indicator" style="font-size: 12px; color: var(--gray-6);">⇅</span>
        </span>
      </div>
      <div style="width: 200px; text-align: center; flex-shrink: 0;">
        <span class="sortable" style="cursor: pointer; user-select: none; display: inline-flex; align-items: center; gap: 6px;" onclick="toggleDatabaseNameSort()">
          Database Name
          <span id="database-name-sort-indicator" style="font-size: 12px; color: var(--gray-6);">⇅</span>
        </span>
      </div>
      <div style="width: 120px; text-align: center; flex-shrink: 0;">
        <span class="sortable" style="cursor: pointer; user-select: none; display: inline-flex; align-items: center; gap: 6px;" onclick="toggleTableStatusSort()">
          Processing Status
          <span id="table-status-sort-indicator" style="font-size: 12px; color: var(--gray-6);">⇅</span>
        </span>
      </div>
      <div style="width: 100px; text-align: center; flex-shrink: 0;">Actions</div>
    `;
  }

  // Update sort indicators
  updateTableNameSortIndicator();
  updateDBTypeSortIndicator();
  updateDatabaseNameSortIndicator();
  updateTableStatusSortIndicator();
}

// Q&A File Selection Module Functions

// Show Q&A file selection modal
async function showQAFileSelectionModal() {
  try {
    showModal('qa-file-selection-modal');

    const response = await API.request('/api/files/list');
    const excelFiles = response.filter(f =>
      f.name.endsWith('.xls') || f.name.endsWith('.xlsx')
    );

    renderQAFileModalList(excelFiles);
  } catch (error) {
    showToast(t('toast_load_files_failed_kb', { error: error.message }), 'error');
  }
}

// Render Q&A file modal list
function renderQAFileModalList(files) {
  const container = document.getElementById('qa-file-modal-list');

  if (files.length === 0) {
    container.innerHTML = '<div class="empty-state"><p>No Excel files found</p></div>';
    return;
  }

  container.innerHTML = files.map(file => {
    const isSelected = currentKBDetail.configuration.selectedQAFiles.includes(file.name);
    return `
      <div style="padding: 12px; border-bottom: 1px solid var(--gray-4); display: flex; align-items: center; gap: 12px;">
        <input type="checkbox" class="qa-file-checkbox" value="${escapeHtml(file.name)}" ${isSelected ? 'checked' : ''}>
        <div style="flex: 1;">
          <div style="font-weight: 500;">📊 ${escapeHtml(file.name)}</div>
          <div style="font-size: 12px; color: var(--gray-7);">
            ${formatFileSize(file.size)}
          </div>
        </div>
      </div>
    `;
  }).join('');
}

// Confirm Q&A file selection
async function confirmQAFileSelection() {
  const checkboxes = document.querySelectorAll('.qa-file-checkbox:checked');
  const selectedFiles = Array.from(checkboxes).map(cb => cb.value);

  // Validate QA files format if any files are selected
  if (selectedFiles.length > 0) {
    try {
      // Show loading indicator
      showToast('Validating Excel file format...', 'info');

      // Validate all selected files in parallel
      const validationPromises = selectedFiles.map(filename =>
        validateQAFileFormat(filename)
      );

      const results = await Promise.all(validationPromises);
      const invalidFiles = selectedFiles.filter((_, i) => !results[i]);
      const validFiles = selectedFiles.filter((_, i) => results[i]);

      // Show warning for invalid files
      if (invalidFiles.length > 0) {
        const invalidFilesList = invalidFiles.join('\n• ');
        await confirmDialog(
          `The following files do not meet the Excel format requirements:\n\n• ${invalidFilesList}\n\n` +
          `Format requirements:\n` +
          `• Sheet name: "example"\n` +
          `• Column headers: question, answer, howtofind\n\n` +
          `Invalid files have been automatically filtered.\n` +
          `Valid file count: ${validFiles.length}`
        );
      }

      // Only keep valid files
      currentKBDetail.configuration.selectedQAFiles = validFiles;
      hideModal('qa-file-selection-modal');
      renderSelectedQAFiles();

      if (validFiles.length > 0) {
        showToast(`Selected ${validFiles.length} Q&A file(s) with valid format`, 'success');
      } else {
        showToast('No files with valid format selected', 'warning');
      }
    } catch (error) {
      console.error('Validation error:', error);
      showToast('Failed to validate file format: ' + error.message, 'error');
    }
  } else {
    currentKBDetail.configuration.selectedQAFiles = [];
    hideModal('qa-file-selection-modal');
    renderSelectedQAFiles();
    showToast('Selection cancelled', 'info');
  }
}

// Validate Q&A file format
async function validateQAFileFormat(filename) {
  try {
    const response = await API.request(`/api/knowledge/files/validate-qa/${encodeURIComponent(filename)}`);
    return response.valid || false;
  } catch (error) {
    console.error(`Validation error for ${filename}:`, error);
    return false;
  }
}

// Render selected Q&A files
async function renderSelectedQAFiles() {
  const container = document.getElementById('selected-qa-files-container');
  const headerElement = document.getElementById('selected-qa-files-header');
  const { currentPage, itemsPerPage } = selectedQAFilesPagination;

  // Get sorted and filtered files
  const sortedFiles = await getSortedAndFilteredQAFiles();

  if (sortedFiles.length === 0) {
    container.innerHTML = `<div class="empty-state"><p>${t('no_qa_files_yet')}</p></div>`;
    document.getElementById('selected-qa-files-pagination').style.display = 'none';
    if (headerElement) headerElement.style.display = 'none';
    return;
  }

  // Show header when there are files
  if (headerElement) {
    headerElement.style.display = 'flex';
    // Update header content based on select mode
    updateSelectedQAFilesHeader();
  }

  // Get file status
  let fileStatusMap = {};
  if (currentKBDetail.id) {
    try {
      fileStatusMap = await API.getKnowledgeBaseFileStatus(currentKBDetail.id);
    } catch (error) {
      console.warn('Failed to load Q&A file status:', error);
    }
  }

  const startIndex = (currentPage - 1) * itemsPerPage;
  const endIndex = startIndex + itemsPerPage;
  const pageFiles = sortedFiles.slice(startIndex, endIndex);

  container.innerHTML = pageFiles.map((fileName) => {
    const status = fileStatusMap[fileName];
    const statusBadge = status ? renderStatusBadge(status.status) : '<span class="badge" style="background-color: var(--gray-3); color: var(--gray-7);">− Not Added</span>';
    const isChecked = selectedQAFilesForBatchDelete.has(fileName);
    const isCompleted = status && status.status === 'completed';
    const fileNameClass = isCompleted ? 'style="overflow: hidden; text-overflow: ellipsis; white-space: nowrap; cursor: pointer; color: var(--primary); text-decoration: underline;" onclick="navigateToQADetail(\'' + escapeHtml(fileName) + '\')"' : 'style="overflow: hidden; text-overflow: ellipsis; white-space: nowrap;"';

    return `
      <div class="selected-item">
        <div style="display: flex; align-items: center; gap: 8px; flex: 1; min-width: 0;">
          ${isSelectedQAFilesSelectMode ? `
            <input type="checkbox" class="selected-qa-file-checkbox" value="${escapeHtml(fileName)}" ${isChecked ? 'checked' : ''} onchange="toggleSelectedQAFileForBatchDelete('${escapeHtml(fileName)}', this.checked)">
          ` : ''}
          <img src="/assets/images/analytics.svg" alt="Analytics" style="width: 16px; height: 16px;">
          <span ${fileNameClass}>${escapeHtml(fileName)}</span>
        </div>
        <div style="width: 120px; display: flex; justify-content: center; align-items: center;">
          ${statusBadge}
        </div>
        <div style="width: 100px; display: flex; justify-content: center; align-items: center;">
          <button class="btn btn-small btn-secondary" onclick="removeSelectedQAFileByName('${escapeHtml(fileName)}')">
            <img src="assets/images/delete.svg" alt="Delete" style="width: 16px; height: 16px; filter: grayscale(100%) brightness(0.6);">
          </button>
        </div>
      </div>
    `;
  }).join('');

  // Always show pagination when there are files
  renderSelectedQAFilesPagination(sortedFiles);
  document.getElementById('selected-qa-files-pagination').style.display = 'flex';

  // Update select all checkbox state after rendering file list
  if (isSelectedQAFilesSelectMode) {
    updateQAFilesSelectAllCheckboxState();
  }
}

// Remove selected Q&A file by name
function removeSelectedQAFileByName(fileName) {
  currentKBDetail.configuration.selectedQAFiles = currentKBDetail.configuration.selectedQAFiles.filter(f => f !== fileName);
  renderSelectedQAFiles();
  showToast(t('toast_qa_file_removed'), 'info');
}

// Handle selected Q&A files search
function handleSelectedQAFilesSearch(query) {
  selectedQAFilesSearchQuery = query;
  selectedQAFilesPagination.currentPage = 1;
  renderSelectedQAFiles();
}

// Toggle selected Q&A files select mode
function toggleSelectedQAFilesSelectMode() {
  isSelectedQAFilesSelectMode = !isSelectedQAFilesSelectMode;
  
  toggleSelectModeUI({
    isSelectMode: isSelectedQAFilesSelectMode,
    selectBtnId: 'selected-qa-files-select-btn',
    batchDeleteBtnId: 'selected-qa-files-batch-delete-btn',
    batchDeleteTextId: 'selected-qa-files-batch-delete-text'
  });

  if (!isSelectedQAFilesSelectMode) {
    selectedQAFilesForBatchDelete.clear();
  }

  renderSelectedQAFiles();
}

// Toggle selected Q&A file for batch delete
function toggleSelectedQAFileForBatchDelete(fileName, checked) {
  if (checked) {
    selectedQAFilesForBatchDelete.add(fileName);
  } else {
    selectedQAFilesForBatchDelete.delete(fileName);
  }

  updateBatchDeleteButton('selected-qa-files-batch-delete-btn', 'selected-qa-files-batch-delete-text', selectedQAFilesForBatchDelete.size);
  updateQAFilesSelectAllCheckboxState();
}

// Batch delete selected Q&A files
async function batchDeleteSelectedQAFiles() {
  const count = selectedQAFilesForBatchDelete.size;
  if (count === 0) {
    showToast(t('toast_select_files_first'), 'warning');
    return;
  }

  const confirmed = await confirmDialog(`Are you sure you want to delete the selected ${count} Q&A file(s)?`);
  if (!confirmed) return;

  // Remove selected files from configuration
  const filesToDelete = Array.from(selectedQAFilesForBatchDelete);
  currentKBDetail.configuration.selectedQAFiles = currentKBDetail.configuration.selectedQAFiles.filter(
    f => !filesToDelete.includes(f)
  );

  // Clear selection and exit select mode
  selectedQAFilesForBatchDelete.clear();
  isSelectedQAFilesSelectMode = false;
  const selectBtn = document.getElementById('selected-qa-files-select-btn');
  const batchDeleteBtn = document.getElementById('selected-qa-files-batch-delete-btn');
  selectBtn.innerHTML = `
    <img src="assets/images/select.svg" alt="Select" style="width: 16px; height: 16px; margin-right: 6px; vertical-align: middle;">
    Batch Select
  `;
  batchDeleteBtn.style.display = 'none';

  renderSelectedQAFiles();
  showToast(`Deleted ${count} Q&A file(s)`, 'success');
}

// Update selected Q&A files header based on select mode
function updateSelectedQAFilesHeader() {
  const headerElement = document.getElementById('selected-qa-files-header');
  if (!headerElement) return;

  if (isSelectedQAFilesSelectMode) {
    // In select mode, show checkbox in header
    headerElement.innerHTML = `
      <div style="display: flex; align-items: center; gap: 8px; flex: 1; min-width: 0;">
        <input type="checkbox" id="select-all-qa-files-checkbox" onchange="toggleSelectAllQAFilesCurrentPage(this.checked)">
        <span class="sortable" style="cursor: pointer; user-select: none; display: flex; align-items: center; gap: 6px;" onclick="toggleQAFileNameSort()">
          ${t('file_name')}
          <span id="qa-filename-sort-indicator" style="font-size: 12px; color: var(--gray-6);">⇅</span>
        </span>
      </div>
      <div style="width: 120px; text-align: center;">
        <span class="sortable" style="cursor: pointer; user-select: none; display: inline-flex; align-items: center; gap: 6px;" onclick="toggleQAFileStatusSort()">
          ${t('processing_status')}
          <span id="qa-status-sort-indicator" style="font-size: 12px; color: var(--gray-6);">⇅</span>
        </span>
      </div>
      <div style="width: 100px; text-align: center;">${t('actions')}</div>
    `;
    // Update checkbox state
    updateQAFilesSelectAllCheckboxState();
  } else {
    // Normal mode, no checkbox
    headerElement.innerHTML = `
      <div style="flex: 1; min-width: 0;">
        <span class="sortable" style="cursor: pointer; user-select: none; display: inline-flex; align-items: center; gap: 6px;" onclick="toggleQAFileNameSort()">
          ${t('file_name')}
          <span id="qa-filename-sort-indicator" style="font-size: 12px; color: var(--gray-6);">⇅</span>
        </span>
      </div>
      <div style="width: 120px; text-align: center;">
        <span class="sortable" style="cursor: pointer; user-select: none; display: inline-flex; align-items: center; gap: 6px;" onclick="toggleQAFileStatusSort()">
          ${t('processing_status')}
          <span id="qa-status-sort-indicator" style="font-size: 12px; color: var(--gray-6);">⇅</span>
        </span>
      </div>
      <div style="width: 100px; text-align: center;">${t('actions')}</div>
    `;
  }

  // Update sort indicators
  updateQAFileNameSortIndicator();
  updateQAFileStatusSortIndicator();
}

// Update the "select all" checkbox state for Q&A files
function updateQAFilesSelectAllCheckboxState() {
  const checkbox = document.getElementById('select-all-qa-files-checkbox');
  if (!checkbox) return;

  // Get all file checkboxes on the current page
  const fileCheckboxes = document.querySelectorAll('.selected-qa-file-checkbox');

  if (fileCheckboxes.length === 0) {
    checkbox.checked = false;
    return;
  }

  // Check if all checkboxes are checked
  const allChecked = Array.from(fileCheckboxes).every(cb => cb.checked);
  checkbox.checked = allChecked;
}

// Toggle select all Q&A files on current page
async function toggleSelectAllQAFilesCurrentPage(checked) {
  const { currentPage, itemsPerPage } = selectedQAFilesPagination;

  // Get sorted and filtered files
  const sortedFiles = await getSortedAndFilteredQAFiles();

  const startIndex = (currentPage - 1) * itemsPerPage;
  const endIndex = startIndex + itemsPerPage;
  const pageFiles = sortedFiles.slice(startIndex, endIndex);

  if (checked) {
    pageFiles.forEach(fileName => selectedQAFilesForBatchDelete.add(fileName));
  } else {
    pageFiles.forEach(fileName => selectedQAFilesForBatchDelete.delete(fileName));
  }

  updateBatchDeleteButton('selected-qa-files-batch-delete-btn', 'selected-qa-files-batch-delete-text', selectedQAFilesForBatchDelete.size);
  renderSelectedQAFiles();
}

// Render selected Q&A files pagination
function renderSelectedQAFilesPagination(filteredFiles) {
  const container = document.getElementById('selected-qa-files-pagination');
  const { currentPage, itemsPerPage } = selectedQAFilesPagination;
  const totalPages = Math.ceil(filteredFiles.length / itemsPerPage);
  const startIndex = (currentPage - 1) * itemsPerPage;
  const endIndex = Math.min(startIndex + itemsPerPage, filteredFiles.length);

  container.innerHTML = renderPaginationHTML({
    currentPage,
    totalPages,
    itemsPerPage,
    pageOptions: [5, 10, 20, 50],
    startIndex,
    endIndex,
    totalItems: filteredFiles.length,
    callbacks: {
      onPageSizeChange: 'changeSelectedQAFilesPerPage',
      onFirstPage: 'goToFirstQAFilesPage',
      onPrevPage: 'changeQAFilesPage.bind(null, -1)',
      onPageNumber: 'goToQAFilesPageNumber',
      onNextPage: 'changeQAFilesPage.bind(null, 1)',
      onLastPage: 'goToLastQAFilesPage'
    }
  });
}

// Change selected Q&A files per page
function changeSelectedQAFilesPerPage(size) {
  selectedQAFilesPagination.itemsPerPage = parseInt(size);
  selectedQAFilesPagination.currentPage = 1;
  renderSelectedQAFiles();
}

// Go to first Q&A files page
function goToFirstQAFilesPage() {
  selectedQAFilesPagination.currentPage = 1;
  renderSelectedQAFiles();
}

// Go to last Q&A files page
async function goToLastQAFilesPage() {
  const sortedFiles = await getSortedAndFilteredQAFiles();
  const totalPages = Math.ceil(sortedFiles.length / selectedQAFilesPagination.itemsPerPage);
  selectedQAFilesPagination.currentPage = totalPages;
  renderSelectedQAFiles();
}

// Go to Q&A files page number
async function goToQAFilesPageNumber(pageNum) {
  const sortedFiles = await getSortedAndFilteredQAFiles();
  const totalPages = Math.ceil(sortedFiles.length / selectedQAFilesPagination.itemsPerPage);
  const page = parseInt(pageNum);

  if (page >= 1 && page <= totalPages) {
    selectedQAFilesPagination.currentPage = page;
    renderSelectedQAFiles();
  }
}

// Change Q&A files page
async function changeQAFilesPage(delta) {
  const { currentPage } = selectedQAFilesPagination;
  const sortedFiles = await getSortedAndFilteredQAFiles();
  const totalPages = Math.ceil(sortedFiles.length / selectedQAFilesPagination.itemsPerPage);

  const newPage = currentPage + delta;
  if (newPage >= 1 && newPage <= totalPages) {
    selectedQAFilesPagination.currentPage = newPage;
    renderSelectedQAFiles();
  }
}

// Get sorted and filtered Q&A files list
async function getSortedAndFilteredQAFiles() {
  // Safety check: ensure currentKBDetail and configuration exist
  if (!currentKBDetail || !currentKBDetail.configuration || !currentKBDetail.configuration.selectedQAFiles) {
    return [];
  }

  const { selectedQAFiles } = currentKBDetail.configuration;

  // Filter files based on search query
  const filteredFiles = selectedQAFiles.filter(fileName =>
    fileName.toLowerCase().includes(selectedQAFilesSearchQuery.toLowerCase())
  );

  // Get file status if sorting by status
  let fileStatusMap = {};
  if (qaFileStatusSortOrder && currentKBDetail.id) {
    try {
      fileStatusMap = await API.getKnowledgeBaseFileStatus(currentKBDetail.id);
    } catch (error) {
      console.warn('Failed to load Q&A file status:', error);
    }
  }

  // Apply sorting if needed
  let sortedFiles = [...filteredFiles];
  if (qaFileNameSortOrder) {
    sortedFiles = sortQAFilesByName(sortedFiles, qaFileNameSortOrder);
  } else if (qaFileStatusSortOrder) {
    sortedFiles = sortQAFilesByStatus(sortedFiles, fileStatusMap, qaFileStatusSortOrder);
  }

  return sortedFiles;
}

// Toggle Q&A file name sort order
function toggleQAFileNameSort() {
  // Clear other sort
  qaFileStatusSortOrder = null;

  // Cycle through: null -> asc -> desc -> null
  if (qaFileNameSortOrder === null) {
    qaFileNameSortOrder = 'asc';
  } else if (qaFileNameSortOrder === 'asc') {
    qaFileNameSortOrder = 'desc';
  } else {
    qaFileNameSortOrder = null;
  }

  // Re-render with new sort
  renderSelectedQAFiles();
}

// Toggle Q&A file status sort order
function toggleQAFileStatusSort() {
  // Clear other sort
  qaFileNameSortOrder = null;

  // Cycle through: null -> asc -> desc -> null
  if (qaFileStatusSortOrder === null) {
    qaFileStatusSortOrder = 'asc';
  } else if (qaFileStatusSortOrder === 'asc') {
    qaFileStatusSortOrder = 'desc';
  } else {
    qaFileStatusSortOrder = null;
  }

  // Re-render with new sort
  renderSelectedQAFiles();
}

// Update Q&A file name sort indicator
function updateQAFileNameSortIndicator() {
  updateSortIndicator('qa-filename-sort-indicator', qaFileNameSortOrder);
}

// Update Q&A file status sort indicator
function updateQAFileStatusSortIndicator() {
  updateSortIndicator('qa-status-sort-indicator', qaFileStatusSortOrder);
}

// Sort Q&A files by name
function sortQAFilesByName(files, order) {
  return sortByName(files, order);
}

// Sort Q&A files by status
function sortQAFilesByStatus(files, statusMap, order) {
  return sortByStatus(files, statusMap, order);
}

// Configuration Module Functions

// Show configuration modal
async function showConfigModal() {
  const modal = document.getElementById('config-modal');
  const kbContainer = document.getElementById('config-display-kb');
  const yamlContainer = document.getElementById('config-display-yaml');

  // Debug: log the current KB detail
  console.log('[Config Modal] currentKBDetail:', currentKBDetail);
  console.log('[Config Modal] configuration:', currentKBDetail.configuration);

  // Render KB configuration
  const configuration = currentKBDetail.configuration || {};
  if (!configuration || Object.keys(configuration).length === 0) {
    kbContainer.textContent = 'No configuration';
  } else {
    kbContainer.textContent = JSON.stringify(configuration, null, 2);
  }

  // Load and render YAML configuration
  yamlContainer.textContent = 'Loading...';
  try {
    // Fetch YAML configuration from backend
    // Backend will use KB-specific YAML if exists, otherwise default.yaml
    // Use KB ID to match config files named by ID (e.g., 11.yaml)
    const response = await API.request(`/api/config/${currentKBDetail.id}`);

    if (response) {
      // Format the YAML config as pretty JSON
      // Remove metadata if present
      const { _metadata, ...configData } = response;

      // Display which config file is being used
      let configInfo = '';
      if (_metadata) {
        console.log('[Config Modal] YAML metadata:', _metadata);
        const configPath = _metadata.config_path || 'configs/rag/default.yaml';
        const isCustom = _metadata.is_custom || false;
        configInfo = `📍 Current configuration: ${configPath}${isCustom ? ' (KB-specific config)' : ' (default config)'}\n\n`;
      }

      yamlContainer.textContent = configInfo + JSON.stringify(configData, null, 2);
    } else {
      yamlContainer.textContent = 'Unable to load configuration file';
    }
  } catch (error) {
    console.error('[Config Modal] Failed to load YAML:', error);
    yamlContainer.textContent = 'Loading failed\n\nError: ' + error.message;
  }

  // Show modal with animation
  modal.style.display = 'flex';
  setTimeout(() => {
    modal.classList.add('show');
  }, 10);
}

// Hide configuration modal
function hideConfigModal() {
  const modal = document.getElementById('config-modal');
  modal.classList.remove('show');
  setTimeout(() => {
    modal.style.display = 'none';
  }, 300);
}

// Switch configuration tab
function switchConfigTab(tab, event) {
  // Update tab buttons - only within the config modal
  document.querySelectorAll('#config-modal .config-tab').forEach(btn => {
    btn.classList.remove('active');
  });
  if (event && event.target) {
    event.target.classList.add('active');
  }

  // Update tab content - only within the config modal
  document.querySelectorAll('#config-modal .config-tab-content').forEach(content => {
    content.classList.remove('active');
  });

  if (tab === 'kb') {
    document.getElementById('config-tab-kb').classList.add('active');
  } else if (tab === 'yaml') {
    document.getElementById('config-tab-yaml').classList.add('active');
  }
}

// Save knowledge base configuration
async function saveKnowledgeBaseConfiguration() {
  if (!currentKBDetail.id) {
    showToast('Invalid knowledge base ID', 'error');
    return;
  }

  try {
    // Prepare configuration data
    const configuration = {
      tools: currentKBDetail.configuration.tools,
      selectedFiles: currentKBDetail.configuration.selectedFiles,
      selectedQAFiles: currentKBDetail.configuration.selectedQAFiles,
      dbConnections: currentKBDetail.configuration.dbConnections.map(conn => ({
        type: conn.type,
        host: conn.host,
        port: conn.port,
        database: conn.database,
        username: conn.username,
        password: conn.password,
        file_path: conn.file_path,
        tables: conn.tables
      }))
    };

    showToast('Saving configuration...', 'info');

    // Call API to save
    await API.updateKBConfiguration(currentKBDetail.id, configuration);

    showToast('Configuration saved successfully', 'success');

  } catch (error) {
    showToast('Failed to save configuration: ' + error.message, 'error');
  }
}

// File Selection Modal Batch Operations

// Select all files (across all pages)
function selectAllFilesInModal() {
  const { filteredFiles } = fileModalPagination;

  // Add all filtered files to selection
  const allFileNames = filteredFiles.map(f => f.name);
  const existingSelection = new Set(currentKBDetail.configuration.selectedFiles);
  allFileNames.forEach(name => existingSelection.add(name));
  currentKBDetail.configuration.selectedFiles = Array.from(existingSelection);

  // Check all checkboxes on current page
  document.querySelectorAll('#file-modal-list .file-checkbox').forEach(cb => {
    cb.checked = true;
  });

  updateFileSelectionCount();
  showToast(`Selected all ${filteredFiles.length} file(s)`, 'success');
}

// Select only files on current page
function selectCurrentPageFilesInModal() {
  const checkboxes = document.querySelectorAll('#file-modal-list .file-checkbox');
  const existingSelection = new Set(currentKBDetail.configuration.selectedFiles);
  let count = 0;

  checkboxes.forEach(cb => {
    if (!cb.checked) {
      cb.checked = true;
      existingSelection.add(cb.value);
      count++;
    }
  });

  currentKBDetail.configuration.selectedFiles = Array.from(existingSelection);
  updateFileSelectionCount();

  if (count > 0) {
    showToast(`Selected ${checkboxes.length} file(s) on current page`, 'success');
  }
}

// Deselect all files
function deselectAllFilesInModal() {
  // Clear all selections
  currentKBDetail.configuration.selectedFiles = [];

  // Uncheck all checkboxes on current page
  document.querySelectorAll('#file-modal-list .file-checkbox').forEach(cb => {
    cb.checked = false;
  });

  updateFileSelectionCount();
  showToast('All selections cancelled', 'info');
}

// Update file selection count display
function updateFileSelectionCount() {
  const countElement = document.getElementById('file-selection-count');
  if (!countElement) return;

  const { filteredFiles } = fileModalPagination;
  const totalCount = filteredFiles.length;

  // Count how many of the filtered files are in the selection
  const selectedCount = currentKBDetail.configuration.selectedFiles.filter(fileName =>
    filteredFiles.some(f => f.name === fileName)
  ).length;

  if (selectedCount > 0) {
    countElement.textContent = `Selected ${selectedCount} / ${totalCount}`;
    countElement.style.color = 'var(--success)';
  } else {
    countElement.textContent = `0 / ${totalCount}`;
    countElement.style.color = 'var(--gray-7)';
  }
}

// File Sorting Functions

// Get sorted and filtered files list
async function getSortedAndFilteredFiles() {
  // Safety check: ensure currentKBDetail and configuration exist
  if (!currentKBDetail || !currentKBDetail.configuration || !currentKBDetail.configuration.selectedFiles) {
    return [];
  }

  const { selectedFiles } = currentKBDetail.configuration;

  // Filter files based on search query
  const filteredFiles = selectedFiles.filter(fileName =>
    fileName.toLowerCase().includes(selectedFilesSearchQuery.toLowerCase())
  );

  // Get file status if sorting by status
  let fileStatusMap = {};
  if (kbDetailFileStatusSortOrder && currentKBDetail.id) {
    try {
      fileStatusMap = await API.getKnowledgeBaseFileStatus(currentKBDetail.id);
    } catch (error) {
      console.warn('Failed to load file status:', error);
    }
  }

  // Apply sorting if needed
  let sortedFiles = [...filteredFiles];
  if (kbDetailFileNameSortOrder) {
    sortedFiles = sortFilesByName(sortedFiles, kbDetailFileNameSortOrder);
  } else if (kbDetailFileStatusSortOrder) {
    sortedFiles = sortFilesByStatus(sortedFiles, fileStatusMap, kbDetailFileStatusSortOrder);
  }

  return sortedFiles;
}

// Toggle file name sort order
function toggleFileNameSort() {
  // Clear other sort
  kbDetailFileStatusSortOrder = null;

  // Cycle through: null -> asc -> desc -> null
  if (kbDetailFileNameSortOrder === null) {
    kbDetailFileNameSortOrder = 'asc'; // First click: A-Z
  } else if (kbDetailFileNameSortOrder === 'asc') {
    kbDetailFileNameSortOrder = 'desc'; // Second click: Z-A
  } else {
    kbDetailFileNameSortOrder = null; // Third click: clear sort
  }

  // Re-render with new sort
  renderSelectedFiles();
}

// Toggle file status sort order
function toggleFileStatusSort() {
  // Clear other sort
  kbDetailFileNameSortOrder = null;

  // Cycle through: null -> asc -> desc -> null
  if (kbDetailFileStatusSortOrder === null) {
    kbDetailFileStatusSortOrder = 'asc'; // First click: pending -> processing -> completed -> failed
  } else if (kbDetailFileStatusSortOrder === 'asc') {
    kbDetailFileStatusSortOrder = 'desc'; // Second click: reverse
  } else {
    kbDetailFileStatusSortOrder = null; // Third click: clear sort
  }

  // Re-render with new sort
  renderSelectedFiles();
}

// Update file name sort indicator
function updateFileNameSortIndicator() {
  updateSortIndicator('filename-sort-indicator', kbDetailFileNameSortOrder);
}

// Update file status sort indicator
function updateFileStatusSortIndicator() {
  updateSortIndicator('status-sort-indicator', kbDetailFileStatusSortOrder);
}

// Sort files by name
function sortFilesByName(files, order) {
  return sortByName(files, order);
}

// Sort files by status
function sortFilesByStatus(files, statusMap, order) {
  return sortByStatus(files, statusMap, order);
}

// Navigate to Q&A detail page
function navigateToQADetail(fileName) {
  if (!currentKBDetail.id) {
    console.error('[KB Detail] Cannot navigate: KB ID not found');
    return;
  }

  const encodedKBId = encodeURIComponent(currentKBDetail.id);
  const encodedFileName = encodeURIComponent(fileName);
  // Open QA detail page in new tab
  const url = `pages/qa-detail.html?kb_id=${encodedKBId}&file=${encodedFileName}`;
  window.open(url, '_blank');
}
