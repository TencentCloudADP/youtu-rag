// Q&A Detail Page Component

// State
let currentQADetail = {
  kbId: null,
  fileName: null,
  qaList: []
};

let qaSearchQuery = '';
let isQASelectMode = false;
let selectedQAForBatchExecute = new Set();
let currentQAInModal = null;

// Pagination state
let qaPagination = {
  currentPage: 1,
  itemsPerPage: 20
};

// Initialize Q&A detail page
function initQADetail() {
  console.log('[QA Detail] Initializing Q&A detail page');

  // Get KB ID and file name from URL parameters
  const urlParams = new URLSearchParams(window.location.search);
  const kbId = urlParams.get('kb_id');
  const fileName = urlParams.get('file');

  if (!kbId || !fileName) {
    console.error('[QA Detail] Invalid URL parameters');
    showToast(t('toast_invalid_url'), 'error');
    window.close();
    return;
  }

  currentQADetail.kbId = decodeURIComponent(kbId);
  currentQADetail.fileName = decodeURIComponent(fileName);

  console.log('[QA Detail] KB ID:', currentQADetail.kbId);
  console.log('[QA Detail] File Name:', currentQADetail.fileName);

  // Set page title
  document.getElementById('qa-detail-filename').textContent = currentQADetail.fileName;

  // Initialize Memory switch
  initMemorySwitch();

  // Load Q&A data
  loadQAData();

  // Setup i18n listener
  if (typeof i18n !== 'undefined') {
    i18n.onChange(() => {
      // Re-render Q&A list when language changes
      if (currentQADetail.qaList.length > 0) {
        renderQAList();
      }
    });
  }
}

// Load Q&A data from API
async function loadQAData() {
  try {
    console.log('[QA Detail] Loading Q&A data...');

    // Call API to get QA associations
    const data = await API.getQAAssociations(currentQADetail.kbId, currentQADetail.fileName);

    currentQADetail.qaList = data.qa_list || [];

    // Update info text
    document.getElementById('qa-detail-info').textContent =
      `${t('total_qa_count', { count: currentQADetail.qaList.length })} | ${t('knowledge_base')}: ${currentQADetail.kbId}`;

    // Render Q&A list
    renderQAList();

  } catch (error) {
    console.error('[QA Detail] Failed to load Q&A data:', error);
    showToast(t('toast_load_qa_failed', { error: error.message }), 'error');
  }
}

// Get filtered Q&A list
function getFilteredQAList() {
  if (!qaSearchQuery) {
    return currentQADetail.qaList;
  }

  const query = qaSearchQuery.toLowerCase();
  return currentQADetail.qaList.filter(qa =>
    qa.question.toLowerCase().includes(query) ||
    qa.answer.toLowerCase().includes(query)
  );
}

// Render Q&A list
function renderQAList() {
  const container = document.getElementById('qa-list-container');
  const headerElement = document.getElementById('qa-list-header');
  const { currentPage, itemsPerPage } = qaPagination;

  const filteredList = getFilteredQAList();

  if (filteredList.length === 0) {
    container.innerHTML = `<div class="empty-state"><p>${t('no_qa_found')}</p></div>`;
    document.getElementById('qa-pagination').style.display = 'none';
    if (headerElement) headerElement.style.display = 'none';
    return;
  }

  // Show header
  if (headerElement) {
    headerElement.style.display = 'flex';
    updateQAListHeader();
  }

  // Pagination
  const startIndex = (currentPage - 1) * itemsPerPage;
  const endIndex = startIndex + itemsPerPage;
  const pageItems = filteredList.slice(startIndex, endIndex);

  container.innerHTML = pageItems.map((qa) => {
    const isChecked = selectedQAForBatchExecute.has(qa.id);
    const statusBadge = renderLearningStatusBadge(qa.learning_status);

    // Safely handle answer field (may be null)
    const answerText = qa.answer || '';
    const answerPreview = answerText.length > 100 ? answerText.substring(0, 100) + '...' : answerText;

    return `
      <div class="selected-item">
        <div style="width: 40px; display: flex; justify-content: center; align-items: center;">
          ${isQASelectMode ? `
            <input type="checkbox" class="qa-checkbox" value="${qa.id}" ${isChecked ? 'checked' : ''} onchange="toggleQAForBatchExecute(${qa.id}, this.checked)">
          ` : `
            <span style="color: var(--gray-6); font-size: 14px;">${qa.id}</span>
          `}
        </div>
        <div style="flex: 1; min-width: 0; cursor: pointer;" onclick="showQADetailModal(${qa.id})">
          <div style="overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-weight: 500;">
            ${escapeHtml(qa.question || '')}
          </div>
          <div style="font-size: 12px; color: var(--gray-6); margin-top: 4px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
            ${escapeHtml(answerPreview)}
          </div>
        </div>
        <div style="width: 150px; display: flex; justify-content: center; align-items: center;">
          ${statusBadge}
        </div>
        <div style="width: 150px; display: flex; justify-content: center; align-items: center; font-size: 13px; color: var(--gray-7);">
          ${formatDateTime(qa.updated_at)}
        </div>
        <div style="width: 100px; display: flex; justify-content: center; align-items: center;">
          <button class="btn btn-small btn-primary" onclick="executeQA(${qa.id}); event.stopPropagation();">
            <img src="/assets/images/play.svg" alt="Execute" style="width: 14px; height: 14px;">
          </button>
        </div>
      </div>
    `;
  }).join('');

  // Render pagination
  renderQAPagination(filteredList);
  document.getElementById('qa-pagination').style.display = 'flex';

  // Update select all checkbox state
  if (isQASelectMode) {
    updateQASelectAllCheckboxState();
  }
}

// Update Q&A list header
function updateQAListHeader() {
  const headerElement = document.getElementById('qa-list-header');
  if (!headerElement) return;

  if (isQASelectMode) {
    headerElement.innerHTML = `
      <div style="width: 40px; display: flex; justify-content: center; align-items: center;">
        <input type="checkbox" id="select-all-qa-checkbox" onchange="toggleSelectAllQACurrentPage(this.checked)">
      </div>
      <div style="flex: 1; min-width: 0;" data-i18n="question">${t('question')}</div>
      <div style="width: 150px; text-align: center;" data-i18n="learning_status">${t('learning_status')}</div>
      <div style="width: 150px; text-align: center;" data-i18n="updated_at">${t('updated_at')}</div>
      <div style="width: 100px; text-align: center;" data-i18n="actions">${t('actions')}</div>
    `;
    updateQASelectAllCheckboxState();
  } else {
    headerElement.innerHTML = `
      <div style="width: 40px; text-align: center;">#</div>
      <div style="flex: 1; min-width: 0;" data-i18n="question">${t('question')}</div>
      <div style="width: 150px; text-align: center;" data-i18n="learning_status">${t('learning_status')}</div>
      <div style="width: 150px; text-align: center;" data-i18n="updated_at">${t('updated_at')}</div>
      <div style="width: 100px; text-align: center;" data-i18n="actions">${t('actions')}</div>
    `;
  }
}

// Render learning status badge
function renderLearningStatusBadge(status) {
  const statusConfig = {
    pending: { color: 'var(--gray-5)', bg: 'var(--gray-2)', text: t('status_pending') },
    learning: { color: 'var(--warning)', bg: '#fef3c7', text: t('status_learning') },
    completed: { color: 'var(--success)', bg: '#d1fae5', text: t('status_completed') },
    failed: { color: 'var(--error)', bg: '#fee2e2', text: t('status_failed') }
  };

  const config = statusConfig[status] || statusConfig.pending;
  return `<span class="badge" style="background-color: ${config.bg}; color: ${config.color};">${config.text}</span>`;
}

// Format date time
function formatDateTime(dateString) {
  if (!dateString) return '-';
  const date = new Date(dateString);
  return date.toLocaleString(i18n.getLang() === 'zh' ? 'zh-CN' : 'en-US', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit'
  });
}

// Handle Q&A search
function handleQASearch(query) {
  qaSearchQuery = query;
  qaPagination.currentPage = 1;
  renderQAList();
}

// Toggle Q&A select mode
function toggleQASelectMode() {
  isQASelectMode = !isQASelectMode;

  const selectBtn = document.getElementById('qa-select-btn');
  const batchExecuteBtn = document.getElementById('qa-batch-execute-btn');

  if (isQASelectMode) {
    selectBtn.innerHTML = `
      <img src="/assets/images/close.svg" alt="Cancel" style="width: 16px; height: 16px; margin-right: 6px; vertical-align: middle;">
      <span data-i18n="exit_select_mode">${t('exit_select_mode')}</span>
    `;
    batchExecuteBtn.style.display = 'inline-flex';
  } else {
    selectBtn.innerHTML = `
      <img src="/assets/images/select.svg" alt="Select" style="width: 16px; height: 16px; margin-right: 6px; vertical-align: middle;">
      <span data-i18n="batch_select">${t('batch_select')}</span>
    `;
    batchExecuteBtn.style.display = 'none';
    selectedQAForBatchExecute.clear();
  }

  renderQAList();
}

// Toggle Q&A for batch execute
function toggleQAForBatchExecute(qaId, checked) {
  if (checked) {
    selectedQAForBatchExecute.add(qaId);
  } else {
    selectedQAForBatchExecute.delete(qaId);
  }

  updateBatchExecuteButton();
  updateQASelectAllCheckboxState();
}

// Toggle select all Q&A on current page
function toggleSelectAllQACurrentPage(checked) {
  const filteredList = getFilteredQAList();
  const { currentPage, itemsPerPage } = qaPagination;
  const startIndex = (currentPage - 1) * itemsPerPage;
  const endIndex = startIndex + itemsPerPage;
  const pageItems = filteredList.slice(startIndex, endIndex);

  pageItems.forEach(qa => {
    if (checked) {
      selectedQAForBatchExecute.add(qa.id);
    } else {
      selectedQAForBatchExecute.delete(qa.id);
    }
  });

  renderQAList();
  updateBatchExecuteButton();
}

// Update Q&A select all checkbox state
function updateQASelectAllCheckboxState() {
  const checkbox = document.getElementById('select-all-qa-checkbox');
  if (!checkbox) return;

  const filteredList = getFilteredQAList();
  const { currentPage, itemsPerPage } = qaPagination;
  const startIndex = (currentPage - 1) * itemsPerPage;
  const endIndex = startIndex + itemsPerPage;
  const pageItems = filteredList.slice(startIndex, endIndex);

  const allChecked = pageItems.length > 0 && pageItems.every(qa => selectedQAForBatchExecute.has(qa.id));
  const someChecked = pageItems.some(qa => selectedQAForBatchExecute.has(qa.id));

  checkbox.checked = allChecked;
  checkbox.indeterminate = someChecked && !allChecked;
}

// Update batch execute button
function updateBatchExecuteButton() {
  const textElement = document.getElementById('qa-batch-execute-text');
  if (textElement) {
    textElement.textContent = t('batch_execute', { count: selectedQAForBatchExecute.size });
  }
}

// Show Q&A detail modal
function showQADetailModal(qaId) {
  const qa = currentQADetail.qaList.find(q => q.id === qaId);
  if (!qa) return;

  currentQAInModal = qa;

  document.getElementById('modal-question').textContent = qa.question || '-';
  document.getElementById('modal-answer').textContent = qa.answer || '-';
  document.getElementById('modal-howtofind').textContent = qa.howtofind || '-';
  document.getElementById('modal-source-file').textContent = qa.source_file || '-';
  document.getElementById('modal-created-at').textContent = formatDateTime(qa.created_at);
  document.getElementById('modal-learning-status').innerHTML = renderLearningStatusBadge(qa.learning_status);
  document.getElementById('modal-updated-at').textContent = formatDateTime(qa.updated_at);

  document.getElementById('qa-detail-modal').style.display = 'flex';
}

// Execute Q&A from modal
function executeQAFromModal() {
  if (currentQAInModal) {
    executeQA(currentQAInModal.id);
    hideModal('qa-detail-modal', (modalId) => {
      if (modalId === 'qa-detail-modal') {
        currentQAInModal = null;
      }
    });
  }
}

// Check if memory is enabled
async function checkMemoryEnabled() {
  try {
    const response = await fetch(API_BASE + '/api/memory/config');
    const data = await response.json();
    return data.enabled || false;
  } catch (error) {
    console.error('[QA Detail] Failed to check memory status:', error);
    return false;
  }
}

// Execute single Q&A
async function executeQA(qaId) {
  console.log('[QA Detail] Executing Q&A:', qaId);

  try {
    // Check memory status
    const memoryEnabled = await checkMemoryEnabled();
    if (!memoryEnabled) {
      showToast(t('toast_memory_not_enabled') || 'Memory feature is not enabled, please enable Memory in the chat page first', 'warning');
      return;
    }

    showToast(t('toast_qa_execution_started', { id: qaId }), 'info');

    // Call API to execute QA
    await API.executeQA(currentQADetail.kbId, qaId);

    showToast(t('toast_qa_execution_completed', { id: qaId }), 'success');

    // Reload QA data to get updated status
    await loadQAData();
  } catch (error) {
    console.error('[QA Detail] Failed to execute Q&A:', error);
    showToast(`Execution failed: ${error.message}`, 'error');
  }
}

// Batch execute Q&A
async function batchExecuteQA() {
  if (selectedQAForBatchExecute.size === 0) {
    showToast(t('toast_select_qa_first'), 'warning');
    return;
  }

  console.log('[QA Detail] Batch executing Q&A:', Array.from(selectedQAForBatchExecute));

  try {
    // Check memory status
    const memoryEnabled = await checkMemoryEnabled();
    if (!memoryEnabled) {
      showToast(t('toast_memory_not_enabled') || 'Memory feature is not enabled, please enable Memory in the chat page first', 'warning');
      return;
    }

    showToast(t('toast_batch_execution_started', { count: selectedQAForBatchExecute.size }), 'info');

    // Call API to batch execute QA
    const qaIds = Array.from(selectedQAForBatchExecute);
    await API.batchExecuteQA(currentQADetail.kbId, qaIds);

    showToast(t('toast_batch_execution_completed', { count: selectedQAForBatchExecute.size }), 'success');

    selectedQAForBatchExecute.clear();
    toggleQASelectMode();

    // Reload QA data to get updated status
    await loadQAData();
  } catch (error) {
    console.error('[QA Detail] Failed to batch execute Q&A:', error);
    showToast(`Batch execution failed: ${error.message}`, 'error');
  }
}

// Render Q&A pagination
function renderQAPagination(filteredList) {
  const container = document.getElementById('qa-pagination');
  const { currentPage, itemsPerPage } = qaPagination;
  const totalPages = Math.ceil(filteredList.length / itemsPerPage);
  const startIndex = (currentPage - 1) * itemsPerPage;
  const endIndex = Math.min(startIndex + itemsPerPage, filteredList.length);

  container.innerHTML = renderPaginationHTML({
    currentPage,
    totalPages,
    itemsPerPage,
    pageOptions: [10, 20, 50, 100],
    startIndex,
    endIndex,
    totalItems: filteredList.length,
    callbacks: {
      onPageSizeChange: 'changeQAPerPage',
      onFirstPage: 'goToFirstQAPage',
      onPrevPage: 'changeQAPage.bind(null, -1)',
      onPageNumber: 'goToQAPageNumber',
      onNextPage: 'changeQAPage.bind(null, 1)',
      onLastPage: 'goToLastQAPage'
    }
  });
}

// Change Q&A per page
function changeQAPerPage(size) {
  qaPagination.itemsPerPage = parseInt(size);
  qaPagination.currentPage = 1;
  renderQAList();
}

// Go to first Q&A page
function goToFirstQAPage() {
  qaPagination.currentPage = 1;
  renderQAList();
}

// Go to last Q&A page
function goToLastQAPage() {
  const filteredList = getFilteredQAList();
  const totalPages = Math.ceil(filteredList.length / qaPagination.itemsPerPage);
  qaPagination.currentPage = totalPages;
  renderQAList();
}

// Go to Q&A page number
function goToQAPageNumber(pageNum) {
  const filteredList = getFilteredQAList();
  const totalPages = Math.ceil(filteredList.length / qaPagination.itemsPerPage);
  const page = parseInt(pageNum);

  if (page >= 1 && page <= totalPages) {
    qaPagination.currentPage = page;
    renderQAList();
  }
}

// Change Q&A page
function changeQAPage(delta) {
  const { currentPage } = qaPagination;
  const filteredList = getFilteredQAList();
  const totalPages = Math.ceil(filteredList.length / qaPagination.itemsPerPage);

  const newPage = currentPage + delta;
  if (newPage >= 1 && newPage <= totalPages) {
    qaPagination.currentPage = newPage;
    renderQAList();
  }
}

// Go back to KB detail page
function goBackToKBDetail() {
  router.navigate(`/knowledge/${currentQADetail.kbId}`);
}

// Escape HTML
function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

// Initialize Memory switch
async function initMemorySwitch() {
  const memorySwitch = document.getElementById('memory-switch');
  if (memorySwitch) {
    // Get .env status from backend API during initialization
    try {
      const response = await fetch(APP_CONFIG.API_BASE + '/api/memory/config');
      if (response.ok) {
        const data = await response.json();
        memorySwitch.checked = data.enabled;

        const memoryLabel = document.getElementById('memory-label');
        if (memoryLabel) {
          memoryLabel.classList.toggle('active', data.enabled);
        }
      }
    } catch (e) {
      console.error("Failed to sync memory config", e);
    }
  }
}
