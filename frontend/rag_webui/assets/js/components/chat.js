// Chat Component

// ========== UI State Management ==========

/**
 * Agent Knowledge Base Requirements Configuration
 * - required: Must select a knowledge base to use
 * - optional: Optional, recommended to select a knowledge base
 * - none: No knowledge base required
 */
const AGENT_KB_REQUIREMENTS = {

  'ragref/kb_search/kb_search.yaml': 'required',           // KB Search
  'ragref/meta_retrieval/meta_retrieval.yaml': 'required', // Meta Retrieval
  'ragref/file/file_qa.yaml': 'required',                  // File QA
  'ragref/excel/excel.yaml': 'required',                   // Excel Agent
  'ragref/text2sql/text2sql.yaml': 'required',             // Text2SQL
  'orchestrator/parallel.yaml': 'required',                     // Parallel Orchestrator

  'simple/base.yaml': 'none',                              // Chat
  'simple/base_search.yaml': 'none',                       // Web Search
  'auto_select': 'optional',                               // Auto Select - optional knowledge base
};

/**
 * Get current Agent's knowledge base requirement
 */
function getCurrentAgentKbRequirement() {
  const agentSelect = document.getElementById('agent-select');
  if (!agentSelect || !agentSelect.value) {
    return 'none';
  }
  return AGENT_KB_REQUIREMENTS[agentSelect.value] || 'none';
}

/**
 * Update send button state
 */
function updateSendButtonState() {
  const input = document.getElementById('chat-input');
  const sendBtn = document.getElementById('send-btn');
  const agentSelect = document.getElementById('agent-select');
  const kbSelect = document.getElementById('kb-select');

  if (!input || !sendBtn) return;

  const hasContent = input.value.trim().length > 0;
  const hasAgent = agentSelect && agentSelect.value;
  const hasKb = kbSelect && kbSelect.value;

  // Get current Agent's knowledge base requirement
  const kbRequirement = getCurrentAgentKbRequirement();

  // Check if send conditions are met
  let canSend = hasContent && hasAgent;

  // If Agent requires knowledge base, check if knowledge base is selected
  if (kbRequirement === 'required') {
    canSend = canSend && hasKb;
  }

  if (canSend) {
    sendBtn.classList.remove('disabled');
    sendBtn.classList.add('active');
  } else {
    sendBtn.classList.add('disabled');
    sendBtn.classList.remove('active');
  }
}

/**
 * Switch input area mode (center -> bottom)
 */
function switchToBottomMode() {
  const container = document.getElementById('chat-input-container');
  if (container && container.classList.contains('center-mode')) {
    container.classList.remove('center-mode');
    container.classList.add('bottom-mode');
  }
}

/**
 * Switch input area mode (bottom -> center)
 */
function switchToCenterMode() {
  const container = document.getElementById('chat-input-container');
  if (container && container.classList.contains('bottom-mode')) {
    container.classList.remove('bottom-mode');
    container.classList.add('center-mode');
  }
}

/**
 * Update clear history button state
 */
function updateClearButtonState() {
  const clearBtn = document.getElementById('clear-history-btn');
  const messagesContainer = document.getElementById('chat-messages');

  if (!clearBtn || !messagesContainer) return;

  const hasMessages = messagesContainer.children.length > 0;

  if (hasMessages) {
    clearBtn.classList.remove('disabled');
    messagesContainer.classList.add('has-messages');
  } else {
    clearBtn.classList.add('disabled');
    messagesContainer.classList.remove('has-messages');
  }
}

/**
 * Update knowledge base selector prompt
 */
function updateKbSelectorHint() {
  const kbSelect = document.getElementById('kb-select');
  if (!kbSelect) return;

  const kbRequirement = getCurrentAgentKbRequirement();
  const firstOption = kbSelect.options[0];

  if (!firstOption) return;

  // Update prompt text and style based on Agent's knowledge base requirement
  if (kbRequirement === 'required') {
    firstOption.text = t('kb_required_hint');
    // Add visual hint for required (light yellow)
    if (!kbSelect.value) {
      kbSelect.style.borderColor = '#efb95bff';
      kbSelect.style.backgroundColor = '#fef3c7';
    }
  } else if (kbRequirement === 'optional') {
    firstOption.text = t('kb_optional_hint');
    // Restore default style
    kbSelect.style.borderColor = '';
    kbSelect.style.backgroundColor = '';
  } else {
    firstOption.text = t('kb_none_hint');
    // Restore default style
    kbSelect.style.borderColor = '';
    kbSelect.style.backgroundColor = '';
  }
}

// ========== Global State Variables ==========
let chatMessages = [];
let attachedFiles = [];
let isStreaming = false;
let currentAgent = '';
let currentKbId = null;
let knowledgeBaseFiles = [];
let selectedFileIds = [];
let selectedFilesData = []; // New: Store selected file details (including id and name)
// // New: Memory switch state
// let memoryEnabled = false;
// New: Rendered analysis event IDs
const renderedAnalysisIds = new Set();

// Card and streaming state management
let currentAbortController = null;
let lastActiveCard = null;
let currentReasoningCard = null;
let currentToolCallCard = null;
let currentToolOutputCard = null;
let currentOutputCard = null;
let currentExcelTaskCard = null;  // Excel Agent task card
let currentExcelTaskNodeId = null;  // Excel Agent workflow node ID

// Content buffers
let reasoningContent = '';
let toolCallArgs = '';
let toolOutputContent = '';
let outputContent = '';
let currentToolName = '';
let excelTaskContent = '';  // Excel Agent task content buffer

// Timers
let totalTimeStartTime = null;
let totalTimeTimer = null;
let totalElapsedTime = null;

// ========== Parallel Window Management ==========
let parallelWindows = {};
let isParallelMode = false;
let currentParallelContainer = null;
let parallelAgentCards = {};  // { agent_name: { reasoning: card, toolCall: card, output: card } }

/**
 * Get Agent icon
 */
function getAgentIcon(agentName) {
    const icons = {
        'ExcelQA': '📊',
        'KBSearch': '🔍',
        'Text2SQL': '🗄️',
        'WebSearch': '🌐',
        'ExcelAgent': '📈'
    };
    return icons[agentName] || '🤖';
}

/**
 * Create parallel container
 */
function createParallelContainer(tasks) {
    // Clear existing parallel container
    if (currentParallelContainer) {
        currentParallelContainer.remove();
        currentParallelContainer = null;
    }

    // Complete all active cards first
    if (lastActiveCard) {
        completeCard(lastActiveCard);
        lastActiveCard = null;
    }

    // Create wrapper
    const wrapper = document.createElement('div');

    // Create navigation bar (only show when tasks > 2)
    if (tasks.length > 2) {
        const navContainer = document.createElement('div');
        navContainer.className = 'parallel-nav-container';
        navContainer.innerHTML = `
            <div class="parallel-nav-tabs"></div>
            <div class="parallel-view-switcher">
                <button class="parallel-view-btn active" data-view="grid" title="${t('grid_view_title')}">
                    <span>⊞</span>
                    <span>${t('grid_view')}</span>
                </button>
                <button class="parallel-view-btn" data-view="tab" title="${t('tab_view_title')}">
                    <span>▭</span>
                    <span>${t('tab_view')}</span>
                </button>
            </div>
        `;
        wrapper.appendChild(navContainer);

        // Create navigation tab for each task
        const tabsContainer = navContainer.querySelector('.parallel-nav-tabs');
        tasks.forEach((task, index) => {
            const tab = document.createElement('div');
            tab.className = 'parallel-nav-tab';
            if (index === 0) tab.classList.add('active');
            tab.dataset.agentName = task.agent_name;
            tab.innerHTML = `
                <span class="parallel-nav-tab-icon">${getAgentIcon(task.agent_name)}</span>
                <span class="parallel-nav-tab-name" title="${task.agent_name}">${task.agent_name}</span>
                <span class="parallel-nav-tab-status">⏳</span>
            `;
            tab.onclick = () => switchParallelTab(task.agent_name);
            tabsContainer.appendChild(tab);
        });

        // Bind view switcher buttons
        navContainer.querySelectorAll('.parallel-view-btn').forEach(btn => {
            btn.onclick = () => switchParallelView(btn.dataset.view);
        });
    }

    // Create window container
    const container = document.createElement('div');
    const taskCount = tasks.length;
    const gridClass = taskCount <= 2 ? 'grid-2' : 'grid-4';
    container.className = `parallel-container ${gridClass} grid-mode`;
    wrapper.appendChild(container);

    // Add to stream
    const chatMessagesEl = document.getElementById('chat-messages');
    chatMessagesEl.appendChild(wrapper);

    currentParallelContainer = wrapper;
    parallelWindows = {};
    isParallelMode = true;

    // Create window for each task
    tasks.forEach((task, index) => {
        const window = createParallelWindow(task);
        if (index === 0) window.classList.add('tab-active');
        container.appendChild(window);
        parallelWindows[task.agent_name] = window;
    });

    // Scroll to view
    setTimeout(() => {
        wrapper.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }, 100);

    return wrapper;
}

/**
 * Create single parallel Agent window
 */
function createParallelWindow(task) {
    const window = document.createElement('div');
    window.className = 'parallel-agent-window';
    window.dataset.agentName = task.agent_name;

    window.innerHTML = `
        <div class="parallel-window-header">
            <div class="parallel-window-title">
                <span>${getAgentIcon(task.agent_name)}</span>
                <span>${task.agent_name}</span>
            </div>
            <div class="parallel-window-status">
                <span>⏳</span>
                <span>Waiting...</span>
            </div>
        </div>
        <div class="parallel-progress-bar">
            <div class="parallel-progress-bar-fill" style="width: 0%"></div>
        </div>
        <div class="parallel-window-content">
            <div class="parallel-window-empty">
                <div class="parallel-window-empty-icon">⏳</div>
                <div class="parallel-window-empty-text">Preparing to execute...</div>
            </div>
        </div>
    `;

    return window;
}

/**
 * Update parallel window status
 */
function updateParallelWindowStatus(agentName, status, message) {
    const window = parallelWindows[agentName];
    if (!window) return;

    const header = window.querySelector('.parallel-window-header');
    const statusEl = window.querySelector('.parallel-window-status');
    const progressBar = window.querySelector('.parallel-progress-bar-fill');

    // Update window status
    if (status === 'running') {
        window.classList.add('active');
        window.classList.remove('completed', 'error');
        header.classList.remove('completed', 'error');
        statusEl.innerHTML = `<span>▶️</span><span>${t('executing')}</span>`;
        progressBar.style.width = '50%';
    } else if (status === 'completed') {
        window.classList.add('completed');
        window.classList.remove('active', 'error');
        header.classList.add('completed');
        header.classList.remove('error');
        statusEl.innerHTML = `<span>✅</span><span>${t('completed')}</span>`;
        progressBar.style.width = '100%';
    } else if (status === 'error') {
        window.classList.add('error');
        window.classList.remove('active', 'completed');
        header.classList.add('error');
        header.classList.remove('completed');
        statusEl.innerHTML = `<span>❌</span><span>${message || t('failed')}</span>`;
        progressBar.style.width = '100%';
        progressBar.style.background = '#f56c6c';
    }

    // Sync update navigation tab status
    updateParallelNavTab(agentName, status);
}

/**
 * Update navigation tab status
 */
function updateParallelNavTab(agentName, status) {
    if (!currentParallelContainer) return;

    const tab = currentParallelContainer.querySelector(`.parallel-nav-tab[data-agent-name="${agentName}"]`);
    if (!tab) return;

    const statusIcon = tab.querySelector('.parallel-nav-tab-status');

    // Remove all status classes
    tab.classList.remove('completed', 'error');

    // Update status
    if (status === 'running') {
        statusIcon.textContent = '▶️';
    } else if (status === 'completed') {
        tab.classList.add('completed');
        statusIcon.textContent = '✅';
    } else if (status === 'error') {
        tab.classList.add('error');
        statusIcon.textContent = '❌';
    }
}

/**
 * Switch parallel tab
 */
function switchParallelTab(agentName) {
    if (!currentParallelContainer) return;

    const container = currentParallelContainer.querySelector('.parallel-container');
    if (!container) return;

    // If in grid mode, scroll to corresponding window
    if (container.classList.contains('grid-mode')) {
        const window = parallelWindows[agentName];
        if (window) {
            window.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
        return;
    }

    // Tab mode: switch displayed window
    // Remove active state from all windows
    Object.values(parallelWindows).forEach(w => {
        w.classList.remove('tab-active');
    });

    // Activate selected window
    const window = parallelWindows[agentName];
    if (window) {
        window.classList.add('tab-active');
    }

    // Update tab active state
    currentParallelContainer.querySelectorAll('.parallel-nav-tab').forEach(tab => {
        tab.classList.remove('active');
    });

    const activeTab = currentParallelContainer.querySelector(`.parallel-nav-tab[data-agent-name="${agentName}"]`);
    if (activeTab) {
        activeTab.classList.add('active');
    }
}

/**
 * Switch parallel view mode
 */
function switchParallelView(viewMode) {
    if (!currentParallelContainer) return;

    const container = currentParallelContainer.querySelector('.parallel-container');
    if (!container) return;

    const navContainer = currentParallelContainer.querySelector('.parallel-nav-container');
    if (!navContainer) return;

    // Update container class
    if (viewMode === 'grid') {
        container.classList.remove('tab-mode');
        container.classList.add('grid-mode');

        // In grid mode, remove tab-active class from all windows to show all windows
        Object.values(parallelWindows).forEach(w => {
            w.classList.remove('tab-active');
        });
    } else if (viewMode === 'tab') {
        container.classList.remove('grid-mode');
        container.classList.add('tab-mode');

        // In tab mode, ensure one window is active
        const activeTab = navContainer.querySelector('.parallel-nav-tab.active');
        const agentName = activeTab ? activeTab.dataset.agentName : Object.keys(parallelWindows)[0];

        Object.values(parallelWindows).forEach(w => {
            w.classList.remove('tab-active');
        });

        const activeWindow = parallelWindows[agentName];
        if (activeWindow) {
            activeWindow.classList.add('tab-active');
        }
    }

    // Update button state
    navContainer.querySelectorAll('.parallel-view-btn').forEach(btn => {
        btn.classList.remove('active');
        if (btn.dataset.view === viewMode) {
            btn.classList.add('active');
        }
    });
}

/**
 * Add content to parallel window
 */
function addContentToParallelWindow(agentName, card) {
    const window = parallelWindows[agentName];
    if (!window) {
        console.warn(`Parallel window not found for agent: ${agentName}`);
        return;
    }

    const content = window.querySelector('.parallel-window-content');

    // Remove empty state
    const emptyState = content.querySelector('.parallel-window-empty');
    if (emptyState) {
        emptyState.remove();
    }

    // Add card
    content.appendChild(card);

    // Scroll to bottom
    content.scrollTop = content.scrollHeight;
}

/**
 * Clean up parallel mode
 */
function exitParallelMode() {
    // Complete all agent cards, stop timers
    Object.keys(parallelAgentCards).forEach(agentName => {
        const agentCards = parallelAgentCards[agentName];
        if (agentCards.reasoning) {
            completeCard(agentCards.reasoning);
        }
        if (agentCards.toolCall) {
            completeCard(agentCards.toolCall);
        }
        if (agentCards.output) {
            completeCard(agentCards.output);
        }
    });

    isParallelMode = false;
    parallelWindows = {};
    currentParallelContainer = null;
    parallelAgentCards = {};
}

/**
 * Create parallel group header
 */
function createParallelGroupHeader(groupIdx, taskCount) {
    const chatMessagesEl = document.getElementById('chat-messages');

    const header = document.createElement('div');
    header.className = 'parallel-group-header';
    header.innerHTML = `
        <div class="parallel-group-header-icon">🚀</div>
        <div class="parallel-group-header-content">
            <div class="parallel-group-title">Parallel Task Group ${groupIdx + 1}</div>
            <div class="parallel-group-subtitle">${taskCount} tasks executing in parallel</div>
        </div>
    `;

    chatMessagesEl.appendChild(header);
    return header;
}

async function initMemorySwitch() {
  const memorySwitch = document.getElementById('memory-switch');
  if (memorySwitch) {
    memorySwitch.addEventListener('change', toggleMemorySwitch);
    
    // Initialize from backend API to get .env state
    try {
        const response = await fetch(API_BASE + '/api/memory/config');
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

async function toggleMemorySwitch() {
  const memorySwitch = document.getElementById('memory-switch');
  const isEnabled = !!(memorySwitch && memorySwitch.checked);

  // UI visual feedback
  const memoryLabel = document.getElementById('memory-label');
  if (memoryLabel) {
    memoryLabel.classList.toggle('active', isEnabled);
  }

  // Send request to modify backend .env
  try {
      await fetch(API_BASE + '/api/memory/config', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ enabled: isEnabled })
      });
      console.log(`Memory env updated to: ${isEnabled}`);
  } catch (e) {
      console.error("Failed to update memory config", e);
      // Optional: rollback UI state on failure
  }
}

// Expose toggleMemorySwitch to window for HTML onclick handler
window.toggleMemorySwitch = toggleMemorySwitch;

// Initialize chat page
async function initChat() {
  await loadAgents();
  await loadKnowledgeBasesForChat();
  setupMarkdownRenderer();
  // New: Initialize Memory switch
  initMemorySwitch();

  // Restore chat history from localStorage
  restoreChatHistory();

  // Setup agent change listener
  const agentSelect = document.getElementById('agent-select');
  if (agentSelect) {
    agentSelect.addEventListener('change', handleAgentChange);
  }

  // Add language switch listener
  i18n.onChange(() => {
    // Reload Agent and knowledge base selectors to update placeholder text
    loadAgents();
    loadKnowledgeBasesForChat();
    // Update knowledge base selector hint
    updateKbSelectorHint();
    // Update file upload button state (update title)
    updateFileUploadButtonState();
    // Reload file selector (if knowledge base is selected)
    const kbSelect = document.getElementById('kb-select');
    if (kbSelect && kbSelect.value) {
      loadKnowledgeBaseFiles(kbSelect.value);
    }
    // Update all copy button titles
    updateCopyButtonTitles();
  });
}

// Load agents
async function loadAgents() {
  try {
    const response = await API.getAgents();
    const agents = response.agents || [];
    currentAgent = response.current || '';

    const select = document.getElementById('agent-select');
    if (!select) return;

    // Add fixed Auto Select option
    const autoSelectOption = `<option value="auto_select">🤖 Auto Select</option>`;

    select.innerHTML = `<option value="">${t('select_agent_placeholder')}</option>` +
      autoSelectOption +
      agents.map(agent => {
        const icon = agent.icon || '🤖';
        const selected = agent.config_path === currentAgent ? 'selected' : '';
        return `<option value="${escapeHtml(agent.config_path)}" ${selected}>${icon} ${escapeHtml(agent.name)}</option>`;
      }).join('');

    // Add change listener
    select.addEventListener('change', handleAgentChange);

    // Update upload button state after loading
    updateFileUploadButtonState();

    console.log(`Loaded ${agents.length} agents, current: ${currentAgent}`);
  } catch (error) {
    console.error('Failed to load agents:', error);
    showToast(t('toast_load_agent_failed'), 'error');
  }
}

// Handle agent change
async function handleAgentChange(event) {
  const configPath = event.target.value;

  // Update upload button state
  updateFileUploadButtonState();

  // Update KB selector hint
  updateKbSelectorHint();

  // Update send button state
  updateSendButtonState();

  // Skip agent switch for auto_select (it's not a real agent config)
  if (configPath === 'auto_select') {
    console.log('Auto Select mode enabled - no agent switch needed');
    return;
  }

  if (!configPath) return;

  try {
    await API.switchAgent(configPath);
    currentAgent = configPath;

    // Get agent name for toast
    const selectedOption = event.target.options[event.target.selectedIndex];
    const agentName = selectedOption.text;

    showToast(t('toast_switch_agent_success', { name: agentName }), 'success');
    console.log(`Switched to agent: ${configPath}`);
  } catch (error) {
    console.error('Failed to switch agent:', error);
    showToast(t('toast_switch_agent_failed', { error: error.message }), 'error');

    // Revert selection
    event.target.value = currentAgent;
  }
}

// Load knowledge bases for chat
async function loadKnowledgeBasesForChat() {
  try {
    const data = await API.getKnowledgeBases();
    const kbs = data.knowledge_bases || data || [];
    const select = document.getElementById('kb-select');
    if (!select) return;

    select.innerHTML = `<option value="">${t('select_kb_placeholder')}</option>` +
      kbs.map(kb => `<option value="${escapeHtml(kb.id || kb.name)}">${escapeHtml(kb.name)}</option>`).join('');

    // Add change listener
    select.addEventListener('change', handleKnowledgeBaseChange);

    // Update upload button state after loading
    updateFileUploadButtonState();

    // Update KB selector hint based on current agent
    updateKbSelectorHint();
  } catch (error) {
    console.error('Failed to load knowledge bases:', error);
  }
}

// Handle knowledge base change
async function handleKnowledgeBaseChange(event) {
  const kbId = event.target.value;
  currentKbId = kbId;

  // Clear visual hint (if knowledge base is selected)
  const kbSelect = document.getElementById('kb-select');
  if (kbSelect && kbId) {
    kbSelect.style.borderColor = '';
    kbSelect.style.backgroundColor = '';
  }

  // Update upload button state
  updateFileUploadButtonState();

  // Update send button state
  updateSendButtonState();

  const fileSelect = document.getElementById('file-select');
  if (!fileSelect) return;

  if (!kbId) {
    // No KB selected, disable file selector
    fileSelect.disabled = true;
    fileSelect.innerHTML = `<option value="">${t('select_file_placeholder')}</option>`;
    selectedFileIds = [];
    selectedFilesData = [];
    knowledgeBaseFiles = [];
    renderSelectedFilesTags();
    return;
  }

  try {
    // Switch KB on backend
    await API.selectKnowledgeBase(kbId);

    // Clear previously selected files
    selectedFileIds = [];
    selectedFilesData = [];
    renderSelectedFilesTags();

    // Load files for selected KB
    await loadKnowledgeBaseFiles(kbId);

    showToast(t('toast_switch_kb_success'), 'success');
  } catch (error) {
    console.error('Failed to switch knowledge base:', error);
    showToast(t('toast_switch_kb_failed', { error: error.message }), 'error');
  }
}

// Load files for knowledge base
async function loadKnowledgeBaseFiles(kbId) {
  try {
    const files = await API.getKnowledgeBaseFiles(kbId);
    knowledgeBaseFiles = files || [];

    const fileSelect = document.getElementById('file-select');
    if (!fileSelect) return;

    if (knowledgeBaseFiles.length === 0) {
      fileSelect.disabled = true;
      fileSelect.innerHTML = `<option value="">${t('no_files_available')}</option>`;
      selectedFileIds = [];
      selectedFilesData = [];
      renderSelectedFilesTags();
      return;
    }

    // Enable file selector and populate options
    fileSelect.disabled = false;
    fileSelect.innerHTML = `<option value="">${t('select_file_placeholder')}</option>` +
      knowledgeBaseFiles.map(file =>
        `<option value="${escapeHtml(file.id)}">${escapeHtml(file.name || file.filename)}</option>`
      ).join('');

    console.log(`Loaded ${knowledgeBaseFiles.length} files for KB ${kbId}`);
  } catch (error) {
    console.error('Failed to load files:', error);
    showToast(t('toast_load_files_failed', { error: error.message }), 'error');
  }
}

// Handle file selection change (supports multiple selection)
function handleFileSelectChange(selectElement) {
  const fileId = selectElement.value;
  const fileName = selectElement.options[selectElement.selectedIndex].text;

  if (!fileId) {
    return;
  }

  // Check if file is already selected
  const alreadySelected = selectedFilesData.some(f => f.id === fileId);
  if (alreadySelected) {
    showToast(t('toast_file_already_selected'), 'warning');
    selectElement.value = ''; // Reset selector
    return;
  }

  // Add to selected files list
  selectedFilesData.push({
    id: fileId,
    name: fileName
  });

  // Update selectedFileIds array (for API calls)
  selectedFileIds = selectedFilesData.map(f => f.id);

  // Render file tags
  renderSelectedFilesTags();

  // Reset selector to default option
  selectElement.value = '';

  console.log(`Selected file: ${fileName}, currently selected ${selectedFilesData.length} files`);
}

/**
 * Render selected files tags
 */
function renderSelectedFilesTags() {
  const container = document.getElementById('selected-files-tags');
  if (!container) return;

  if (selectedFilesData.length === 0) {
    container.style.display = 'none';
    container.innerHTML = '';
    return;
  }

  container.style.display = 'flex';
  container.innerHTML = selectedFilesData.map((file, index) => {
    // Determine file icon
    let icon = '📄';
    const fileName = file.name.toLowerCase();
    if (fileName.endsWith('.pdf')) {
      icon = '📄';
    } else if (fileName.endsWith('.xlsx') || fileName.endsWith('.xls') || fileName.endsWith('.csv')) {
      icon = '📊';
    } else if (fileName.endsWith('.png') || fileName.endsWith('.jpg') || fileName.endsWith('.jpeg')) {
      icon = '🖼️';
    } else if (fileName.endsWith('.txt') || fileName.endsWith('.md')) {
      icon = '📄';
    }

    return `
      <div class="file-tag" title="${escapeHtml(file.name)}">
        <span class="file-tag-icon">${icon}</span>
        <span class="file-tag-name">${escapeHtml(file.name)}</span>
        <span class="file-tag-remove" onclick="removeSelectedFileTag(${index})" title="${t('remove')}">×</span>
      </div>
    `;
  }).join('');
}

/**
 * Remove selected file tag
 */
function removeSelectedFileTag(index) {
  if (index < 0 || index >= selectedFilesData.length) return;

  const removedFile = selectedFilesData[index];
  selectedFilesData.splice(index, 1);

  // Update selectedFileIds array
  selectedFileIds = selectedFilesData.map(f => f.id);

  // Re-render tags
  renderSelectedFilesTags();

  showToast(t('toast_file_removed', { name: removedFile.name }), 'success');
  console.log(`Removed file: ${removedFile.name}, remaining ${selectedFilesData.length} files`);
}

// IME input state flag
let isComposing = false;

// Handle composition events (IME input)
function handleCompositionStart() {
  isComposing = true;
}

function handleCompositionEnd() {
  isComposing = false;
}

// Handle chat input keydown
function handleChatInputKeydown(event) {
  // Use event.isComposing or custom flag to determine if IME is active
  // event.isComposing is a browser-native property, more reliable
  if (event.isComposing || isComposing) {
    return;
  }
  
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    sendMessage();
  }
}

// Send message
async function sendMessage() {
  const input = document.getElementById('chat-input');
  const sendBtn = document.getElementById('send-btn');

  // DEBUG: Log selectedFileIds before sending
  console.log(`[SendMessage] selectedFileIds:`, selectedFileIds);
  console.log(`[SendMessage] selectedFilesData:`, selectedFilesData);
  console.log(`[SendMessage] currentKbId:`, currentKbId);

  // If streaming, click to stop
  if (isStreaming) {
    stopAgent();
    return;
  }

  const message = input.value.trim();

  if (!message) {
    showToast(t('toast_enter_message'), 'warning');
    return;
  }

  const agentSelect = document.getElementById('agent-select');
  const kbSelect = document.getElementById('kb-select');

  // Check if Agent is selected
  if (!agentSelect.value) {
    showToast(t('toast_select_agent'), 'warning');
    return;
  }

  // Get current Agent's knowledge base requirement
  const kbRequirement = getCurrentAgentKbRequirement();

  // Validate based on Agent's knowledge base requirement
  if (kbRequirement === 'required' && !kbSelect.value) {
    // Get Agent name for prompt
    const selectedOption = agentSelect.options[agentSelect.selectedIndex];
    const agentName = selectedOption.text;
    showToast(t('toast_agent_requires_kb', { name: agentName }), 'warning');
    return;
  }

  // Switch to bottom mode (on first send)
  switchToBottomMode();

  // Add user message to chat
  addMessage('user', message);

  // Clear input
  input.value = '';

  // Update send button state
  updateSendButtonState();

  // Change button to stop state
  sendBtn.innerHTML = '<span style="font-size: 16px;">⏹</span>';
  sendBtn.classList.remove('active', 'disabled');
  sendBtn.classList.add('stop-mode');
  isStreaming = true;

  try {
    // Check if Auto Select is selected
    if (agentSelect.value === 'auto_select') {
      // Auto Select mode: use smart selection API
      const requestBody = {
        query: message,
        stream: true,
        session_id: null,
        kb_id: kbSelect.value ? parseInt(kbSelect.value) : null,
        file_ids: selectedFileIds.length > 0 ? selectedFileIds.map(id => String(id)) : null,
        use_memory: document.getElementById('memory-switch')?.checked || false,
        auto_select: true  // Mark as auto-select mode
      };

      // Clear attached files after sending
      attachedFiles = [];
      renderAttachedFiles();

      // Stream response with auto select
      await streamChatResponse(requestBody);
    } else {

      const requestBody = {
        query: message,
        stream: true,
        session_id: null,
        kb_id: kbSelect.value ? parseInt(kbSelect.value) : null,
        file_ids: selectedFileIds.length > 0 ? selectedFileIds.map(id => String(id)) : null,
        // Modified: read DOM element checked property directly, no longer rely on global let variable
        use_memory: document.getElementById('memory-switch')?.checked || false
      };

      // Clear attached files after sending
      attachedFiles = [];
      renderAttachedFiles();

      // Stream response
      await streamChatResponse(requestBody);
    }

  } catch (error) {
    // If not user-initiated interruption, show error
    if (error.name !== 'AbortError') {
      showToast(t('toast_send_failed', { error: error.message }), 'error');
      addMessage('assistant', t('error_sorry', { error: error.message }));
    }
  } finally {
    // Restore button state
    sendBtn.innerHTML = '<img src="assets/images/send.svg" alt="发送" />';
    sendBtn.classList.remove('stop-mode');
    isStreaming = false;
    updateSendButtonState(); // Update button state based on input content
  }
}

/**
 * Stop Agent execution
 */
function stopAgent() {
  if (currentAbortController) {
    currentAbortController.abort();
    console.log('Agent execution stopped by user');

    // Complete all active cards
    if (lastActiveCard) {
      completeCard(lastActiveCard);
      lastActiveCard = null;
    }

    // Stop timers
    hideTotalTimeDisplay();

    // Add interruption message
    const messagesContainer = document.getElementById('chat-messages');
    if (messagesContainer) {
      const interruptDiv = document.createElement('div');
      interruptDiv.className = 'message system';
      interruptDiv.innerHTML = `
        <div style="text-align: center; color: var(--warning); font-size: 13px; padding: 8px; margin: 8px 0;">
          ⚠️ Agent execution interrupted by user
        </div>
      `;
      messagesContainer.appendChild(interruptDiv);
      messagesContainer.scrollTop = messagesContainer.scrollHeight;
      
      // Save chat history after adding interruption message
      saveChatHistory();
    }

    showToast(t('toast_execution_stopped'), 'info');
  }
}

// Add message to chat
function addMessage(role, content) {
  const messagesContainer = document.getElementById('chat-messages');
  if (!messagesContainer) return;

  // Remove empty state if exists
  const emptyState = messagesContainer.querySelector('.empty-state');
  if (emptyState) {
    emptyState.remove();
  }

  const messageDiv = document.createElement('div');
  messageDiv.className = `message ${role}`;

  const time = new Date().toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' });

  // Store original content for copying
  messageDiv.dataset.originalContent = content;

  // Add copy button (all messages have it)
  const copyButton = `<button class="copy-btn" onclick="copyMessageContent(this)" title="${t('copy_content')}">📋</button>`;

  messageDiv.innerHTML = `
    <div class="message-content">
      <div class="message-bubble">
        <div class="message-text">${role === 'assistant' ? renderMarkdown(content) : escapeHtml(content)}</div>
      </div>
      <div class="message-footer">
        ${copyButton}
        <div class="message-time">${time}</div>
      </div>
    </div>
  `;

  messagesContainer.appendChild(messageDiv);
  messagesContainer.scrollTop = messagesContainer.scrollHeight;

  chatMessages.push({ role, content, time });

  // Update clear history button state
  updateClearButtonState();

  // Save chat history to localStorage
  saveChatHistory();
}

// Stream chat response with intermediate steps
// ========== Streaming Response Helper Functions ==========

/**
 * Reset all card states and content buffers
 */
function resetStreamState() {
  // Reset card states
  currentReasoningCard = null;
  currentToolCallCard = null;
  currentToolOutputCard = null;
  currentOutputCard = null;
  currentExcelTaskCard = null;
  currentExcelTaskNodeId = null;
  lastActiveCard = null;

  // Reset content buffers
  reasoningContent = '';
  toolCallArgs = '';
  toolOutputContent = '';
  outputContent = '';
  currentToolName = '';
  excelTaskContent = '';
}

/**
 * Determine if event should be routed to parallel window
 */
function shouldRouteToParallel(event) {
  return isParallelMode && event.agent_name && parallelWindows[event.agent_name];
}

/**
 * Complete all currently active cards
 */
function completeAllActiveCards() {
  if (currentReasoningCard) {
    completeCard(currentReasoningCard);
    currentReasoningCard = null;
  }
  if (currentToolCallCard) {
    completeCard(currentToolCallCard);
    currentToolCallCard = null;
  }
  if (currentToolOutputCard) {
    completeCard(currentToolOutputCard);
    currentToolOutputCard = null;
  }
  if (currentOutputCard) {
    completeCard(currentOutputCard);
    currentOutputCard = null;
  }
  if (currentExcelTaskCard) {
    completeCard(currentExcelTaskCard);
    currentExcelTaskCard = null;
    currentExcelTaskNodeId = null;
  }
  if (lastActiveCard) {
    completeCard(lastActiveCard);
    lastActiveCard = null;
  }

  // Reset content buffers
  reasoningContent = '';
  toolCallArgs = '';
  toolOutputContent = '';
  outputContent = '';
  excelTaskContent = '';
}

// ========== Parallel Mode Event Handlers ==========

/**
 * Handle parallel group start event
 */
function handleParallelGroupStart(event) {
  console.log('Parallel group starting:', event);

  // Complete previous active card first
  if (lastActiveCard) {
    completeCard(lastActiveCard);
    lastActiveCard = null;
  }

  // Create parallel group header
  if (event.group_idx !== undefined) {
    createParallelGroupHeader(event.group_idx, event.tasks?.length || 0);
  }

  // Create parallel window container
  if (event.tasks && event.tasks.length > 0) {
    createParallelContainer(event.tasks);
  }
}

/**
 * Handle parallel task start event
 */
function handleParallelTaskStart(event) {
  console.log('Parallel task starting:', event.agent_name, event.task);

  // Update window status
  if (event.agent_name) {
    updateParallelWindowStatus(event.agent_name, 'running');
  }
}

/**
 * Handle parallel task complete event
 */
function handleParallelTaskDone(event) {
  console.log('Parallel task completed:', event.agent_name);

  // Update window status
  if (event.agent_name) {
    updateParallelWindowStatus(event.agent_name, 'completed');

    // Complete all cards for this agent, stop timers
    if (parallelAgentCards[event.agent_name]) {
      const agentCards = parallelAgentCards[event.agent_name];
      if (agentCards.reasoning) {
        completeCard(agentCards.reasoning);
      }
      if (agentCards.toolCall) {
        completeCard(agentCards.toolCall);
      }
      if (agentCards.output) {
        completeCard(agentCards.output);
      }
    }
  }
}

/**
 * Handle parallel task error event
 */
function handleParallelTaskError(event) {
  console.error('Parallel task error:', event.agent_name, event.error);

  // Update window status
  if (event.agent_name) {
    updateParallelWindowStatus(event.agent_name, 'error', event.error);
  }
}

/**
 * Handle parallel group complete event
 */
function handleParallelGroupDone() {
  console.log('Parallel group completed');
  // Can add visual feedback for group completion
}

/**
 * Handle merge start event
 */
function handleMergeStart() {
  console.log('Starting result merge');

  // Complete all windows first
  Object.keys(parallelWindows).forEach(agentName => {
    const window = parallelWindows[agentName];
    if (window && !window.classList.contains('completed')) {
      updateParallelWindowStatus(agentName, 'completed');
    }
  });
}

/**
 * Handle merge complete event
 */
function handleMergeDone() {
  console.log('Result merge completed');
  exitParallelMode();
}

// ========== Core Event Handlers ==========

/**
 * Handle reasoning event - parallel mode
 */
function handleReasoningParallel(event) {
  if (event.done) {
    // Reasoning complete, collapse card
    if (parallelAgentCards[event.agent_name]?.reasoning) {
      completeCard(parallelAgentCards[event.agent_name].reasoning);
      parallelAgentCards[event.agent_name].reasoning = null;
    }
  } else if (event.content) {
    if (!parallelAgentCards[event.agent_name]) {
      parallelAgentCards[event.agent_name] = {};
    }

    if (!parallelAgentCards[event.agent_name].reasoning) {
      const card = createCollapsibleCard('Reasoning', '🧠', '#f0f9ff');
      parallelAgentCards[event.agent_name].reasoning = card;
      parallelAgentCards[event.agent_name].reasoningContent = '';
      addContentToParallelWindow(event.agent_name, card);
    }

    parallelAgentCards[event.agent_name].reasoningContent += event.content;
    updateCardContent(
      parallelAgentCards[event.agent_name].reasoning,
      parallelAgentCards[event.agent_name].reasoningContent,
      'text'
    );
  }
}

/**
 * Handle reasoning event - normal mode
 */
function handleReasoningNormal(event) {
  if (event.done) {
    // Reasoning complete, collapse card
    if (currentReasoningCard) {
      completeCard(currentReasoningCard);
      lastActiveCard = null;
      currentReasoningCard = null;
    }
  } else if (event.content) {
    if (!currentReasoningCard) {
      // Complete previous active card first
      if (lastActiveCard && lastActiveCard !== currentReasoningCard) {
        completeCard(lastActiveCard);
      }
      currentReasoningCard = createCollapsibleCard('Reasoning', '🧠', '#f0f9ff');
      lastActiveCard = currentReasoningCard;
      reasoningContent = '';
    }
    reasoningContent += event.content;
    updateCardContent(currentReasoningCard, reasoningContent, 'text');
  }
}

/**
 * Handle tool call event - parallel mode
 */
function handleToolCallParallel(event) {
  if (event.done) {
    if (parallelAgentCards[event.agent_name]?.toolCall) {
      completeCard(parallelAgentCards[event.agent_name].toolCall);
      parallelAgentCards[event.agent_name].toolCall = null;
    }
  } else {
    const toolName = event.tool_name;

    if (!parallelAgentCards[event.agent_name]) {
      parallelAgentCards[event.agent_name] = {};
    }

    // If new tool, complete previous tool card first
    if (parallelAgentCards[event.agent_name].toolCall &&
        parallelAgentCards[event.agent_name].toolCall.dataset.toolName !== toolName) {
      completeCard(parallelAgentCards[event.agent_name].toolCall);
      parallelAgentCards[event.agent_name].toolCall = null;
    }

    if (!parallelAgentCards[event.agent_name].toolCall) {
      const card = createCollapsibleCard(`Tool use: ${toolName}`, '🔧', '#ecf5ff');
      card.dataset.toolName = toolName;
      parallelAgentCards[event.agent_name].toolCall = card;
      parallelAgentCards[event.agent_name].toolCallArgs = '';
      addContentToParallelWindow(event.agent_name, card);
    }

    if (event.arguments) {
      parallelAgentCards[event.agent_name].toolCallArgs = event.arguments;
      updateCardContent(
        parallelAgentCards[event.agent_name].toolCall,
        parallelAgentCards[event.agent_name].toolCallArgs,
        event.mode || 'json'
      );
    } else if (event.arguments_delta) {
      parallelAgentCards[event.agent_name].toolCallArgs += event.arguments_delta;
      updateCardContent(
        parallelAgentCards[event.agent_name].toolCall,
        parallelAgentCards[event.agent_name].toolCallArgs,
        event.mode || 'json'
      );
    }
  }
}

/**
 * Handle tool call event - normal mode
 */
function handleToolCallNormal(event) {
  if (event.done) {
    // Tool call parameters received, collapse card
    if (currentToolCallCard) {
      completeCard(currentToolCallCard);
      lastActiveCard = null;
      currentToolCallCard = null;
    }
  } else {
    currentToolName = event.tool_name;

    // If new tool, complete previous tool card first
    if (currentToolCallCard && currentToolCallCard.dataset.toolName !== currentToolName) {
      completeCard(currentToolCallCard);
      currentToolCallCard = null;
    }

    // Create new tool call card
    if (!currentToolCallCard) {
      // Complete previous active card first
      if (lastActiveCard && lastActiveCard !== currentToolCallCard) {
        completeCard(lastActiveCard);
      }
      currentToolCallCard = createCollapsibleCard(`Tool use: ${currentToolName}`, '🔧', '#ecf5ff');
      currentToolCallCard.dataset.toolName = currentToolName;
      lastActiveCard = currentToolCallCard;
      toolCallArgs = '';
    }

    // Update parameter content
    if (event.arguments) {
      toolCallArgs = event.arguments;
      updateCardContent(currentToolCallCard, toolCallArgs, event.mode || 'json');
    } else if (event.arguments_delta) {
      toolCallArgs += event.arguments_delta;
      updateCardContent(currentToolCallCard, toolCallArgs, event.mode || 'json');
    }
  }
}

/**
 * Handle tool output event - parallel mode
 */
function handleToolOutputParallel(event) {
  console.log('[tool_output] Parallel mode:', event.agent_name);

  if (!parallelAgentCards[event.agent_name]) {
    parallelAgentCards[event.agent_name] = {};
  }

  // Create tool output card
  const card = createCollapsibleCard('Tool output', '📤', '#f0fdf4');
  const content = event.output || '';
  updateCardContent(card, content, 'text');
  addContentToParallelWindow(event.agent_name, card);

  // Tool output is one-time, complete and collapse immediately
  setTimeout(() => {
    if (card) {
      completeCard(card);
    }
  }, 500);
}

/**
 * Handle tool output event - normal mode
 */
function handleToolOutputNormal(event) {
  // Complete previous tool output card first
  if (currentToolOutputCard) {
    completeCard(currentToolOutputCard);
    currentToolOutputCard = null;
  }

  // Complete previous active card first
  if (lastActiveCard && lastActiveCard !== currentToolOutputCard) {
    completeCard(lastActiveCard);
  }

  // Create new tool output card
  currentToolOutputCard = createCollapsibleCard('Tool output', '📤', '#f0fdf4');
  lastActiveCard = currentToolOutputCard;
  toolOutputContent = event.output || '';
  updateCardContent(currentToolOutputCard, toolOutputContent, 'text');

  // Tool output is one-time, complete and collapse immediately
  setTimeout(() => {
    if (currentToolOutputCard) {
      completeCard(currentToolOutputCard);
      lastActiveCard = null;
      currentToolOutputCard = null;
    }
  }, 500);
}

/**
 * Handle text output event - parallel mode
 */
function handleDeltaParallel(event) {
  if (event.done) {
    if (parallelAgentCards[event.agent_name]?.output) {
      completeCard(parallelAgentCards[event.agent_name].output);
      parallelAgentCards[event.agent_name].output = null;
    }
  } else if (event.content) {
    if (!parallelAgentCards[event.agent_name]) {
      parallelAgentCards[event.agent_name] = {};
    }

    if (!parallelAgentCards[event.agent_name].output) {
      const card = createCollapsibleCard('Analysis', '💬', '#f5f7fa');
      parallelAgentCards[event.agent_name].output = card;
      parallelAgentCards[event.agent_name].outputContent = '';
      addContentToParallelWindow(event.agent_name, card);
    }

    parallelAgentCards[event.agent_name].outputContent += event.content;
    updateCardContent(
      parallelAgentCards[event.agent_name].output,
      parallelAgentCards[event.agent_name].outputContent,
      'text'
    );
  }
}

/**
 * Handle text output event - normal mode
 */
function handleDeltaNormal(event) {
  if (event.done) {
    // Output complete, collapse card
    if (currentOutputCard) {
      completeCard(currentOutputCard);
      lastActiveCard = null;
      currentOutputCard = null;
    }
  } else if (event.content) {
    // Create output card (first time)
    if (!currentOutputCard) {
      // Complete previous active card first
      if (lastActiveCard && lastActiveCard !== currentOutputCard) {
        completeCard(lastActiveCard);
      }
      currentOutputCard = createCollapsibleCard('Analysis', '💬', '#f5f7fa');
      lastActiveCard = currentOutputCard;
      outputContent = '';
    }
    outputContent += event.content;
    updateCardContent(currentOutputCard, outputContent, 'text');
  }
}

/**
 * Handle tool log event
 */
function handleToolLog(event) {
  const routeToParallel = shouldRouteToParallel(event);

  if (routeToParallel) {
    // Parallel mode
    if (!parallelAgentCards[event.agent_name]) {
      parallelAgentCards[event.agent_name] = {};
    }

    // Create tool log card
    const card = createCollapsibleCard(event.tool_name || 'Tool Log', '📝', '#f0fdf4');
    updateCardContent(card, event.message || '', 'text');
    addContentToParallelWindow(event.agent_name, card);

    // Complete and collapse immediately
    setTimeout(() => {
      if (card) {
        completeCard(card);
      }
    }, 100);
  } else {
    // Normal mode
    // Complete previous active card first
    if (lastActiveCard) {
      completeCard(lastActiveCard);
    }

    // Create tool log card
    const toolLogCard = createCollapsibleCard(event.tool_name || 'Tool Log', '📝', '#f0fdf4');
    updateCardContent(toolLogCard, event.message || '', 'text');

    // Complete and collapse immediately
    setTimeout(() => {
      if (toolLogCard) {
        completeCard(toolLogCard);
      }
    }, 100);

    lastActiveCard = null;
  }
}

/**
 * Handle run_item event
 */
function handleRunItem(event) {
  console.log('Run item event:', event.item_type, 'agent_name:', event.agent_name);

  const routeToParallel = shouldRouteToParallel(event);

  if (!routeToParallel) {
    // Normal mode：完成所有当前活跃的卡片
    completeAllActiveCards();
  }

  // Create corresponding card based on item_type
  let cardIcon = '📦';
  let cardTitle = 'Run Item';
  let cardContent = '';
  let bgColor = '#fff9e6';

  if (event.item_type === 'reasoning_item') {
    cardIcon = '🧠';
    cardTitle = 'Reasoning Item';
    cardContent = event.reasoning_summary || 'No reasoning summary';
    bgColor = '#f0f9ff';
  } else if (event.item_type === 'tool_call_item') {
    cardIcon = '🔧';
    cardTitle = `Tool Call: ${event.tool_name || 'unknown'}`;
    cardContent = event.tool_arguments || '';
    bgColor = '#ecf5ff';
  } else if (event.item_type === 'tool_call_output_item') {
    cardIcon = '📤';
    cardTitle = 'Tool Output';
    cardContent = event.tool_output || '';
    bgColor = '#f0fdf4';
  } else if (event.item_type === 'handoff_call_item') {
    cardIcon = '🔀';
    cardTitle = `Handoff: ${event.handoff_name || 'unknown'}`;
    cardContent = event.handoff_arguments || '';
    bgColor = '#fef3f2';
  } else if (event.item_type === 'handoff_output_item') {
    cardIcon = '🔄';
    cardTitle = 'Handoff Output';
    cardContent = `From: ${event.source_agent || 'unknown'}\nTo: ${event.target_agent || 'unknown'}`;
    bgColor = '#f0fdf4';
  } else {
    cardTitle = `Run Item: ${event.item_type}`;
    cardContent = event.raw_info || JSON.stringify(event, null, 2);
  }

  // Create card
  const card = createCollapsibleCard(cardTitle, cardIcon, bgColor);
  updateCardContent(card, cardContent, 'text');

  // Route to parallel window or main stream
  if (routeToParallel) {
    if (!parallelAgentCards[event.agent_name]) {
      parallelAgentCards[event.agent_name] = {};
    }
    addContentToParallelWindow(event.agent_name, card);
  }

  // Complete card immediately
  setTimeout(() => {
    if (card) {
      completeCard(card);
    }
  }, 100);
}

/**
 * Handle Excel Agent event - parallel mode
 */
function handleExcelAgentParallel(event) {
  if (event.done) {
    // Excel Agent task complete, collapse card
    if (parallelAgentCards[event.agent_name].excelTask) {
      // If has title, reset card title
      if (event.title) {
        const titleElement = parallelAgentCards[event.agent_name].excelTask.querySelector('.execution-card-title span:last-child');
        if (titleElement) {
          titleElement.textContent = event.title;
        }
      }
      completeCard(parallelAgentCards[event.agent_name].excelTask);
      parallelAgentCards[event.agent_name].excelTask = null;
      parallelAgentCards[event.agent_name].excelTaskContent = '';
    }
    // Complete workflow tree node
    if (parallelAgentCards[event.agent_name].excelTaskNodeId) {
      completeWorkflowNode(parallelAgentCards[event.agent_name].excelTaskNodeId);
      parallelAgentCards[event.agent_name].excelTaskNodeId = null;
    }
  } else if (event.content) {
    // Create or update Excel Task card
    if (!parallelAgentCards[event.agent_name].excelTask) {
      // Complete previous active card first
      if (parallelAgentCards[event.agent_name].lastActiveCard &&
          parallelAgentCards[event.agent_name].lastActiveCard !== parallelAgentCards[event.agent_name].excelTask) {
        completeCard(parallelAgentCards[event.agent_name].lastActiveCard);
      }

      const card = createCollapsibleCard(event.title || 'Excel Agent', '🔧', '#f5f7fa');
      parallelAgentCards[event.agent_name].excelTask = card;
      parallelAgentCards[event.agent_name].lastActiveCard = card;
      parallelAgentCards[event.agent_name].excelTaskContent = '';
      addContentToParallelWindow(event.agent_name, card);

      // Create workflow tree node
      const node = createWorkflowNode('excel_agent', event.title || 'Excel Agent', '🔧');
      parallelAgentCards[event.agent_name].excelTaskNodeId = node.id;
      workflowTree.currentStack.push(node.id);
    }

    // If has clean flag, clear previous content
    if (event.clean) {
      parallelAgentCards[event.agent_name].excelTaskContent = '';
    }

    // Update content
    parallelAgentCards[event.agent_name].excelTaskContent += event.content;
    updateCardContent(
      parallelAgentCards[event.agent_name].excelTask,
      parallelAgentCards[event.agent_name].excelTaskContent,
      event.mode || 'text'
    );
  }
}

/**
 * Handle Excel Agent event - normal mode
 */
function handleExcelAgentNormal(event) {
  if (event.done) {
    // Excel Agent task complete, collapse card
    if (currentExcelTaskCard) {
      // If has title, reset card title
      if (event.title) {
        const titleElement = currentExcelTaskCard.querySelector('.execution-card-title span:last-child');
        if (titleElement) {
          titleElement.textContent = event.title;
        }
      }
      completeCard(currentExcelTaskCard);
      lastActiveCard = null;
      currentExcelTaskCard = null;
      excelTaskContent = '';
    }
    // Complete workflow tree node
    if (currentExcelTaskNodeId) {
      completeWorkflowNode(currentExcelTaskNodeId);
      currentExcelTaskNodeId = null;
    }
  } else if (event.content) {
    // Create or update Excel Task card
    if (!currentExcelTaskCard) {
      // Complete previous active card first
      if (lastActiveCard && lastActiveCard !== currentExcelTaskCard) {
        completeCard(lastActiveCard);
      }
      currentExcelTaskCard = createCollapsibleCard(event.title || 'Excel Agent', '🔧', '#f5f7fa');
      lastActiveCard = currentExcelTaskCard;
      excelTaskContent = '';

      // Create workflow tree node
      const node = createWorkflowNode('excel_agent', event.title || 'Excel Agent', '🔧');
      currentExcelTaskNodeId = node.id;
      workflowTree.currentStack.push(node.id);
    }

    // If has clean flag, clear previous content
    if (event.clean) {
      excelTaskContent = '';
    }

    // Update content
    excelTaskContent += event.content;
    updateCardContent(currentExcelTaskCard, excelTaskContent, event.mode || 'text');
  }
}

/**
 * Handle completion event
 */
function handleDone(event) {
  console.log('Agent done event:', event);

  // Check if need to terminate current card
  if (event.terminate_card) {
    console.log('Terminating current card due to terminate_card flag');
    completeAllActiveCards();
    // Do not end entire process, continue waiting for subsequent events
  } else if (event.final_output) {
    // This is the final completion signal, contains final_output
    console.log('Agent finished with final output');

    // Before showing final answer, complete last active card
    if (lastActiveCard) {
      completeCard(lastActiveCard);
      lastActiveCard = null;
    }

    // Stop total time timer
    stopTotalTimeTimer();

    // Collapse all intermediate process execution cards
    const messagesContainer = document.getElementById('chat-messages');
    if (messagesContainer) {
      const allCards = messagesContainer.querySelectorAll('.execution-card');
      allCards.forEach(card => {
        completeCard(card, true); // Pass true parameter to collapse card
      });
    }

    // Show final answer
    addFinalAnswerBubble(event.final_output);
  } else {
    // Normal done signal (like single part completion)
    console.log('Done signal received, completing current active card');

    if (lastActiveCard) {
      completeCard(lastActiveCard);
      lastActiveCard = null;
    }
  }
}

// ========== Event Dispatcher ==========

/**
 * Dispatch event to corresponding handler
 */
function dispatchEvent(event) {
  // Parallel task events
  if (event.type === 'parallel_group.start') {
    handleParallelGroupStart(event);
  }
  else if (event.type === 'parallel_task.start') {
    handleParallelTaskStart(event);
  }
  else if (event.type === 'parallel_task.done') {
    handleParallelTaskDone(event);
  }
  else if (event.type === 'parallel_task.error') {
    handleParallelTaskError(event);
  }
  else if (event.type === 'parallel_group.done') {
    handleParallelGroupDone();
  }
  else if (event.type === 'merge.start') {
    handleMergeStart();
  }
  else if (event.type === 'merge.done') {
    handleMergeDone();
  }
  // Analysis events
  else if (event.type === 'analysis') {
    renderAnalysis(event);
  }
  // Reasoning process
  else if (event.type === 'reasoning') {
    if (shouldRouteToParallel(event)) {
      handleReasoningParallel(event);
    } else {
      handleReasoningNormal(event);
    }
  }
  // Tool calls
  else if (event.type === 'tool_call') {
    if (shouldRouteToParallel(event)) {
      handleToolCallParallel(event);
    } else {
      handleToolCallNormal(event);
    }
  }
  // Tool output
  else if (event.type === 'tool_output') {
    console.log('[tool_output] Event:', {
      agent_name: event.agent_name,
      isParallelMode,
      hasWindow: event.agent_name ? !!parallelWindows[event.agent_name] : false,
      availableWindows: Object.keys(parallelWindows)
    });

    if (shouldRouteToParallel(event)) {
      handleToolOutputParallel(event);
    } else {
      handleToolOutputNormal(event);
    }
  }
  // Tool logs
  else if (event.type === 'tool_log') {
    handleToolLog(event);
  }
  // Text output
  else if (event.type === 'delta') {
    if (shouldRouteToParallel(event)) {
      handleDeltaParallel(event);
    } else {
      handleDeltaNormal(event);
    }
  }
  // RunItemStreamEvent
  else if (event.type === 'run_item') {
    handleRunItem(event);
  }
  // Excel Agent specific events
  else if (event.type === 'excel_agent_event') {
    console.log('Excel Agent event:', event.title, 'done:', event.done);

    if (shouldRouteToParallel(event)) {
      if (!parallelAgentCards[event.agent_name]) {
        parallelAgentCards[event.agent_name] = {};
      }
      handleExcelAgentParallel(event);
    } else {
      handleExcelAgentNormal(event);
    }
  }
  // Completion events
  else if (event.type === 'done') {
    handleDone(event);
  }
  // Error handling
  else if (event.type === 'error') {
    // Stop total time timer
    hideTotalTimeDisplay();
    throw new Error(event.error || 'Stream error');
  }
}

async function streamChatResponse(messageData) {
  try {
    // Reset state
    resetStreamState();

    // Create AbortController for interruption
    currentAbortController = new AbortController();

    // Start total time timer
    startTotalTimeTimer();

    // Send POST request to /api/chat with stream enabled
    const response = await fetch(API_BASE + '/api/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(messageData),
      signal: currentAbortController.signal
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: response.statusText }));
      throw new Error(error.detail || 'Request failed');
    }

    // Process SSE stream
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || ''; // Keep incomplete line in buffer

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6).trim();
          if (data === '[DONE]') continue;

          try {
            const event = JSON.parse(data);

            // Use event dispatcher to handle all events
            dispatchEvent(event);

          } catch (e) {
            console.error('Error parsing SSE event:', e, data);
          }
        }
      }
    }

  } catch (error) {
    // If user-initiated interruption, do not show error
    if (error.name === 'AbortError') {
      console.log('Request aborted by user');
      return;
    }

    console.error('Stream error:', error);

    // Stop total time timer
    hideTotalTimeDisplay();

    throw error;
  } finally {
    // Clean up state
    currentAbortController = null;
  }
}

// Keep old code as comment reference
/*
// Old inline event handling code has been refactored into independent handler functions
// All events are now dispatched uniformly through dispatchEvent() function
// See event handler functions above:
// - handleParallelGroupStart, handleParallelTaskStart, etc.
// - handleReasoningParallel, handleReasoningNormal
// - handleToolCallParallel, handleToolCallNormal
// - handleToolOutputParallel, handleToolOutputNormal
// - handleDeltaParallel, handleDeltaNormal
// - handleToolLog, handleRunItem
// - handleExcelAgentParallel, handleExcelAgentNormal
// - handleDone
*/

function renderAnalysis(data) {
    if (document.querySelector(`#analysis-${data.id}`)) {
        return; // If already rendered, return directly
    }
    const analysisElement = document.createElement('div');
    analysisElement.id = `analysis-${data.id}`;
    analysisElement.textContent = data.content;
    document.getElementById('analysis-container').appendChild(analysisElement);
}

/**
 * Add final answer bubble
 */
function addFinalAnswerBubble(content) {
  const messagesContainer = document.getElementById('chat-messages');
  if (!messagesContainer) return;

  const messageDiv = document.createElement('div');
  messageDiv.className = 'message assistant final-answer';

  const time = new Date().toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' });
  
  // Render markdown and wrap in a container with custom class for styling control
  const markdownHtml = renderMarkdown(content);
  
  // Create a temporary container to parse and modify the HTML
  const tempDiv = document.createElement('div');
  tempDiv.innerHTML = markdownHtml;
  
  // Remove inline max-width style from images to allow CSS control
  const images = tempDiv.querySelectorAll('img');
  images.forEach(img => {
    img.style.maxWidth = '';
  });
  
  const htmlContent = `<div class="final-answer-markdown">${tempDiv.innerHTML}</div>`;

  // Build timestamp HTML, show if total time exists
  let timestampHtml = `<div class="message-time">${time}`;
  if (totalElapsedTime !== null && totalElapsedTime !== undefined) {
    timestampHtml += ` <span style="margin-left: 8px; color: var(--success); font-weight: 600;">⏱️ ${totalElapsedTime.toFixed(2)}s</span>`;
  }
  timestampHtml += `</div>`;

  // Store original content in element data attribute (for copying)
  messageDiv.dataset.originalContent = content;

  messageDiv.innerHTML = `
    <div class="message-content">
      <div class="message-bubble">
        <div class="message-text">${htmlContent}</div>
      </div>
      <div class="message-footer">
        <button class="copy-btn" onclick="copyMessageContent(this)" title="${t('copy_content')}">📋</button>
        ${timestampHtml}
      </div>
    </div>
  `;

  messagesContainer.appendChild(messageDiv);
  messagesContainer.scrollTop = messagesContainer.scrollHeight;

  chatMessages.push({ role: 'assistant', content, time });

  // Reset total time
  totalElapsedTime = null;

  // Update clear history button state
  updateClearButtonState();

  // Save chat history to localStorage
  saveChatHistory();
}

/**
 * 处理聊天中的文件链接点击事件（非 HTML 文件）
 */
window.handleChatFileLinkClick = function(event, href, fileName) {
  event.preventDefault();
  
  // 移除 file:// 协议头，获取实际路径
  let filePath = href.replace(/^file:\/\//, '');
  
  // 处理 Windows 路径（file:///C:/... -> C:/...）
  if (filePath.match(/^\/[A-Z]:\//)) {
    filePath = filePath.substring(1);
  }
  
  console.log('Chat file link clicked:', filePath);
  
  // 在聊天侧边栏预览文件
  openChatFileSidebar(filePath, fileName);
};

/**
 * 打开聊天文件预览侧边栏
 */
function openChatFileSidebar(filePath, fileName) {
  // 检查或创建侧边栏
  let sidebar = document.getElementById('chat-file-sidebar');
  let overlay = document.getElementById('chat-sidebar-overlay');
  
  if (!sidebar) {
    // 创建侧边栏和遮罩
    const sidebarHtml = `
      <div id="chat-sidebar-overlay" class="sidebar-overlay" style="display: none;" onclick="closeChatFileSidebar()"></div>
      <div id="chat-file-sidebar" class="file-content-sidebar" style="display: none;">
        <div class="sidebar-header">
          <h3 class="sidebar-title" id="chat-sidebar-filename">文件预览</h3>
          <div class="sidebar-header-actions">
            <!-- Markdown toggle button will be inserted here if needed -->
            <button class="btn btn-small btn-secondary" id="chat-sidebar-download" onclick="downloadChatSidebarFile()" title="Download" style="display: none;">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                <polyline points="7 10 12 15 17 10"></polyline>
                <line x1="12" y1="15" x2="12" y2="3"></line>
              </svg>
            </button>
            <button class="sidebar-close" onclick="closeChatFileSidebar()">&times;</button>
          </div>
        </div>
        <div class="sidebar-body">
          <div id="chat-sidebar-loading" style="text-align: center; padding: 48px;">
            <div class="spinner spinner-large" style="margin: 0 auto;"></div>
            <p style="margin-top: 16px; color: var(--gray-7);">加载文件内容...</p>
          </div>
          <div id="chat-sidebar-display" style="display: none;">
            <!-- File content will be displayed here -->
          </div>
        </div>
      </div>
    `;
    document.body.insertAdjacentHTML('beforeend', sidebarHtml);
    
    sidebar = document.getElementById('chat-file-sidebar');
    overlay = document.getElementById('chat-sidebar-overlay');
  }
  
  const titleElement = document.getElementById('chat-sidebar-filename');
  const loadingElement = document.getElementById('chat-sidebar-loading');
  const displayElement = document.getElementById('chat-sidebar-display');
  
  // 设置文件名
  if (titleElement) {
    titleElement.textContent = fileName;
  }
  
  // 显示侧边栏和遮罩
  if (sidebar) {
    sidebar.style.display = 'flex';
    setTimeout(() => sidebar.classList.add('show'), 10);
  }
  if (overlay) {
    overlay.style.display = 'block';
    setTimeout(() => overlay.classList.add('show'), 10);
  }
  
  // 重置内容
  if (loadingElement) loadingElement.style.display = 'block';
  if (displayElement) {
    displayElement.style.display = 'none';
    displayElement.innerHTML = '';
  }
  
  // 加载文件内容
  loadChatFileContent(filePath, fileName);
}

/**
 * 关闭聊天文件预览侧边栏
 */
window.closeChatFileSidebar = function() {
  const sidebar = document.getElementById('chat-file-sidebar');
  const overlay = document.getElementById('chat-sidebar-overlay');
  const downloadBtn = document.getElementById('chat-sidebar-download');
  
  if (sidebar) {
    // 清理 Markdown 切换按钮（使用共享函数）
    cleanupMarkdownToggleButton('#chat-file-sidebar');
    
    sidebar.classList.remove('show');
    setTimeout(() => sidebar.style.display = 'none', 300);
  }
  if (overlay) {
    overlay.classList.remove('show');
    setTimeout(() => overlay.style.display = 'none', 300);
  }
  
  // 隐藏下载按钮并清理存储的文件数据
  if (downloadBtn) {
    downloadBtn.style.display = 'none';
  }
  window._chatSidebarFileBlob = null;
  window._chatSidebarFileName = null;
};

/**
 * 下载聊天侧边栏中的当前文件
 */
window.downloadChatSidebarFile = function() {
  if (!window._chatSidebarFileBlob || !window._chatSidebarFileName) {
    console.error('No file data available for download');
    return;
  }
  
  try {
    // 创建下载链接
    const url = URL.createObjectURL(window._chatSidebarFileBlob);
    const a = document.createElement('a');
    a.href = url;
    a.download = window._chatSidebarFileName;
    document.body.appendChild(a);
    a.click();
    
    // 清理
    setTimeout(() => {
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }, 100);
  } catch (error) {
    console.error('Download failed:', error);
  }
};

/**
 * 加载聊天文件内容（通过本地文件代理）
 */
async function loadChatFileContent(filePath, fileName) {
  const loadingElement = document.getElementById('chat-sidebar-loading');
  const displayElement = document.getElementById('chat-sidebar-display');
  const downloadBtn = document.getElementById('chat-sidebar-download');
  
  try {
    // 使用本地文件代理 API
    const response = await fetch(`${API_BASE}/api/local-file-proxy?path=${encodeURIComponent(filePath)}`);
    
    if (!response.ok) {
      throw new Error(`加载失败: ${response.statusText}`);
    }
    
    const blob = await response.blob();
    
    // 统一从 filePath 提取扩展名（更可靠）
    const fileExt = filePath.split('.').pop().toLowerCase();
    
    // 存储当前文件的 blob 和文件名，供下载使用
    window._chatSidebarFileBlob = blob;
    window._chatSidebarFileName = fileName;
    
    // 显示下载按钮
    if (downloadBtn) {
      downloadBtn.style.display = 'flex';
    }
    
    if (loadingElement) loadingElement.style.display = 'none';
    if (displayElement) displayElement.style.display = 'block';
    
    // 根据文件扩展名选择渲染器
    if (['png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp', 'svg'].includes(fileExt)) {
      // 图片渲染器
      renderImageInSidebar(blob, displayElement, fileName);
    } else if (fileExt === 'pdf') {
      // PDF 渲染器
      await renderPdfInSidebar(blob, displayElement);
    } else if (['xlsx', 'xls'].includes(fileExt)) {
      // Excel 渲染器
      await renderExcelInSidebar(blob, displayElement);
    } else if (fileExt === 'csv') {
      // CSV 渲染器
      const text = await blob.text();
      renderCSVInSidebar(text, displayElement);
    } else if (fileExt === 'md') {
      // Markdown 渲染器
      const text = await blob.text();
      renderMarkdownWithToggle(displayElement, text, fileName, '#chat-file-sidebar');
    } else if (['txt', 'json', 'xml', 'log', 'py', 'js', 'html', 'css', 'yaml', 'yml', 'java', 'cpp', 'c', 'h', 'go', 'rs', 'ts', 'tsx', 'jsx', 'sh', 'bash'].includes(fileExt)) {
      // 纯文本渲染器
      const text = await blob.text();
      renderTextInSidebar(text, displayElement);
    } else {
      // 未知类型，尝试作为文本渲染
      await renderUnknownFileInSidebar(blob, displayElement, fileExt);
    }
  } catch (error) {
    console.error('Load file content error:', error);
    if (loadingElement) loadingElement.style.display = 'none';
    if (displayElement) {
      displayElement.style.display = 'block';
      displayElement.innerHTML = `
        <div class="file-type-notice">
          <div class="icon">❌</div>
          <h3>加载失败</h3>
          <p style="margin-top: var(--spacing-md); color: var(--error);">${error.message}</p>
        </div>
      `;
    }
  }
}

/**
 * 在侧边栏渲染 Excel（使用 utils.js 中的通用函数）
 */
async function renderExcelInSidebar(blob, displayElement) {
  if (typeof XLSX !== 'undefined') {
    await window.renderExcelInContainer(blob, displayElement, {
      maxRows: 3000,
      showSheetSelector: true
    });
  } else {
    displayElement.innerHTML = `
      <div class="file-type-notice">
        <div class="icon">📊</div>
        <h3>Excel 文件</h3>
        <p>XLSX.js 库未加载，无法预览</p>
      </div>
    `;
  }
}

/**
 * 在侧边栏渲染图片
 */
function renderImageInSidebar(blob, displayElement, fileName) {
  const imageUrl = URL.createObjectURL(blob);
  displayElement.innerHTML = `
    <div style="text-align: center; padding: var(--spacing-md);">
      <img src="${imageUrl}" alt="${escapeHtml(fileName)}" style="max-width: 100%; height: auto; border-radius: var(--radius-md);">
    </div>
  `;
}

/**
 * 在侧边栏渲染纯文本
 */
function renderTextInSidebar(text, displayElement) {
  displayElement.innerHTML = `<pre style="background-color: var(--gray-2); padding: var(--spacing-md); border-radius: var(--radius-md); overflow-x: auto; font-size: var(--font-size-sm); line-height: 1.6; white-space: pre-wrap; word-wrap: break-word;">${escapeHtml(text)}</pre>`;
}

/**
 * 在侧边栏渲染 PDF
 */
async function renderPdfInSidebar(blob, displayElement) {
  if (typeof pdfjsLib === 'undefined') {
    displayElement.innerHTML = `
      <div class="file-type-notice">
        <div class="icon">📄</div>
        <h3>PDF 文件</h3>
        <p>PDF.js 库未加载，无法预览</p>
      </div>
    `;
    return;
  }

  try {
    const arrayBuffer = await blob.arrayBuffer();
    const loadingTask = pdfjsLib.getDocument({ data: arrayBuffer });
    const pdf = await loadingTask.promise;
    
    displayElement.innerHTML = '<div id="chat-pdf-container" style="overflow-y: auto; max-height: 600px;"></div>';
    const container = displayElement.querySelector('#chat-pdf-container');
    
    for (let pageNum = 1; pageNum <= pdf.numPages; pageNum++) {
      const page = await pdf.getPage(pageNum);
      const scale = 1.5;
      const viewport = page.getViewport({ scale });
      
      const canvas = document.createElement('canvas');
      canvas.style.display = 'block';
      canvas.style.margin = '0 auto var(--spacing-md)';
      canvas.style.border = '1px solid var(--gray-4)';
      container.appendChild(canvas);
      
      const context = canvas.getContext('2d');
      canvas.height = viewport.height;
      canvas.width = viewport.width;
      
      const scaledViewport = page.getViewport({ scale });
      if (context) {
        await page.render({ canvasContext: context, viewport: scaledViewport }).promise;
      }
    }
  } catch (error) {
    console.error('PDF rendering error:', error);
    displayElement.innerHTML = `
      <div class="file-type-notice">
        <div class="icon">❌</div>
        <h3>PDF 解析失败</h3>
        <p style="color: var(--error);">${error.message}</p>
      </div>
    `;
  }
}

/**
 * 在侧边栏渲染未知类型文件（尝试文本 fallback）
 */
async function renderUnknownFileInSidebar(blob, displayElement, fileExt) {
  try {
    const text = await blob.text();
    // 尝试作为文本显示
    renderTextInSidebar(text, displayElement);
  } catch (error) {
    // 无法作为文本读取
    displayElement.innerHTML = `
      <div class="file-type-notice">
        <div class="icon">📦</div>
        <h3>未知文件类型</h3>
        <p>文件类型: ${fileExt ? fileExt.toUpperCase() : '无扩展名'}</p>
        <p style="color: var(--gray-6);">无法预览此文件</p>
      </div>
    `;
  }
}

/**
 * 在侧边栏渲染 CSV 文件
 */
function renderCSVInSidebar(text, displayElement) {
  try {
    // 简单的 CSV 解析（支持引号和逗号）
    const lines = text.split('\n').filter(line => line.trim());
    if (lines.length === 0) {
      displayElement.innerHTML = '<div class="file-type-notice"><div class="icon">📄</div><h3>空文件</h3></div>';
      return;
    }

    // 解析 CSV 行（处理引号内的逗号）
    function parseCSVLine(line) {
      const result = [];
      let current = '';
      let inQuotes = false;
      
      for (let i = 0; i < line.length; i++) {
        const char = line[i];
        
        if (char === '"') {
          inQuotes = !inQuotes;
        } else if (char === ',' && !inQuotes) {
          result.push(current.trim());
          current = '';
        } else {
          current += char;
        }
      }
      result.push(current.trim());
      return result;
    }

    const rows = lines.map(line => parseCSVLine(line));
    const maxCols = Math.max(...rows.map(row => row.length));
    
    // 生成表格 HTML
    let tableHTML = `
      <div style="overflow-x: auto; max-height: 600px; background-color: var(--gray-1);">
        <table style="width: 100%; border-collapse: collapse; font-size: var(--font-size-sm);">
          <thead style="position: sticky; top: 0; background-color: var(--gray-3); z-index: 1;">
            <tr>`;
    
    // 表头
    const headers = rows[0];
    headers.forEach((header, index) => {
      tableHTML += `<th style="padding: var(--spacing-sm) var(--spacing-md); border: 1px solid var(--gray-4); text-align: left; font-weight: 600; white-space: nowrap;">${escapeHtml(header)}</th>`;
    });
    
    tableHTML += `</tr></thead><tbody>`;
    
    // 数据行（最多显示 1000 行）
    const maxRows = Math.min(rows.length, 1000);
    for (let i = 1; i < maxRows; i++) {
      tableHTML += '<tr>';
      for (let j = 0; j < maxCols; j++) {
        const cell = rows[i][j] || '';
        tableHTML += `<td style="padding: var(--spacing-sm) var(--spacing-md); border: 1px solid var(--gray-4); white-space: nowrap;">${escapeHtml(cell)}</td>`;
      }
      tableHTML += '</tr>';
    }
    
    tableHTML += '</tbody></table>';
    
    // 如果行数过多，显示提示
    if (rows.length > 1000) {
      tableHTML += `<div style="padding: var(--spacing-md); text-align: center; color: var(--gray-6); background-color: var(--gray-2);">仅显示前 1000 行，共 ${rows.length} 行</div>`;
    }
    
    tableHTML += '</div>';
    
    displayElement.innerHTML = tableHTML;
  } catch (error) {
    console.error('CSV rendering error:', error);
    displayElement.innerHTML = `<pre style="background-color: var(--gray-2); padding: var(--spacing-md); border-radius: var(--radius-md); overflow-x: auto; font-size: var(--font-size-sm); line-height: 1.6; white-space: pre-wrap; word-wrap: break-word;">${escapeHtml(text)}</pre>`;
  }
}


// Handle file attachment
function handleFileAttachment(files) {
  if (!files || files.length === 0) return;

  for (let file of files) {
    attachedFiles.push(file);
  }

  renderAttachedFiles();
  showToast(`已添加 ${files.length} 个文件`, 'success');
}

// Render attached files
function renderAttachedFiles() {
  const container = document.getElementById('attached-files');
  if (!container) return;

  if (attachedFiles.length === 0) {
    container.innerHTML = '';
    return;
  }

  container.innerHTML = attachedFiles.map((file, index) => `
    <span class="badge badge-primary" style="display: inline-flex; align-items: center; gap: 8px;">
      ${escapeHtml(file.name)}
      <span onclick="removeAttachedFile(${index})" style="cursor: pointer; font-weight: bold;">&times;</span>
    </span>
  `).join('');
}

// Remove attached file
function removeAttachedFile(index) {
  attachedFiles.splice(index, 1);
  renderAttachedFiles();
}

// Clear chat history
async function clearChatHistory() {
  const clearBtn = document.getElementById('clear-history-btn');

  // If button is disabled, do not execute
  if (clearBtn && clearBtn.classList.contains('disabled')) {
    return;
  }

  const confirmed = await confirmDialog(t('clear_history_confirm'));
  if (!confirmed) return;

  chatMessages = [];

  const messagesContainer = document.getElementById('chat-messages');
  if (messagesContainer) {
    messagesContainer.innerHTML = ''; // Clear all messages
  }

  // Clear localStorage
  localStorage.removeItem('chat-history');

  // Switch back to center mode
  switchToCenterMode();

  // Update clear history button state
  updateClearButtonState();

  showToast(t('toast_chat_cleared'), 'success');
}

// Save chat history to localStorage
function saveChatHistory() {
  try {
    const messagesContainer = document.getElementById('chat-messages');
    if (messagesContainer && chatMessages.length > 0) {
      const historyData = {
        messages: chatMessages,
        html: messagesContainer.innerHTML,
        hasMessages: messagesContainer.classList.contains('has-messages'),
        timestamp: Date.now()
      };
      localStorage.setItem('chat-history', JSON.stringify(historyData));
    }
  } catch (error) {
    console.error('Failed to save chat history:', error);
  }
}

// Restore chat history from localStorage
function restoreChatHistory() {
  try {
    const savedHistory = localStorage.getItem('chat-history');
    if (!savedHistory) {
      return;
    }

    const historyData = JSON.parse(savedHistory);
    if (!historyData || !historyData.messages || historyData.messages.length === 0) {
      return;
    }

    // Restore chatMessages array
    chatMessages = historyData.messages;

    // Restore HTML content
    const messagesContainer = document.getElementById('chat-messages');
    if (messagesContainer && historyData.html) {
      messagesContainer.innerHTML = historyData.html;
      
      // Restore has-messages class
      if (historyData.hasMessages) {
        messagesContainer.classList.add('has-messages');
      }

      // Restore input container to bottom mode if there are messages
      if (chatMessages.length > 0) {
        const inputContainer = document.getElementById('chat-input-container');
        if (inputContainer) {
          inputContainer.classList.remove('center-mode');
          inputContainer.classList.add('bottom-mode');
        }
      }

      // Reattach event listeners to copy buttons
      reattachCopyButtonListeners();

      // Update clear button state
      updateClearButtonState();

      // Scroll to bottom after DOM is fully rendered
      requestAnimationFrame(() => {
        setTimeout(() => {
          messagesContainer.scrollTop = messagesContainer.scrollHeight;
          console.log(`[Chat] Restored ${chatMessages.length} messages and scrolled to bottom`);
        }, 100);
      });
    }
  } catch (error) {
    console.error('Failed to restore chat history:', error);
    // If restore fails, clear the corrupted data
    localStorage.removeItem('chat-history');
  }
}

// Reattach event listeners to copy buttons after restoring HTML
function reattachCopyButtonListeners() {
  // Copy buttons use onclick="copyMessageContent(this)" in HTML
  // So they should work automatically when HTML is restored
  // We just need to make sure the global function is available
  // No additional event listener attachment is needed
  console.log('[Chat] Copy buttons restored with HTML');
}

// ========== Collapsible Card Component System ==========

/**
 * Create collapsible card
 * @param {string} title - Card title
 * @param {string} icon - Icon emoji
 * @param {string} bgColor - Background color
 * @returns {HTMLElement} Card element
 */
function createCollapsibleCard(title, icon, bgColor) {
  const messagesContainer = document.getElementById('chat-messages');
  if (!messagesContainer) return null;

  // Remove empty state if exists
  const emptyState = messagesContainer.querySelector('.empty-state');
  if (emptyState) {
    emptyState.remove();
  }

  const card = document.createElement('div');
  card.className = 'execution-card expanded';

  // Record start time
  const startTime = Date.now();
  card.dataset.startTime = startTime;

  card.innerHTML = `
    <div class="execution-card-header" onclick="toggleExecutionCard(this)">
      <div class="execution-card-title">
        <span class="status-icon running">${icon}</span>
        <span>${escapeHtml(title)}</span>
      </div>
      <div class="card-header-flex">
        <span class="execution-time">0.00s</span>
        <span class="toggle-icon">▼</span>
      </div>
    </div>
    <div class="execution-card-body">
      <div class="card-content" style="background: ${bgColor}; border-radius: 4px;">
        <div class="loading-dots">
          <div class="loading-dot"></div>
          <div class="loading-dot"></div>
          <div class="loading-dot"></div>
        </div>
      </div>
    </div>
  `;

  messagesContainer.appendChild(card);
  messagesContainer.scrollTop = messagesContainer.scrollHeight;

  // Start timer
  startCardTimer(card);

  // Update clear history button state
  updateClearButtonState();

  return card;
}

/**
 * Update card content
 * @param {HTMLElement} card - Card element
 * @param {string} content - Content
 * @param {string} type - Type (text/json/code/markdown)
 */
function updateCardContent(card, content, type = 'text') {
  if (!card) return;

  const contentDiv = card.querySelector('.card-content');
  if (!contentDiv) return;

  if (type === 'text') {
    // Plain text display
    contentDiv.innerHTML = `<div style="line-height: 1.6; white-space: pre-wrap;">${escapeHtml(content)}</div>`;
  } else if (type === 'json') {
    // JSON format uses markdown display
    try {
      const jsonObj = typeof content === 'string' ? JSON.parse(content) : content;
      const jsonText = JSON.stringify(jsonObj, null, 2);
      const markdownJson = '```json\n' + jsonText + '\n```';
      const renderedHtml = marked.parse(markdownJson);
      contentDiv.innerHTML = `<div class="markdown-content" style="margin: 0;">${renderedHtml}</div>`;
    } catch (e) {
      // JSON parsing failed, fallback to plain text
      contentDiv.innerHTML = `<pre style="margin: 0; white-space: pre-wrap; font-family: 'Monaco', 'Menlo', 'Consolas', monospace; font-size: 13px; line-height: 1.6;">${escapeHtml(content)}</pre>`;
    }
  } else if (type === 'code') {
    // Code format
    let codeContent = '';
    let language = '';

    // Detect JSON field format {"code": "..."} or {"sql": "..."}
    const codeFieldMatch = content.match(/\{"code"\s*:\s*"/);
    const sqlFieldMatch = content.match(/\{"sql"\s*:\s*"/);

    if (codeFieldMatch || sqlFieldMatch) {
      const fieldName = codeFieldMatch ? 'code' : 'sql';
      language = fieldName;

      // Manual parsing for streaming JSON
      const fieldMatch = content.match(new RegExp(`"${fieldName}"\\s*:\\s*"`));
      const codeStartIndex = fieldMatch ? content.indexOf(fieldMatch[0]) + fieldMatch[0].length : -1;
      codeContent = codeStartIndex > -1 ? content.substring(codeStartIndex) : '';

      // Remove trailing "} or "
      codeContent = codeContent.replace(/"\}$/g, '').replace(/"$/g, '');

      // Handle JSON escape sequences properly
      // Must process in correct order to avoid double-unescaping
      codeContent = codeContent
        .replace(/\\\\/g, '\x00')      // Temporarily replace \\ with placeholder
        .replace(/\\"/g, '"')           // \" -> "
        .replace(/\\n/g, '\n')          // \n -> newline
        .replace(/\\t/g, '\t')          // \t -> tab
        .replace(/\\r/g, '\r')          // \r -> carriage return
        .replace(/\x00/g, '\\');        // Restore \\ -> \
    } else {
      // Markdown code block format ```language\n...\n```
      const mdCodeMatch = content.match(/```(\w+)\n([\s\S]*?)(?:```)?$/);

      if (mdCodeMatch) {
        language = mdCodeMatch[1];
        codeContent = mdCodeMatch[2];
      } else {
        codeContent = content;
        language = 'text';
      }
    }

    // Use Markdown to render code block
    const markdownCode = '```' + language + '\n' + codeContent + '\n```';

    try {
      const renderedHtml = marked.parse(markdownCode);
      contentDiv.innerHTML = `<div class="markdown-content" style="margin: 0;">${renderedHtml}</div>`;
    } catch (e) {
      contentDiv.innerHTML = `<pre style="margin: 0; white-space: pre-wrap; font-family: 'Monaco', 'Menlo', 'Consolas', monospace; font-size: 13px; line-height: 1.6;">${escapeHtml(codeContent)}</pre>`;
    }
  } else if (type === 'markdown') {
    // Markdown rendering
    const renderedHtml = marked.parse(content);
    contentDiv.innerHTML = `<div class="markdown-content">${renderedHtml}</div>`;
  }

  // Auto scroll to bottom
  const messagesContainer = document.getElementById('chat-messages');
  if (messagesContainer) {
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
  }

  // Scroll card content to bottom
  const cardBody = card.querySelector('.execution-card-body');
  if (cardBody) {
    cardBody.scrollTop = cardBody.scrollHeight;
  }
}

/**
 * Start card timer
 * @param {HTMLElement} card - Card element
 */
function startCardTimer(card) {
  if (!card) return;

  const timeElement = card.querySelector('.execution-time');
  const startTime = parseInt(card.dataset.startTime);

  // Update time every 100ms
  const timerId = setInterval(() => {
    // If card is completed, stop timer
    if (card.classList.contains('completed')) {
      clearInterval(timerId);
      return;
    }

    const elapsed = (Date.now() - startTime) / 1000;
    if (timeElement) {
      timeElement.textContent = elapsed.toFixed(2) + 's';
    }
  }, 100);

  // Save timer ID to card
  card.dataset.timerId = timerId;
}

/**
 * Complete card (stop timer, but keep expanded)
 * @param {HTMLElement} card - Card element
 * @param {boolean} collapse - Whether to collapse (default false, keep expanded)
 */
function completeCard(card, collapse = false) {
  if (!card) return;

  // Add completed state class
  card.classList.add('completed');

  // Calculate and display final runtime
  const timeSpan = card.querySelector('.execution-time');
  if (timeSpan && card.dataset.startTime) {
    const startTime = parseInt(card.dataset.startTime);
    const elapsed = (Date.now() - startTime) / 1000;
    timeSpan.textContent = elapsed.toFixed(2) + 's';
  }

  // Clear timer
  if (card.dataset.timerId) {
    clearInterval(parseInt(card.dataset.timerId));
  }

  // Stop icon rotation, add success state
  const statusIcon = card.querySelector('.status-icon');
  if (statusIcon) {
    statusIcon.classList.remove('running');
    statusIcon.classList.add('success');
  }

  // Only collapse when specified (default keep expanded)
  if (collapse) {
    card.classList.remove('expanded');
  }
}

/**
 * Toggle card expand/collapse
 * @param {HTMLElement} header - Card header element
 */
function toggleExecutionCard(header) {
  const card = header.parentElement;
  card.classList.toggle('expanded');
}

// ========== Total Time Timer ==========

/**
 * Start total time timer
 */
function startTotalTimeTimer() {
  // Record start time
  totalTimeStartTime = Date.now();

  // Update time every 100ms
  totalTimeTimer = setInterval(() => {
    const elapsed = (Date.now() - totalTimeStartTime) / 1000;
    // Can update UI to show total time here (if needed)
    console.log(`Total time: ${elapsed.toFixed(2)}s`);
  }, 100);

  console.log('Total time timer started');
}

/**
 * Stop total time timer
 */
function stopTotalTimeTimer() {
  if (totalTimeTimer) {
    clearInterval(totalTimeTimer);
    totalTimeTimer = null;
  }

  if (totalTimeStartTime) {
    const elapsed = (Date.now() - totalTimeStartTime) / 1000;
    totalElapsedTime = elapsed;
    console.log(`Total time: ${elapsed.toFixed(2)}s`);
  }

  totalTimeStartTime = null;
}

/**
 * Hide total time display
 */
function hideTotalTimeDisplay() {
  if (totalTimeTimer) {
    clearInterval(totalTimeTimer);
    totalTimeTimer = null;
  }

  totalTimeStartTime = null;
}

// ========== Copy Functionality ==========

/**
 * Update all copy button titles when language changes
 */
function updateCopyButtonTitles() {
  const copyButtons = document.querySelectorAll('.copy-btn');
  copyButtons.forEach(button => {
    button.title = t('copy_content');
  });
}

/**
 * Copy message content to clipboard
 * @param {HTMLElement} button - Copy button element
 */
function copyMessageContent(button) {
  // Find nearest message element
  const messageDiv = button.closest('.message');
  if (!messageDiv) return;

  // Get stored original content
  const content = messageDiv.dataset.originalContent;
  if (!content) {
    showToast(t('toast_cannot_copy'), 'error');
    return;
  }

  // Use Clipboard API to copy to clipboard
  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(content)
      .then(() => {
        // Success notification
        const originalText = button.textContent;
        button.textContent = '✓';
        button.style.color = 'var(--success)';

        setTimeout(() => {
          button.textContent = originalText;
          button.style.color = '';
        }, 2000);

        showToast(t('toast_copy_success'), 'success');
      })
      .catch(err => {
        console.error('Failed to copy:', err);
        // Fallback solution
        fallbackCopyTextToClipboard(content, button);
      });
  } else {
    // Fallback solution（老浏览器）
    fallbackCopyTextToClipboard(content, button);
  }
}

/**
 * Fallback copy solution (compatible with old browsers)
 */
function fallbackCopyTextToClipboard(text, button) {
  const textArea = document.createElement('textarea');
  textArea.value = text;

  // Avoid scrolling to bottom
  textArea.style.top = '0';
  textArea.style.left = '0';
  textArea.style.position = 'fixed';
  textArea.style.opacity = '0';

  document.body.appendChild(textArea);
  textArea.focus();
  textArea.select();

  try {
    const successful = document.execCommand('copy');
    if (successful) {
      const originalText = button.textContent;
      button.textContent = '✓';
      button.style.color = 'var(--success)';

      setTimeout(() => {
        button.textContent = originalText;
        button.style.color = '';
      }, 2000);

      showToast(t('toast_copy_success'), 'success');
    } else {
      showToast(t('toast_copy_failed'), 'error');
    }
  } catch (err) {
    console.error('Fallback copy failed:', err);
    showToast(t('toast_copy_failed'), 'error');
  }

  document.body.removeChild(textArea);
}

// ========== File Upload Functionality ==========

/**
 * Update file upload button state
 */
function updateFileUploadButtonState() {
  const uploadBtn = document.getElementById('file-upload-btn');
  const agentSelect = document.getElementById('agent-select');
  const kbSelect = document.getElementById('kb-select');

  if (!uploadBtn || !agentSelect || !kbSelect) return;

  const hasAgent = agentSelect.value && agentSelect.value !== '';
  const hasKb = kbSelect.value && kbSelect.value !== '';

  if (hasAgent && hasKb) {
    uploadBtn.classList.remove('disabled');
    uploadBtn.title = t('upload_file_title');
  } else {
    uploadBtn.classList.add('disabled');
    if (!hasAgent && !hasKb) {
      uploadBtn.title = t('please_select_agent_kb_first');
    } else if (!hasAgent) {
      uploadBtn.title = t('toast_select_agent');
    } else {
      uploadBtn.title = t('select_knowledge_base');
    }
  }
}

/**
 * Handle file upload button click
 */
function handleFileUploadClick() {
  const uploadBtn = document.getElementById('file-upload-btn');
  if (uploadBtn && uploadBtn.classList.contains('disabled')) {
    showToast(t('please_select_agent_kb'), 'warning');
    return;
  }
  document.getElementById('file-attachment').click();
}

/**
 * Handle file selection
 */
async function handleFileAttachment(files) {
  if (!files || files.length === 0) return;

  console.log(`[File Upload] Selected ${files.length} file(s)`);

  // Check for duplicate files
  const duplicateFiles = [];
  const newFiles = [];

  for (const file of files) {
    try {
      const response = await fetch(`${API_BASE}/api/files/check-exists/${encodeURIComponent(file.name)}`);
      if (response.ok) {
        const data = await response.json();
        if (data.exists) {
          duplicateFiles.push(file);
        } else {
          newFiles.push(file);
        }
      } else {
        newFiles.push(file);
      }
    } catch (error) {
      console.error(`Error checking file ${file.name}:`, error);
      newFiles.push(file);
    }
  }

  // If duplicate files exist, ask user
  let filesToUpload = [...newFiles];
  if (duplicateFiles.length > 0) {
    const duplicateNames = duplicateFiles.map(f => f.name).join('\n• ');
    const shouldOverwrite = confirm(
      t('file_overwrite_confirm', { files: duplicateNames })
    );

    if (shouldOverwrite) {
      filesToUpload = [...filesToUpload, ...duplicateFiles];
    } else if (newFiles.length === 0) {
      showToast(t('toast_all_files_exist'), 'warning');
      document.getElementById('file-attachment').value = '';
      return;
    }
  }

  if (filesToUpload.length === 0) {
    showToast(t('toast_no_files_to_upload'), 'info');
    document.getElementById('file-attachment').value = '';
    return;
  }

  // Upload all files
  for (const file of filesToUpload) {
    await uploadFileWithProgress(file);
  }

  // Clear file input
  document.getElementById('file-attachment').value = '';
}

/**
 * Upload single file and show progress
 */
async function uploadFileWithProgress(file) {
  try {
    console.log(`[File Upload] Uploading: ${file.name}`);

    // Create FormData
    const formData = new FormData();
    formData.append('file', file);

    // Call upload API (with progress)
    const response = await fetch(`${API_BASE}/api/files/upload-with-progress`, {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error('[File Upload] Upload failed:', errorText);
      throw new Error(`上传失败: ${response.status}`);
    }

    const result = await response.json();
    const taskId = result.task_id;

    console.log(`[File Upload] Task created: ${taskId}`);

    // Create progress display card
    const progressCard = createUploadProgressCard(file.name, taskId);

    // Poll progress
    await pollUploadProgress(taskId, progressCard);

    // Associate file with knowledge base
    console.log(`[File Upload] Associating file "${file.name}" to KB ${currentKbId}`);
    await associateFilesToKb([file.name]);
    console.log(`[File Upload] Association completed`);

    // Wait for DB commit (small delay to ensure consistency)
    await new Promise(resolve => setTimeout(resolve, 500));

    // Refresh KB files and auto-select uploaded file
    if (currentKbId) {
      console.log(`[File Upload] Refreshing KB files for KB ${currentKbId}`);
      await loadKnowledgeBaseFiles(currentKbId);
      
      console.log(`[File Upload] Current knowledgeBaseFiles:`, knowledgeBaseFiles);
      const uploadedFile = knowledgeBaseFiles.find(f => (f.name || f.filename) === file.name);
      
      if (uploadedFile) {
        const uploadedId = uploadedFile.id;
        console.log(`[File Upload] Found uploaded file in KB: ${uploadedFile.name || uploadedFile.filename} (ID: ${uploadedId})`);
        
        const alreadySelected = selectedFilesData.some(f => f.id === uploadedId);
        if (!alreadySelected) {
          selectedFilesData.push({
            id: uploadedId,
            name: uploadedFile.name || uploadedFile.filename
          });
          selectedFileIds = selectedFilesData.map(f => f.id);
          renderSelectedFilesTags();
          updateSendButtonState();
          console.log(`[File Upload] Auto-selected file: ${uploadedFile.name || uploadedFile.filename} (ID: ${uploadedId})`);
          console.log(`[File Upload] selectedFileIds:`, selectedFileIds);
        } else {
          console.log(`[File Upload] File already selected`);
        }
      } else {
        console.warn(`[File Upload] Uploaded file "${file.name}" not found in KB files after refresh`);
        console.warn(`[File Upload] Available files:`, knowledgeBaseFiles.map(f => f.name || f.filename));
      }
    } else {
      console.warn(`[File Upload] No KB selected (currentKbId is null)`);
    }

    console.log(`[File Upload] Upload complete: ${file.name}`);

  } catch (error) {
    console.error(`[File Upload] Error uploading ${file.name}:`, error);
    showToast(`上传 "${file.name}" 失败: ${error.message}`, 'error');
  }
}

/**
 * Create upload progress card
 */
function createUploadProgressCard(filename, taskId) {
  const messagesContainer = document.getElementById('chat-messages');

  const card = document.createElement('div');
  card.className = 'execution-card expanded';
  card.id = `upload-card-${taskId}`;
  card.innerHTML = `
    <div class="execution-card-header" onclick="toggleCard(this)">
      <div class="execution-card-title">
        <span class="status-icon running">⏳</span>
        <span>Uploading file: ${filename}</span>
      </div>
      <div class="card-header-flex">
        <span class="execution-time">${new Date().toLocaleTimeString()}</span>
        <span class="toggle-icon">▼</span>
      </div>
    </div>
    <div class="execution-card-body">
      <div style="padding: 12px; background: #f0f9ff; border-radius: 4px; font-size: 13px;">
        <div style="margin-bottom: 8px;">
          <strong>⏳ Processing file...</strong>
        </div>
        <div class="progress-bar">
          <div class="progress-fill" style="width: 0%;"></div>
        </div>
        <div class="progress-text" style="margin-top: 8px; color: #666; font-size: 12px;">
          Preparing to upload...
        </div>
      </div>
    </div>
  `;

  messagesContainer.appendChild(card);
  messagesContainer.scrollTop = messagesContainer.scrollHeight;

  // Switch to bottom mode
  switchToBottomMode();
  updateClearButtonState();

  return card;
}

/**
 * Poll upload progress
 */
async function pollUploadProgress(taskId, progressCard) {
  const maxAttempts = 300; // Max 5 minutes (300 * 1000ms)
  let attempts = 0;

  while (attempts < maxAttempts) {
    try {
      const response = await fetch(`${API_BASE}/api/files/upload-progress/${taskId}`);

      if (!response.ok) {
        throw new Error('Failed to get progress');
      }

      const progress = await response.json();

      // Update progress display
      updateProgressCard(progressCard, progress);

      // Check if completed
      if (progress.status === 'completed') {
        console.log(`[File Upload] Task ${taskId} completed`);
        return;
      }

      if (progress.status === 'failed') {
        throw new Error(progress.error || t('error_upload_failed'));
      }

      // Wait 1 second before continuing to poll
      await new Promise(resolve => setTimeout(resolve, 1000));
      attempts++;

    } catch (error) {
      console.error(`[File Upload] Polling error:`, error);
      updateProgressCardError(progressCard, error.message);
      throw error;
    }
  }

  throw new Error(t('error_upload_timeout'));
}

/**
 * Update progress card
 */
function updateProgressCard(card, progress) {
  const progressFill = card.querySelector('.progress-fill');
  const progressText = card.querySelector('.progress-text');
  const statusIcon = card.querySelector('.status-icon');
  const cardBody = card.querySelector('.execution-card-body');

  if (progressFill) {
    progressFill.style.width = `${progress.progress || 0}%`;
  }

  if (progressText) {
    progressText.textContent = progress.message || t('processing');
  }

  // Update status icon and title text
  if (progress.status === 'completed') {
    statusIcon.textContent = '✓';
    statusIcon.classList.remove('running');
    statusIcon.classList.add('success');
    card.classList.add('completed');

    // 更��卡片内容，将"正在处理文件"改为"处理完成"
    if (cardBody) {
      cardBody.innerHTML = `
        <div style="padding: 12px; background: #f0f9ff; border-radius: 4px; font-size: 13px;">
          <div style="margin-bottom: 8px;">
            <strong style="color: #10b981;">✓ File processing complete</strong>
          </div>
          <div class="progress-bar">
            <div class="progress-fill" style="width: 100%;"></div>
          </div>
          <div class="progress-text" style="margin-top: 8px; color: #666; font-size: 12px;">
            ${progress.message || t('upload_complete')}
          </div>
        </div>
      `;
    }
  }
}

/**
 * Update progress card错误状态
 */
function updateProgressCardError(card, errorMessage) {
  const statusIcon = card.querySelector('.status-icon');
  const cardBody = card.querySelector('.execution-card-body');

  statusIcon.textContent = '✗';
  statusIcon.classList.remove('running');
  statusIcon.style.color = '#f56565';

  if (cardBody) {
    cardBody.innerHTML = `
      <div style="padding: 12px; background: #fee; border-radius: 4px; font-size: 13px;">
        <strong style="color: #f56565;">✗ Upload failed</strong>
        <div style="margin-top: 8px; color: #666;">
          ${errorMessage}
        </div>
      </div>
    `;
  }
}

/**
 * Render file tags
 */
function renderFileTags() {
  const container = document.getElementById('attached-files');
  if (!container) return;

  container.innerHTML = '';

  if (attachedFiles.length === 0) {
    container.style.display = 'none';
    return;
  }

  container.style.display = 'flex';

  attachedFiles.forEach((file, index) => {
    const tag = document.createElement('div');
    tag.className = 'file-tag';
    tag.innerHTML = `
      <span class="file-tag-icon">📄</span>
      <span class="file-tag-name" title="${file.filename}">${file.filename}</span>
      <span class="file-tag-remove" onclick="removeFileTag(${index})" title="${t('remove')}">×</span>
    `;
    container.appendChild(tag);
  });
}

/**
 * Remove file tag
 */
function removeFileTag(index) {
  attachedFiles.splice(index, 1);
  renderFileTags();
  showToast(t('toast_file_removed_simple'), 'success');
}

/**
 * Associate file with knowledge base
 */
async function associateFilesToKb(filenames) {
  try {
    // Use currently selected knowledge base
    if (!currentKbId) {
      throw new Error(t('error_kb_not_selected'));
    }

    const targetKbId = currentKbId;

    // Get knowledge base info (including config)
    const kbResponse = await fetch(`${API_BASE}/api/knowledge/${targetKbId}`);
    if (!kbResponse.ok) {
      throw new Error(t('error_get_kb_info_failed'));
    }
    const kbData = await kbResponse.json();
    const targetKbName = kbData.name || t('current_kb');

    console.log(`[File Upload] Associating files to KB: ${targetKbName} (ID: ${targetKbId})`);

    // Get config from knowledge base details
    const config = kbData.configuration || {
      tools: {},
      selectedFiles: [],
      selectedQAFiles: [],
      dbConnections: []
    };
    const existingFiles = config.selectedFiles || [];

    // Merge files (deduplicate)
    const allFiles = [...new Set([...existingFiles, ...filenames])];

    // Update knowledge base config
    const updateResponse = await fetch(`${API_BASE}/api/knowledge/${targetKbId}/configuration`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        configuration: {
          ...config,
          selectedFiles: allFiles
        }
      })
    });

    if (!updateResponse.ok) {
      const errorText = await updateResponse.text();
      console.error('[File Upload] Update config failed:', errorText);
      throw new Error(t('error_update_kb_config_failed'));
    }

    console.log(`[File Upload] Files successfully associated to KB: ${targetKbName}`);

    // Reload knowledge base file list to get new file source_config ID
    await loadKnowledgeBaseFiles(targetKbId);

    showToast(t('toast_files_associated', { name: targetKbName }), 'success');

  } catch (error) {
    console.error('[File Upload] Error associating files to KB:', error);
    showToast(t('toast_associate_failed', { error: error.message }), 'error');
  }
}
