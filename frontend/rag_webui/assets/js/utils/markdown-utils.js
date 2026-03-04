// Markdown Utilities - Shared Markdown Rendering and Toggle Functions

/**
 * Render Markdown with code/render view toggle
 * @param {HTMLElement} displayElement - The container element to render markdown in
 * @param {string} markdownText - The markdown text content
 * @param {string} fileName - The file name
 * @param {string} sidebarSelector - The CSS selector for the sidebar container (e.g., '#chat-file-sidebar' or '#file-content-sidebar')
 */
function renderMarkdownWithToggle(displayElement, markdownText, fileName, sidebarSelector) {
  const containerId = 'md-preview-container-' + Date.now();
  const codeViewId = 'md-code-view-' + Date.now();
  const renderViewId = 'md-render-view-' + Date.now();
  
  // Insert toggle button in sidebar header
  const sidebar = document.querySelector(sidebarSelector);
  if (sidebar) {
    const headerActions = sidebar.querySelector('.sidebar-header-actions');
    if (headerActions) {
      // Check if toggle button already exists
      let toggleBtn = headerActions.querySelector('.md-view-toggle-switch');
      if (!toggleBtn) {
        toggleBtn = document.createElement('button');
        toggleBtn.className = 'md-view-toggle-switch';
        toggleBtn.dataset.containerId = containerId;
        toggleBtn.dataset.currentView = 'render';
        toggleBtn.innerHTML = '&lt;/&gt;';
        toggleBtn.title = '切换到代码视图';
        toggleBtn.onclick = function() {
          toggleMarkdownViewSwitch(this);
        };
        // Insert before close button
        const closeBtn = headerActions.querySelector('.sidebar-close');
        if (closeBtn) {
          headerActions.insertBefore(toggleBtn, closeBtn);
        } else {
          headerActions.appendChild(toggleBtn);
        }
      } else {
        // Update existing button's containerId
        toggleBtn.dataset.containerId = containerId;
        toggleBtn.dataset.currentView = 'render';
        toggleBtn.innerHTML = '&lt;/&gt;';
        toggleBtn.title = '切换到代码视图';
      }
    }
  }
  
  // Render content (no longer need top button bar)
  displayElement.innerHTML = `
    <div id="${containerId}" style="display: flex; flex-direction: column; height: 100%;">
      <!-- Render view -->
      <div id="${renderViewId}" class="md-view-content" style="display: block;">
        <div class="markdown-content" style="padding: var(--spacing-md); width: 100%; box-sizing: border-box;">
          ${renderMarkdown(markdownText)}
        </div>
      </div>
      
      <!-- Code view -->
      <div id="${codeViewId}" class="md-view-content" style="display: none;">
        <pre style="background-color: var(--gray-2); padding: var(--spacing-md); margin: 0; overflow-x: auto; font-size: var(--font-size-sm); line-height: 1.6; white-space: pre-wrap; word-wrap: break-word; height: 100%;">${escapeHtml(markdownText)}</pre>
      </div>
    </div>
  `;
}

/**
 * Toggle Markdown view switch
 * @param {HTMLButtonElement} button - The toggle button element
 */
function toggleMarkdownViewSwitch(button) {
  const containerId = button.dataset.containerId;
  const currentView = button.dataset.currentView;
  const container = document.getElementById(containerId);
  
  if (!container) return;
  
  // Find view elements
  const renderView = container.querySelector('[id^="md-render-view-"]');
  const codeView = container.querySelector('[id^="md-code-view-"]');
  
  if (!renderView || !codeView) return;
  
  // Toggle view
  if (currentView === 'render') {
    // Switch to code view
    renderView.style.display = 'none';
    codeView.style.display = 'block';
    button.dataset.currentView = 'code';
    button.innerHTML = '👁';
    button.title = '切换到渲染视图';
  } else {
    // Switch to render view
    renderView.style.display = 'block';
    codeView.style.display = 'none';
    button.dataset.currentView = 'render';
    button.innerHTML = '&lt;/&gt;';
    button.title = '切换到代码视图';
  }
}

/**
 * Clean up Markdown toggle button from sidebar header
 * @param {string} sidebarSelector - The CSS selector for the sidebar container
 */
function cleanupMarkdownToggleButton(sidebarSelector) {
  const sidebar = document.querySelector(sidebarSelector);
  if (sidebar) {
    const headerActions = sidebar.querySelector('.sidebar-header-actions');
    if (headerActions) {
      const toggleBtn = headerActions.querySelector('.md-view-toggle-switch');
      if (toggleBtn) {
        toggleBtn.remove();
      }
    }
  }
}

// Expose functions to window for global access
window.renderMarkdownWithToggle = renderMarkdownWithToggle;
window.toggleMarkdownViewSwitch = toggleMarkdownViewSwitch;
window.cleanupMarkdownToggleButton = cleanupMarkdownToggleButton;
