// Utility Functions

// Show toast notification
function showToast(message, type = 'info') {
  const container = document.getElementById('toast-container');
  if (!container) return;

  const toast = document.createElement('div');
  toast.className = `toast toast-${type}`;

  const icon = type === 'success' ? '✓' : type === 'error' ? '✕' : type === 'warning' ? '⚠' : 'ℹ';

  toast.innerHTML = `
    <span class="toast-icon">${icon}</span>
    <span class="toast-message">${escapeHtml(message)}</span>
  `;

  container.appendChild(toast);

  // Auto remove after 3 seconds
  setTimeout(() => {
    toast.style.animation = 'slideOutRight 0.3s ease';
    setTimeout(() => toast.remove(), 300);
  }, 3000);
}

// Escape HTML to prevent XSS
function escapeHtml(text) {
  // Handle null, undefined, or non-string values
  if (text == null) return '';
  if (typeof text !== 'string') text = String(text);

  const map = {
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#039;'
  };
  return text.replace(/[&<>"']/g, m => map[m]);
}

// Format file size
function formatFileSize(bytes) {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Format date
function formatDate(dateString) {
  const date = new Date(dateString);
  const now = new Date();
  const diff = now - date;
  const seconds = Math.floor(diff / 1000);
  const minutes = Math.floor(seconds / 60);
  const hours = Math.floor(minutes / 60);
  const days = Math.floor(hours / 24);

  if (days > 7) {
    const locale = i18n.getLang() === 'zh' ? 'zh-CN' : 'en-US';
    return date.toLocaleDateString(locale, {
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit'
    });
  } else if (days > 0) {
    return `${days}${t('days_ago')}`;
  } else if (hours > 0) {
    return `${hours}${t('hours_ago')}`;
  } else if (minutes > 0) {
    return `${minutes}${t('minutes_ago')}`;
  } else {
    return t('just_now');
  }
}

// Show loading spinner
function showLoading(element) {
  const spinner = document.createElement('div');
  spinner.className = 'spinner spinner-large';
  spinner.id = 'loading-spinner';
  element.appendChild(spinner);
}

// Hide loading spinner
function hideLoading() {
  const spinner = document.getElementById('loading-spinner');
  if (spinner) spinner.remove();
}

// Show modal
function showModal(modalId) {
  const modal = document.getElementById(modalId);
  if (modal) {
    modal.style.display = 'flex';
    document.body.style.overflow = 'hidden';
  }
}

// Hide modal
function hideModal(modalId, onHideCallback) {
  const modal = document.getElementById(modalId);
  if (modal) {
    modal.style.display = 'none';

    // Only restore body overflow if no other modals are visible
    setTimeout(() => {
      const visibleModals = document.querySelectorAll('.modal-overlay[style*="display: flex"], .modal-overlay[style*="display: block"]');
      const hasVisibleModal = Array.from(visibleModals).some(m => {
        const display = window.getComputedStyle(m).display;
        return display !== 'none';
      });

      if (!hasVisibleModal) {
        document.body.style.overflow = '';
      }
    }, 0);
  }

  // Execute callback if provided
  if (typeof onHideCallback === 'function') {
    onHideCallback(modalId);
  }
}

// Debounce function
function debounce(func, wait) {
  let timeout;
  return function executedFunction(...args) {
    const later = () => {
      clearTimeout(timeout);
      func(...args);
    };
    clearTimeout(timeout);
    timeout = setTimeout(later, wait);
  };
}

// Throttle function
function throttle(func, limit) {
  let inThrottle;
  return function(...args) {
    if (!inThrottle) {
      func.apply(this, args);
      inThrottle = true;
      setTimeout(() => inThrottle = false, limit);
    }
  };
}

// Copy to clipboard
async function copyToClipboard(text) {
  try {
    await navigator.clipboard.writeText(text);
    showToast(t('toast_copy_success'), 'success');
  } catch (err) {
    // Fallback for older browsers
    const textarea = document.createElement('textarea');
    textarea.value = text;
    textarea.style.position = 'fixed';
    textarea.style.opacity = '0';
    document.body.appendChild(textarea);
    textarea.select();
    try {
      document.execCommand('copy');
      showToast(t('toast_copy_success'), 'success');
    } catch (err) {
      showToast(t('toast_copy_failed'), 'error');
    }
    document.body.removeChild(textarea);
  }
}

// Confirm dialog
function confirmDialog(message) {
  return new Promise((resolve) => {
    const confirmed = window.confirm(message);
    resolve(confirmed);
  });
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

// ========== Markdown Rendering Functions ==========

// Markdown renderer (uses marked.js if available)
function renderMarkdown(text) {
  if (!text) return '';

  // Use marked.js if available
  if (typeof marked !== 'undefined') {
    try {
      return `<div class="markdown-content">${marked.parse(text)}</div>`;
    } catch (error) {
      console.error('Markdown parsing error:', error);
    }
  }

  // Fallback: Simple implementation
  let html = text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');

  html = html.replace(/^### (.*$)/gim, '<h3>$1</h3>');
  html = html.replace(/^## (.*$)/gim, '<h2>$1</h2>');
  html = html.replace(/^# (.*$)/gim, '<h1>$1</h1>');
  html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
  html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');
  html = html.replace(/```(.*?)```/gs, '<pre><code>$1</code></pre>');
  html = html.replace(/`(.*?)`/g, '<code>$1</code>');
  html = html.replace(/\n/g, '<br>');

  return `<div class="markdown-content">${html}</div>`;
}

// Setup Markdown renderer (can be enhanced with marked.js)
function setupMarkdownRenderer() {
  // Check if marked.js is available
  if (typeof marked !== 'undefined') {
    // Configure marked if available
    marked.setOptions({
      breaks: true,        // 支持换行
      gfm: true,          // GitHub Flavored Markdown
      headerIds: true,    // 为标题添加 ID
      mangle: false,      // 不混淆邮箱地址
      sanitize: false,    // 允许 HTML（谨慎使用）
      smartLists: true,   // 智能列表
      smartypants: false, // 不转换引号
      xhtml: false        // 不使用 XHTML 标签
    });
    
    // 自定义渲染器（可选）
    const renderer = new marked.Renderer();
    
    // 自定义链接渲染：增强文件链接样式
    renderer.link = function(href, title, text) {
      const isExternal = href.startsWith('http://') || href.startsWith('https://');
      
      // 检测文件扩展名
      const fileExtMatch = href.match(/\.([a-z0-9]+)(\?|$)/i);
      const fileExt = fileExtMatch ? fileExtMatch[1].toLowerCase() : '';
      
      // 图片文件扩展名（这些将以图片形式显示而非链接）
      const imageExts = ['png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp', 'svg'];
      
      // 文件链接（排除图片）
      const isFileLink = (href.startsWith('file://') || href.startsWith('/')) && 
                         fileExt && !imageExts.includes(fileExt);
      
      const titleAttr = title ? ` title="${title}"` : '';
      
      // 文件链接：添加特殊样式和点击处理
      if (isFileLink) {
        const fileName = text || href.split('/').pop().split('?')[0];
        // HTML 文件：通过代理在新标签页打开
        if (fileExt === 'html' || fileExt === 'htm') {
          // 移除 file:// 前缀获取本地路径
          const localPath = href.startsWith('file://') ? href.substring(7) : href;
          const proxyUrl = `/api/local-file-proxy?path=${encodeURIComponent(localPath)}`;
          return `<a href="${proxyUrl}" class="file-link file-link-html"${titleAttr} target="_blank" rel="noopener noreferrer">🌐 ${text}</a>`;
        }
        // 其他文件：在侧边栏预览
        return `<a href="${href}" class="file-link file-link-${fileExt}"${titleAttr} onclick="handleChatFileLinkClick(event, '${href.replace(/'/g, "\\'")}', '${fileName.replace(/'/g, "\\'")}')" title="点击预览">📄 ${text}</a>`;
      }
      
      // 外部链接：在新标签页打开
      if (isExternal) {
        return `<a href="${href}"${titleAttr} target="_blank" rel="noopener noreferrer">${text}</a>`;
      }
      
      // 普通链接
      return `<a href="${href}"${titleAttr}>${text}</a>`;
    };
    
    // 自定义图片渲染：处理本地文件路径
    renderer.image = function(href, title, text) {
      const titleAttr = title ? ` title="${title}"` : '';
      const altAttr = text ? ` alt="${text}"` : '';
      
      // 检查是否是本地文件路径（绝对路径）
      const isLocalFile = href.startsWith('/') || href.startsWith('file://');
      
      let imageSrc = href;
      let originalPath = href;
      
      // 如果是本地文件，使用代理接口
      if (isLocalFile) {
        // 移除 file:// 协议头
        let localPath = href.replace(/^file:\/\//, '');
        originalPath = localPath;
        // 对路径进行 URL 编码
        imageSrc = `${API_BASE}/api/local-file-proxy?path=${encodeURIComponent(localPath)}`;
      }
      
      // 返回图片标签，包含加载失败处理和预览功能
      return `<img src="${imageSrc}"${altAttr}${titleAttr} 
        class="markdown-image" 
        style="max-width: 100%; height: auto;" loading="lazy"
        onclick="previewImageLightbox('${imageSrc.replace(/'/g, "\\'")}', '${originalPath.replace(/'/g, "\\'")}', '${(text || '').replace(/'/g, "\\'")}')"
        onerror="this.style.display='none'; this.insertAdjacentHTML('afterend', '<div class=image-load-error>🖼️ <span>图片加载失败</span><br><small>${href}</small></div>')">`;
    };
    
    marked.use({ renderer });
  }
}

// ========== Image Lightbox Functions ==========

/**
 * 图片灯箱预览（支持缩放、拖动、下载）
 */
window.previewImageLightbox = function(src, originalPath, alt) {
  const overlay = document.createElement('div');
  overlay.className = 'image-lightbox-overlay';
  overlay.innerHTML = `
    <div class="image-lightbox-container">
      <button class="image-lightbox-close" onclick="this.closest('.image-lightbox-overlay').remove()" title="关闭 (ESC)">✕</button>
      <div class="image-lightbox-content" id="lightbox-content">
        <img src="${src}" alt="${escapeHtml(alt || '')}" class="image-lightbox-img" id="lightbox-img" draggable="false">
      </div>
      <div class="image-lightbox-controls">
        <button class="lightbox-btn" onclick="zoomLightboxImage(-0.2)" title="缩小">🔍−</button>
        <button class="lightbox-btn" onclick="resetLightboxImage()" title="重置">↺</button>
        <button class="lightbox-btn" onclick="zoomLightboxImage(0.2)" title="放大">🔍+</button>
        <button class="lightbox-btn" onclick="copyLightboxImage('${src.replace(/'/g, "\\'")}', '${originalPath.replace(/'/g, "\\'")}', '${(alt || '').replace(/'/g, "\\'")}')" title="复制图片">📋</button>
      </div>
    </div>
  `;
  
  // 点击背景关闭
  overlay.onclick = function(e) {
    if (e.target === overlay || e.target.classList.contains('image-lightbox-content')) {
      overlay.remove();
    }
  };
  
  // ESC 键关闭
  const escHandler = function(e) {
    if (e.key === 'Escape') {
      overlay.remove();
      document.removeEventListener('keydown', escHandler);
    }
  };
  document.addEventListener('keydown', escHandler);
  
  document.body.appendChild(overlay);
  
  // 初始化拖动和缩放
  initLightboxDrag();
};

/**
 * 初始化图片灯箱拖动功能
 */
let lightboxState = {
  scale: 1,
  translateX: 0,
  translateY: 0,
  isDragging: false,
  startX: 0,
  startY: 0
};

function initLightboxDrag() {
  const img = document.getElementById('lightbox-img');
  const content = document.getElementById('lightbox-content');
  
  if (!img || !content) return;
  
  // 重置状态
  lightboxState = {
    scale: 1,
    translateX: 0,
    translateY: 0,
    isDragging: false,
    startX: 0,
    startY: 0
  };
  
  updateLightboxTransform();
  
  // 鼠标拖动
  img.addEventListener('mousedown', function(e) {
    if (e.button !== 0) return; // 仅左键
    lightboxState.isDragging = true;
    lightboxState.startX = e.clientX - lightboxState.translateX;
    lightboxState.startY = e.clientY - lightboxState.translateY;
    img.style.cursor = 'grabbing';
    e.preventDefault();
  });
  
  document.addEventListener('mousemove', function(e) {
    if (!lightboxState.isDragging) return;
    lightboxState.translateX = e.clientX - lightboxState.startX;
    lightboxState.translateY = e.clientY - lightboxState.startY;
    updateLightboxTransform();
  });
  
  document.addEventListener('mouseup', function() {
    if (lightboxState.isDragging) {
      lightboxState.isDragging = false;
      img.style.cursor = 'grab';
    }
  });
  
  // 触摸拖动
  img.addEventListener('touchstart', function(e) {
    if (e.touches.length === 1) {
      lightboxState.isDragging = true;
      lightboxState.startX = e.touches[0].clientX - lightboxState.translateX;
      lightboxState.startY = e.touches[0].clientY - lightboxState.translateY;
      e.preventDefault();
    }
  });
  
  img.addEventListener('touchmove', function(e) {
    if (!lightboxState.isDragging || e.touches.length !== 1) return;
    lightboxState.translateX = e.touches[0].clientX - lightboxState.startX;
    lightboxState.translateY = e.touches[0].clientY - lightboxState.startY;
    updateLightboxTransform();
    e.preventDefault();
  });
  
  img.addEventListener('touchend', function() {
    lightboxState.isDragging = false;
  });
  
  // 鼠标滚轮缩放（优化：降低敏感度 + 节流控制）
  let lastWheelTime = 0;
  const wheelThrottle = 50; // 节流间隔（毫秒）
  
  content.addEventListener('wheel', function(e) {
    e.preventDefault();
    
    const now = Date.now();
    if (now - lastWheelTime < wheelThrottle) {
      return; // 跳过过于频繁的滚轮事件
    }
    lastWheelTime = now;
    
    const delta = e.deltaY > 0 ? -0.1 : 0.1; // 降低缩放增量
    zoomLightboxImage(delta);
  }, { passive: false });
}

/**
 * 更新灯箱图片变换
 */
function updateLightboxTransform() {
  const img = document.getElementById('lightbox-img');
  if (!img) return;
  
  img.style.transform = `translate(${lightboxState.translateX}px, ${lightboxState.translateY}px) scale(${lightboxState.scale})`;
}

/**
 * 缩放灯箱图片
 */
window.zoomLightboxImage = function(delta) {
  lightboxState.scale = Math.max(0.1, Math.min(5, lightboxState.scale + delta));
  updateLightboxTransform();
};

/**
 * 重置灯箱图片
 */
window.resetLightboxImage = function() {
  lightboxState.scale = 1;
  lightboxState.translateX = 0;
  lightboxState.translateY = 0;
  updateLightboxTransform();
};

/**
 * 复制灯箱图片到剪贴板
 */
window.copyLightboxImage = async function(src, originalPath, altText) {
  try {
    // 显示加载提示
    showToast('正在复制图片...', 'info');
    
    // 获取图片 blob
    const response = await fetch(src);
    const blob = await response.blob();
    
    // 使用 Clipboard API 复制图片
    if (navigator.clipboard && ClipboardItem) {
      await navigator.clipboard.write([
        new ClipboardItem({
          [blob.type]: blob
        })
      ]);
      showToast('图片已复制到剪贴板', 'success');
    } else {
      // 浏览器不支持图片复制
      showToast('您的浏览器不支持图片复制功能', 'warning');
    }
  } catch (error) {
    console.error('Copy image error:', error);
    
    // 如果是本地文件，提示用户从文件系统复制
    if (originalPath && !originalPath.startsWith('http')) {
      showToast('本地文件路径：' + originalPath, 'info');
    } else {
      showToast('图片复制失败：' + error.message, 'error');
    }
  }
};

// ========== Excel Rendering Functions ==========

/**
 * 渲染 Excel 文件到指定容器
 * @param {Blob} blob - Excel 文件的 Blob 对象
 * @param {HTMLElement} displayElement - 显示容器元素
 * @param {Object} options - 配置选项
 * @param {number} options.maxRows - 最大显示行数，默认 3000
 * @param {boolean} options.showSheetSelector - 是否显示工作表选择器，默认 true
 */
window.renderExcelInContainer = async function(blob, displayElement, options = {}) {
  // 默认配置
  const config = {
    maxRows: options.maxRows || 3000,
    showSheetSelector: options.showSheetSelector !== false
  };
  
  // Text strings
  const t = {
    rowLimitWarning: (current, total) => `⚠️ Too many rows, showing only first ${current} rows (total ${total} rows)`,
    worksheet: 'Worksheet',
    worksheetLabel: 'Worksheet: ',
    totalSheets: (count) => `Total ${count} worksheet(s)`,
    parseError: 'Excel parsing failed'
  };
  
  try {
    const arrayBuffer = await blob.arrayBuffer();
    const workbook = XLSX.read(arrayBuffer, { type: 'array' });
    
    // 存储 workbook 到全局，供切换工作表使用
    window._currentExcelWorkbook = workbook;
    window._currentExcelDisplayElement = displayElement;
    window._currentExcelConfig = config;
    window._currentExcelI18n = t;
    
    // 渲染第一个工作表
    renderExcelSheet(0);
    
  } catch (error) {
    console.error('Excel rendering error:', error);
    displayElement.innerHTML = `
      <div class="file-type-notice">
        <div class="icon">❌</div>
        <h3>${t.parseError}</h3>
        <p style="color: var(--error);">${escapeHtml(error.message)}</p>
      </div>
    `;
  }
};

/**
 * 渲染指定的 Excel 工作表
 * @param {number} sheetIndex - 工作表索引
 */
function renderExcelSheet(sheetIndex) {
  const workbook = window._currentExcelWorkbook;
  const displayElement = window._currentExcelDisplayElement;
  const config = window._currentExcelConfig;
  const t = window._currentExcelI18n;
  
  if (!workbook || !displayElement) return;
  
  const sheetName = workbook.SheetNames[sheetIndex];
  const worksheet = workbook.Sheets[sheetName];
  
  // 限制行数防止性能问题
  let rowLimitWarning = '';
  if (worksheet['!ref']) {
    const range = XLSX.utils.decode_range(worksheet['!ref']);
    const totalRows = range.e.r + 1;
    if (totalRows > config.maxRows) {
      range.e.r = config.maxRows - 1;
      worksheet['!ref'] = XLSX.utils.encode_range(range);
      rowLimitWarning = `
        <div class="excel-row-limit-warning" style="margin-bottom: var(--spacing-md); padding: var(--spacing-sm); background-color: var(--warning-bg, #fff3cd); border-left: 3px solid var(--warning, #ffc107); border-radius: var(--radius-md);">
          <span style="color: var(--warning-dark, #856404); font-size: var(--font-size-sm);">${t.rowLimitWarning(config.maxRows, totalRows)}</span>
        </div>
      `;
    }
  }
  
  // 转换为 HTML 表格
  const htmlTable = XLSX.utils.sheet_to_html(worksheet);
  
  // 工作表选择器
  let sheetSelector = '';
  if (config.showSheetSelector) {
    if (workbook.SheetNames.length > 1) {
      // 多工作表：显示下拉选择器
      sheetSelector = `
        <div style="margin-bottom: var(--spacing-md); padding: var(--spacing-sm); background-color: var(--gray-2); border-radius: var(--radius-md);">
          <label style="font-weight: 600; margin-right: var(--spacing-sm); color: var(--gray-9);">${t.worksheet}:</label>
          <select id="excel-sheet-selector" onchange="window.switchExcelSheet(this.value)" style="padding: 6px 12px; border: 1px solid var(--gray-4); border-radius: var(--radius-sm); background-color: white; cursor: pointer; font-size: var(--font-size-sm);">
            ${workbook.SheetNames.map((name, idx) =>
              `<option value="${idx}" ${idx === sheetIndex ? 'selected' : ''}>${escapeHtml(name)}</option>`
            ).join('')}
          </select>
          <span style="margin-left: var(--spacing-sm); color: var(--gray-7); font-size: var(--font-size-sm);">${t.totalSheets(workbook.SheetNames.length)}</span>
        </div>
      `;
    } else {
      // 单工作表：仅显示名称
      sheetSelector = `
        <div style="margin-bottom: var(--spacing-md); padding: var(--spacing-sm); background-color: var(--gray-2); border-radius: var(--radius-md);">
          <span style="font-weight: 600; color: var(--gray-9);">${t.worksheetLabel}</span>
          <span style="color: var(--gray-8);">${escapeHtml(sheetName)}</span>
        </div>
      `;
    }
  }
  
  // 渲染内容
  displayElement.innerHTML = `
    ${sheetSelector}
    ${rowLimitWarning}
    <div class="excel-table-container">
      ${htmlTable.replace('<table', '<table class="excel-table"')}
    </div>
  `;
}

/**
 * 切换 Excel 工作表
 * @param {string|number} sheetIndex - 工作表索引
 */
window.switchExcelSheet = function(sheetIndex) {
  renderExcelSheet(parseInt(sheetIndex));
};

