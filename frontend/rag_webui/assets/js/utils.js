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
