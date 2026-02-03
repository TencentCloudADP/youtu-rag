// Simple Hash Router

class Router {
  constructor() {
    this.routes = {
      '/': 'pages/file-management.html',
      '/files': 'pages/file-management.html',
      '/knowledge': 'pages/knowledge-base.html',
      '/chat': 'pages/chat.html'
    };

    // Dynamic route patterns
    this.dynamicRoutes = [
      {
        pattern: /^\/knowledge\/(.+)$/,
        page: 'pages/knowledge-base-detail.html',
        init: 'initKnowledgeBaseDetail'
      }
    ];

    this.currentPage = null;
    this.init();
  }

  init() {
    // Listen for hash changes
    window.addEventListener('hashchange', () => this.handleRoute());

    // Handle initial route
    this.handleRoute();
  }

  async handleRoute() {
    const hash = window.location.hash.slice(1) || '/';

    // Check static routes first
    let route = this.routes[hash];
    let initFunction = null;

    // If not found in static routes, check dynamic routes
    if (!route) {
      for (const dynamicRoute of this.dynamicRoutes) {
        const match = hash.match(dynamicRoute.pattern);
        if (match) {
          route = dynamicRoute.page;
          initFunction = dynamicRoute.init;
          break;
        }
      }
    }

    if (!route) {
      console.error('Route not found:', hash);
      return;
    }

    // Update active nav item
    this.updateActiveNav(hash);

    // Load page content
    await this.loadPage(route, initFunction);
  }

  async loadPage(pagePath, customInitFunction = null) {
    const container = document.getElementById('content-container');
    if (!container) return;

    try {
      // Show loading
      container.innerHTML = '<div class="spinner spinner-large" style="margin: 50px auto;"></div>';

      // Fetch page content
      const response = await fetch(pagePath);
      if (!response.ok) throw new Error('Failed to load page');

      const html = await response.text();

      // Update content with fade animation
      container.style.opacity = '0';
      setTimeout(() => {
        container.innerHTML = html;
        container.style.opacity = '1';

        // Apply translations to the newly loaded page
        if (typeof updatePageTranslations === 'function') {
          updatePageTranslations();
        }

        // Wait for DOM to be fully rendered before initializing scripts
        requestAnimationFrame(() => {
          setTimeout(() => {
            if (customInitFunction) {
              this.callInitFunction(customInitFunction);
            } else {
              this.initPageScripts(pagePath);
            }
          }, 0);
        });
      }, 150);

    } catch (error) {
      console.error('Error loading page:', error);
      container.innerHTML = `
        <div class="empty-state">
          <div class="icon">⚠</div>
          <h2>加载失败</h2>
          <p>${error.message}</p>
        </div>
      `;
    }
  }

  callInitFunction(functionName) {
    console.log('[Router] Calling custom init function:', functionName);
    if (typeof window[functionName] === 'function') {
      window[functionName]();
    } else {
      console.warn('[Router] Custom init function not found:', functionName);
    }
  }

  updateActiveNav(path) {
    // Remove active class from all nav items
    document.querySelectorAll('.nav-item').forEach(item => {
      item.classList.remove('active');
    });

    // Add active class to current nav item
    const currentNav = document.querySelector(`[href="#${path}"]`);
    if (currentNav) {
      currentNav.classList.add('active');
    }
  }

  initPageScripts(pagePath) {
    console.log('[Router] Initializing scripts for:', pagePath);

    // Initialize page-specific functionality
    if (pagePath.includes('file-management')) {
      if (typeof initFileManagement === 'function') {
        console.log('[Router] Calling initFileManagement()');
        initFileManagement();
      } else {
        console.warn('[Router] initFileManagement function not found');
      }
    } else if (pagePath.includes('knowledge-base')) {
      if (typeof initKnowledgeBase === 'function') {
        console.log('[Router] Calling initKnowledgeBase()');
        initKnowledgeBase();
      } else {
        console.warn('[Router] initKnowledgeBase function not found');
      }
    } else if (pagePath.includes('chat')) {
      if (typeof initChat === 'function') {
        console.log('[Router] Calling initChat()');
        initChat();
      } else {
        console.warn('[Router] initChat function not found');
      }
    }
  }

  navigate(path) {
    window.location.hash = path;
  }
}

// Initialize router when DOM is ready
let router;
document.addEventListener('DOMContentLoaded', () => {
  router = new Router();
});
