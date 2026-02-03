// Frontend Configuration

const CONFIG = {
  API_BASE: '',
  VERSION: '1.0.0',
  APP_NAME: 'Youtu-RAG'
};

window.APP_CONFIG = CONFIG;

if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
  console.log('[Frontend Config]', CONFIG);
}
