// API Configuration and Methods

const API_BASE = window.APP_CONFIG?.API_BASE || '';

class API {
  // Generic fetch wrapper
  static async request(url, options = {}) {
    try {
      const response = await fetch(API_BASE + url, {
        headers: {
          'Content-Type': 'application/json',
          ...options.headers
        },
        ...options
      });

      if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: response.statusText }));
        throw new Error(error.detail || 'Request failed');
      }

      return await response.json();
    } catch (error) {
      console.error('API Error:', error);
      throw error;
    }
  }

  // File Management APIs
  static async getFiles(page = 1, pageSize = 10, search = '') {
    // Backend returns all files without pagination
    const files = await this.request('/api/files/list');

    // Client-side filtering and pagination
    let filteredFiles = files;
    if (search) {
      const searchLower = search.toLowerCase();
      filteredFiles = files.filter(file =>
        file.name.toLowerCase().includes(searchLower) ||
        (file.metadata && JSON.stringify(file.metadata).toLowerCase().includes(searchLower))
      );
    }

    // Map backend format to frontend format
    const mappedFiles = filteredFiles.map(file => ({
      filename: file.name,
      size: file.size,
      created_at: file.last_modified,
      upload_time: file.last_modified,
      last_modified: file.last_modified,
      ocr_status: file.metadata?.ocr_status || 'pending',
      chunk_status: file.metadata?.chunk_status || 'pending',
      etag: file.etag,
      metadata: file.metadata
    }));

    // Sort by last_modified (newest first) BEFORE pagination
    // This ensures newly uploaded or overwritten files appear at the top
    mappedFiles.sort((a, b) => {
      const timeA = a.last_modified || a.created_at || a.upload_time;
      const timeB = b.last_modified || b.created_at || b.upload_time;

      if (!timeA && !timeB) return 0;
      if (!timeA) return 1;
      if (!timeB) return -1;

      const dateA = new Date(timeA);
      const dateB = new Date(timeB);

      return dateB - dateA; 
    });

    // Client-side pagination
    const total = mappedFiles.length;
    const start = (page - 1) * pageSize;
    const end = start + pageSize;
    const paginatedFiles = mappedFiles.slice(start, end);

    return {
      files: paginatedFiles,
      total: total,
      page: page,
      page_size: pageSize,
      total_pages: Math.ceil(total / pageSize)
    };
  }

  static async uploadFile(formData) {
    const response = await fetch(API_BASE + '/api/files/upload', {
      method: 'POST',
      body: formData
    });
    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: response.statusText }));
      throw new Error(error.detail || 'Upload failed');
    }
    return await response.json();
  }

  static async deleteFile(filename) {
    return this.request(`/api/files/delete/${encodeURIComponent(filename)}`, {
      method: 'DELETE'
    });
  }

  static async getFileMetadata(filename) {
    return this.request(`/api/files/metadata/${encodeURIComponent(filename)}`);
  }

  static async updateFileMetadata(filename, metadata) {
    return this.request(`/api/files/metadata/${encodeURIComponent(filename)}`, {
      method: 'PUT',
      body: JSON.stringify(metadata)
    });
  }

  static async getOCRResults(filename) {
    return this.request(`/api/files/ocr-results/${encodeURIComponent(filename)}`);
  }

  static async saveOCRResults(filename, content) {
    return this.request(`/api/files/ocr-results/${encodeURIComponent(filename)}/save`, {
      method: 'POST',
      body: JSON.stringify({ content })
    });
  }

  // Knowledge Base APIs
  static async getKnowledgeBases() {
    console.log('[API] Calling GET /api/knowledge/list');
    console.log('[API] API_BASE:', API_BASE);
    try {
      const result = await this.request('/api/knowledge/list');
      console.log('[API] getKnowledgeBases success:', result);
      return result;
    } catch (error) {
      console.error('[API] getKnowledgeBases error:', error);
      throw error;
    }
  }

  static async createKnowledgeBase(data) {
    return this.request('/api/knowledge/create', {
      method: 'POST',
      body: JSON.stringify(data)
    });
  }

  static async updateKnowledgeBase(id, data) {
    return this.request(`/api/knowledge/${id}`, {
      method: 'PUT',
      body: JSON.stringify(data)
    });
  }

  static async deleteKnowledgeBase(id) {
    return this.request(`/api/knowledge/${id}`, {
      method: 'DELETE'
    });
  }

  static async getKnowledgeBase(id) {
    return this.request(`/api/knowledge/${id}`);
  }

  static async selectKnowledgeBase(id) {
    return this.request(`/api/knowledge-base/select/${id}`, {
      method: 'POST'
    });
  }

  static async getKnowledgeBaseFiles(kbId) {
    return this.request(`/api/knowledge-base/${kbId}/files`);
  }

  static async selectKnowledgeBaseFile(fileIds) {
    return this.request('/api/knowledge-base/select-file', {
      method: 'POST',
      body: JSON.stringify({ source_ids: fileIds })
    });
  }

  static async buildKnowledgeBase(id) {
    return this.request(`/api/knowledge/${id}/build`, {
      method: 'POST',
      body: JSON.stringify({
        force_rebuild: false,
        file_filter: null
      })
    });
  }

  static async getKnowledgeBaseFileStatus(id) {
    return this.request(`/api/knowledge/${id}/file-status`);
  }

  // Chat/Agent APIs
  static async sendMessage(data) {
    return this.request('/api/chat', {
      method: 'POST',
      body: JSON.stringify(data)
    });
  }

  static async getAgentStream(data) {
    const response = await fetch(API_BASE + '/api/agent/stream', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(data)
    });

    if (!response.ok) {
      throw new Error('Stream request failed');
    }

    return response.body;
  }

  // Agent APIs
  static async getAgents() {
    const response = await this.request('/api/agent/list');
    // Backend returns { agents: [...], current: "..." }
    return response;
  }

  static async switchAgent(configPath) {
    return this.request('/api/agent/switch', {
      method: 'POST',
      body: JSON.stringify({ config_path: configPath })
    });
  }

  // Knowledge Base Configuration APIs
  static async updateKBConfiguration(kb_id, configuration) {
    return this.request(`/api/knowledge/${kb_id}/configuration`, {
      method: 'PUT',
      body: JSON.stringify({ configuration })
    });
  }

  static async testDatabaseConnection(dbConfig) {
    return this.request('/api/knowledge/database/test-connection', {
      method: 'POST',
      body: JSON.stringify(dbConfig)
    });
  }

  static async validateQAFile(filename) {
    return this.request(`/api/knowledge/files/validate-qa/${encodeURIComponent(filename)}`);
  }

  // Q&A Association APIs
  static async getQAAssociations(kb_id, source_file) {
    return this.request(`/api/knowledge/${kb_id}/qa/${encodeURIComponent(source_file)}`);
  }

  static async updateQAStatus(kb_id, qa_id, status) {
    return this.request(`/api/knowledge/${kb_id}/qa/${qa_id}/status`, {
      method: 'PUT',
      body: JSON.stringify({ learning_status: status })
    });
  }

  static async executeQA(kb_id, qa_id) {
    return this.request(`/api/knowledge/${kb_id}/qa/${qa_id}/execute`, {
      method: 'POST'
    });
  }

  static async batchExecuteQA(kb_id, qa_ids) {
    return this.request(`/api/knowledge/${kb_id}/qa/batch-execute`, {
      method: 'POST',
      body: JSON.stringify({ qa_ids })
    });
  }
}
