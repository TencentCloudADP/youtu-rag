// Internationalization (i18n) Configuration

const translations = {
  zh: {
    // Version
    version: 'Youtu-RAG v1.0.0',
    version_number: 'v1.0.0',

    // Language
    language: '语言',
    chinese: '中文',
    english: 'English',

    // About Page
    about_title: 'Youtu-RAG',
    about_core_concept: '本地部署 · 自主决策 · 记忆驱动',
    about_badge: '新一代智能体驱动的检索增强生成系统',
    about_subtitle: '具备自主决策与记忆学习能力的智能检索系统。',
    about_subtitle_2: '个人本地知识库管理和问答系统的最佳实践。',
    about_version: '版本',
    about_license: '许可证',
    about_highlights_title: '🔥 技术亮点',
    about_highlights_desc: '突破传统RAG系统限制，打造智能、安全、高效的新一代系统',
    about_stat_agents: '开箱即用 Agent',
    about_stat_memory: '记忆轮数支持',
    about_stat_formats: '文档格式支持',
    about_stat_local: '本地部署能力',
    about_tech_features_title: '✨ 核心技术特性',
    about_feature_memory_title: '双层记忆机制',
    about_feature_memory_desc: '短期会话内信息记忆 + 长期跨会话知识沉淀，实现QA经验的记忆与学习。',
    about_feature_file_title: '文件中心化架构',
    about_feature_file_desc: '以文件为核心的知识组织，支持 PDF、Excel、图片、数据库等多源异构数据接入。',
    about_feature_adaptive_title: '智能检索引擎',
    about_feature_adaptive_desc: '自主决策最优检索策略，支持网络搜索、向量检索、元数据过滤、数据库查询、代码执行等丰富的工具调用。',
    about_feature_ready_agents_title: '开箱即用Agent',
    about_feature_ready_agents_desc: '从简单对话到复杂编排，覆盖多种应用级场景。支持Web Search、KB Search、Meta Retrieval、Excel Agent、Text2SQL等8+智能体。',
    about_feature_ui_title: '轻量级WebUI',
    about_feature_ui_desc: '纯原生 HTML + CSS + JavaScript 实现，无框架依赖。支持文件上传、知识库管理、AI对话、文档预览等完整功能。',
    about_feature_security_title: '安全可控',
    about_feature_security_desc: '相关组件均支持本地部署，数据不出域。集成MinIO对象存储，支持大规模文件本地化管理。',
    // Feature tags
    about_tag_adaptive_search: '自主决策',
    about_tag_sql_query: '工具调用',
    about_tag_diverse_data_sources: '多样化数据源',
    about_tag_short_term_memory: '短期记忆',
    about_tag_long_term_memory: '长期记忆',
    about_tag_qa_learning: 'QA学习',
    about_tag_ready_to_use: '开箱即用',
    about_tag_diverse_scenarios: '多样化场景',
    about_tag_task_coordination: '复杂任务协同',
    about_tag_zero_dependency: '零依赖',
    about_tag_streaming_response: '流式响应',
    about_tag_easy_operation: '操作便捷',
    about_tag_local_deployment: '本地部署',
    about_tag_data_isolation: '数据隔离',
    about_tag_minio: 'MinIO',
    about_architecture_title: '🏗️ 系统架构',
    about_architecture_desc: '从固定流程到自主智能体，通过"感知-决策-执行"闭环实现智能检索',
    about_arch_img_placeholder: '[ 系统架构图占位 - 待添加 ]',
    about_benchmark_title: '📊 评测指标',
    about_benchmark_desc: '完整的评测体系，支持多维度能力验证',
    about_bench_text2sql: '结构化检索 (Text2SQL)',
    about_bench_text2sql_desc: '自然语言转SQL、Schema理解、SQL执行',
    about_bench_text2sql_metric: 'Multi-table准确率',
    about_bench_excel: '半结构化检索 (Excel)',
    about_bench_excel_desc: '表格理解、数据分析、非标准表格解析',
    about_bench_excel_metric: '可视化质量评分',
    about_bench_reading: '阅读理解 (长文本)',
    about_bench_reading_desc: '长文档信息抽取、推理验证',
    about_bench_reading_metric: 'FactGuard准确率',
    about_bench_meta: '元数据检索',
    about_bench_meta_desc: '问题意图理解、元数据过滤和重排',
    about_bench_meta_metric: '平均NDCG@5',
    about_tech_stack_title: '🛠️ 技术栈',
    about_stack_framework: '🤖 Agent框架',
    about_stack_embedding: '🔤 向量化模型',
    about_stack_parsing: '📄 文档解析',
    about_stack_storage: '💾 存储组件',
    about_opensource_title: '🙏 开源致谢',
    about_opensource_desc: 'Youtu-RAG 基于多个开源项目的卓越成果构建而成：',
    
    // Core Features Section
    about_core_features_title: '✨ 核心功能',
    about_core_features_desc: '从智能对话到知识管理，全方位满足个人应用需求',
    
    // Feature 1: Agents
    about_feature_agents_title: '开箱即用 Agent',
    about_agent_chat: 'Chat - 基础对话',
    about_agent_web: 'Web Search - 网络搜索',
    about_agent_kb: 'KB Search - 知识库检索',
    about_agent_meta: 'Meta Retrieval - 元数据检索',
    about_agent_file: 'File QA - 文件问答',
    about_agent_excel: 'Excel Research - 表格分析',
    about_agent_sql: 'Text2SQL - SQL查询',
    about_agent_parallel: 'Parallel Orchestrator - 并行编排',
    
    // Feature 2: Document Formats
    about_feature_formats_title: '多类型文档处理',
    about_format_pdf: 'PDF - 文本提取 / OCR识别',
    about_format_word: 'Word - 格式保留 / 结构提取',
    about_format_excel: 'Excel - 表格解析 / 数据库写入',
    about_format_image: '图片 - OCR识别 / Markdown转换',
    about_format_text: 'Text/Markdown - 纯文本处理',
    about_format_more: '+12 更多格式支持...',
    
    // Feature 3: Knowledge Base
    about_feature_kb_title: '高级知识库能力',
    about_kb_minio: 'MinIO - 对象存储 / 元数据管理',
    about_kb_db: 'SQLite/MySQL - 关系数据库',
    about_kb_vector: 'ChromaDB - 向量存储与检索',
    about_kb_embedding: 'Youtu-Embedding - 向量化模型',
    about_kb_chunking: 'Youtu-HiChunk - 结构化切分',
    about_kb_parsing: 'Youtu-Parsing - 多模态OCR',
    
    // Feature 4: Modern UI
    about_feature_ui_complete_title: '完整前端体验',
    about_ui_upload: '文件上传 / 批量管理',
    about_ui_kb: '知识库构建 / 关联管理',
    about_ui_chat: '智能对话 / 流式响应',
    about_ui_preview: '文档预览 / 效果查看',
    about_ui_memory: '记忆模式 / 上下文管理',
    
    // Feature 5: Local Deployment
    about_feature_local_title: '本地部署能力',
    about_local_all: '所有组件可本地部署',
    about_local_secure: '数据不出域 / 安全可控',
    about_local_model: '端侧小模型 + 大模型',
    about_local_hybrid: '混合部署 / 灵活配置'
  },

  en: {
    // Version
    version: 'Youtu-RAG v1.0.0',
    version_number: 'v1.0.0',

    // Language
    language: 'Language',
    chinese: '中文',
    english: 'English',

    // About Page
    about_title: 'Youtu-RAG',
    about_core_concept: 'Local Deployment · Autonomous Decision · Memory-Driven',
    about_badge: 'Next-Generation Agentic Intelligent Retrieval-Augmented Generation System',
    about_subtitle: 'An intelligent retrieval system with autonomous decision and memory learning capabilities.',
    about_subtitle_2: 'Best practices for personal local knowledge base management and Q&A systems.',
    about_main_img_placeholder: '[ Main Architecture Image - To be Added ]',
    about_version: 'Version',
    about_license: 'License',
    about_highlights_title: '🔥 Technical Highlights',
    about_highlights_desc: 'Breaking through traditional RAG limitations, creating intelligent, secure, and efficient next-gen system',
    about_stat_agents: 'Ready-to-use Agents',
    about_stat_memory: 'Memory Rounds Support',
    about_stat_formats: 'Document Format Support',
    about_stat_local: 'Local Deployment',
    about_tech_features_title: '✨ Core Technical Features',
    about_feature_memory_title: 'Dual-Layer Memory Mechanism',
    about_feature_memory_desc: 'Short-term session-based information memory + long-term cross-session knowledge storage, enabling the learning of QA experience.',
    about_feature_file_title: 'File-Centric Architecture',
    about_feature_file_desc: 'File-centric knowledge organization, supporting access to multi-source heterogeneous data including PDF, Excel, images, databases, etc.',
    about_feature_adaptive_title: 'Agentic Retrieval Engine',
    about_feature_adaptive_desc: 'Autonomous decision-making on the optimal retrieval strategy, supporting a variety of tool calls such as web search, vector retrieval, metadata filtering, database queries, and code execution.',
    about_feature_ready_agents_title: 'Ready-to-Use Agents',
    about_feature_ready_agents_desc: 'From simple conversations to complex orchestrations, covering a wide range of application-level scenarios. Supports over 8 AI agents including Web Search, KB Search, Meta Retrieval, Excel Agent, and Text2SQL.',
    about_feature_ui_title: 'Lightweight WebUI',
    about_feature_ui_desc: 'Pure native HTML + CSS + JavaScript implementation with no framework dependencies. Supports file upload, knowledge base management, AI conversation, document preview, and more.',
    about_feature_security_title: 'Secure & Controllable',
    about_feature_security_desc: 'All components support local deployment with data isolation. Integrated MinIO object storage for large-scale file localization management.',
    // Feature tags
    about_tag_adaptive_search: 'Autonomous Decision',
    about_tag_tool_call: 'Tool Call',
    about_tag_diverse_data_sources: 'Diversified Data Sources',
    about_tag_short_term_memory: 'Short-term Memory',
    about_tag_long_term_memory: 'Long-term Memory',
    about_tag_qa_learning: 'QA Learning',
    about_tag_ready_to_use: 'Ready-to-Use',
    about_tag_diverse_scenarios: 'Diverse Scenarios',
    about_tag_task_coordination: 'Complex Task Collaboration',
    about_tag_zero_dependency: 'Zero Dependency',
    about_tag_streaming_response: 'Streaming Response',
    about_tag_easy_operation: 'Easy Operation',
    about_tag_local_deployment: 'Local Deployment',
    about_tag_data_isolation: 'Data Isolation',
    about_tag_minio: 'MinIO',
    about_architecture_title: '🏗️ System Architecture',
    about_architecture_desc: 'From fixed workflows to autonomous agents, achieving intelligent retrieval through "Perception-Decision-Execution" loop',
    about_arch_img_placeholder: '[ System Architecture Image - To be Added ]',
    about_benchmark_title: '📊 Evaluation',
    about_benchmark_desc: 'Complete evaluation system supporting multi-dimensional capability verification',
    about_bench_text2sql: 'Structured Retrieval (Text2SQL)',
    about_bench_text2sql_desc: 'Natural language to SQL, schema understanding, SQL execution',
    about_bench_text2sql_metric: 'Multi-table Accuracy',
    about_bench_excel: 'Semi-Structured Retrieval (Excel)',
    about_bench_excel_desc: 'Table understanding, data analysis, non-standard table parsing',
    about_bench_excel_metric: 'Visualization Quality Score',
    about_bench_reading: 'Reading Comprehension (Long Text)',
    about_bench_reading_desc: 'Long document information extraction, reasoning verification',
    about_bench_reading_metric: 'FactGuard Accuracy',
    about_bench_meta: 'Metadata Retrieval',
    about_bench_meta_desc: 'Question intent understanding, metadata filtering and reranking',
    about_bench_meta_metric: 'Average NDCG@5',
    about_tech_stack_title: '🛠️ Tech Stack',
    about_stack_framework: '🤖 Agent Framework',
    about_stack_embedding: '🔤 Embedding Models',
    about_stack_parsing: '📄 Document Parsing',
    about_stack_storage: '💾 Storage Components',
    about_opensource_title: '🙏 Open Source Acknowledgments',
    about_opensource_desc: 'Youtu-RAG builds upon the excellent work of several open-source projects:',
    
    // Core Features Section
    about_core_features_title: '✨ Core Features',
    about_core_features_desc: 'From intelligent conversation to knowledge management, meeting all personal application needs',
    
    // Feature 1: Agents
    about_feature_agents_title: 'Ready-to-Use Agents',
    about_agent_chat: 'Chat - Basic Conversation',
    about_agent_web: 'Web Search - Internet Search',
    about_agent_kb: 'KB Search - Knowledge Base Retrieval',
    about_agent_meta: 'Meta Retrieval - Metadata Retrieval',
    about_agent_file: 'File QA - File Q&A',
    about_agent_excel: 'Excel Research - Spreadsheet Analysis',
    about_agent_sql: 'Text2SQL - SQL Query',
    about_agent_parallel: 'Parallel Orchestrator - Parallel Orchestration',
    
    // Feature 2: Document Formats
    about_feature_formats_title: 'Multi-Type Document Processing',
    about_format_pdf: 'PDF - Text Extraction / OCR Recognition',
    about_format_word: 'Word - Format Preservation / Structure Extraction',
    about_format_excel: 'Excel - Table Parsing / Database Writing',
    about_format_image: 'Image - OCR Recognition / Markdown Conversion',
    about_format_text: 'Text/Markdown - Plain Text Processing',
    about_format_more: '+12 More Formats...',
    
    // Feature 3: Knowledge Base
    about_feature_kb_title: 'Advanced Knowledge Base Capabilities',
    about_kb_minio: 'MinIO - Object Storage / Metadata Management',
    about_kb_db: 'SQLite/MySQL - Relational Database',
    about_kb_vector: 'ChromaDB - Vector Storage & Retrieval',
    about_kb_embedding: 'Youtu-Embedding - Vectorization Model',
    about_kb_chunking: 'Youtu-HiChunk - Structured Chunking',
    about_kb_parsing: 'Youtu-Parsing - Multimodal OCR',
    
    // Feature 4: Modern UI
    about_feature_ui_complete_title: 'Complete Frontend Experience',
    about_ui_upload: 'File Upload / Batch Management',
    about_ui_kb: 'Knowledge Base Building / Association Management',
    about_ui_chat: 'Intelligent Conversation / Streaming Response',
    about_ui_preview: 'Document Preview / Effect Viewing',
    about_ui_memory: 'Memory Mode / Context Management',
    
    // Feature 5: Local Deployment
    about_feature_local_title: 'Local Deployment Capability',
    about_local_all: 'All components can be deployed locally',
    about_local_secure: 'Data stays local / Secure & controllable',
    about_local_model: 'Edge small model + Large model',
    about_local_hybrid: 'Hybrid deployment / Flexible configuration'
  }
};

// i18n Manager
class I18n {
  constructor() {
    // Get saved language from localStorage, default to 'en'
    this.currentLang = localStorage.getItem('app_language') || 'en';
    this.listeners = [];
  }

  // Get current language
  getLang() {
    return this.currentLang;
  }

  // Set language
  setLang(lang) {
    if (lang !== 'zh' && lang !== 'en') {
      console.warn('Unsupported language:', lang);
      return;
    }

    this.currentLang = lang;
    localStorage.setItem('app_language', lang);

    // Update HTML lang attribute
    document.documentElement.lang = lang === 'zh' ? 'zh-CN' : 'en';

    // Notify all listeners
    this.listeners.forEach(callback => callback(lang));
  }

  // Toggle language
  toggleLang() {
    const newLang = this.currentLang === 'zh' ? 'en' : 'zh';
    this.setLang(newLang);
  }

  // Get translation
  t(key, params = {}) {
    const translation = translations[this.currentLang]?.[key] || key;

    // Replace parameters in translation
    return translation.replace(/\{(\w+)\}/g, (match, param) => {
      return params[param] !== undefined ? params[param] : match;
    });
  }

  // Add language change listener
  onChange(callback) {
    this.listeners.push(callback);
  }

  // Remove language change listener
  offChange(callback) {
    const index = this.listeners.indexOf(callback);
    if (index > -1) {
      this.listeners.splice(index, 1);
    }
  }
}

// Create global i18n instance
const i18n = new I18n();

// Helper function for quick translation
function t(key, params) {
  return i18n.t(key, params);
}
