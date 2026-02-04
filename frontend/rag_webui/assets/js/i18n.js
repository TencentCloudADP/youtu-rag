// Internationalization (i18n) Configuration

const translations = {
  zh: {
    // Navigation
    nav_files: '源文件',
    nav_knowledge: '知识库',
    nav_chat: '问AI',
    nav_documents: '文档',

    // Common
    loading: '加载中...',
    confirm: '确认',
    cancel: '取消',
    save: '保存',
    delete: '删除',
    edit: '编辑',
    search: '搜索',
    upload: '上传',
    download: '下载',
    close: '关闭',
    back: '返回',
    next: '下一步',
    previous: '上一步',
    submit: '提交',
    reset: '重置',
    refresh: '刷新',

    // Time
    just_now: '刚刚',
    minutes_ago: ' 分钟前',
    hours_ago: ' 小时前',
    days_ago: ' 天前',

    // Toast messages
    toast_copy_success: '已复制到剪贴板',
    toast_copy_failed: '复制失败',
    toast_operation_success: '操作成功',
    toast_operation_failed: '操作失败',
    toast_select_excel_file: '请选择 Excel 文件 (.xlsx 或 .xls)',

    // File Manager
    file_management_title: '源文件资源',
    file_list: '文件列表',
    file_name: '文件名',
    file_size: '文件大小',
    file_type: '文件类型',
    upload_time: '上传时间',
    update_time: '更新时间',
    actions: '操作',
    upload_file: '上传文件',
    batch_delete: '批量删除',
    batch_select: '批量选择',
    select_mode: '选择模式',
    exit_select_mode: '退出选择',
    selected_count: '已选择 {count} 个文件',
    delete_selected: '删除选中({count})',
    confirm_delete: '确认删除',
    confirm_delete_message: '确定要删除选中的文件吗？',
    confirm_delete_file: '⚠️ 确认删除文件',
    load_file_list_failed: '加载文件列表失败',
    loading_file_list: '加载文件列表...',
    no_files: '暂无文件',
    upload_files_prompt: '点击上传按钮添加文件',
    search_placeholder: 'Search by keyword in filename or metadata...',
    export_metadata: '导出Metadata',
    import_metadata: '导入Metadata',
    items_per_page: 'Items per page:',
    edit_metadata: '编辑文件元数据',
    metadata_filename: '文件名',
    metadata_title: '标题',
    metadata_description: '描述',
    metadata_tags: '标签',
    metadata_tags_help: '例如：技术文档, Python, 教程',
    enter_file_title: '输入文件标题',
    enter_file_description: '输入文件描述',
    enter_tags_comma_separated: '输入标签，用逗号分隔',
    import_result_title: '📥 Metadata Import Results',
    file_details_title: '📋 File Details',
    file_content: '文件内容',
    loading_file_content: '加载文件内容...',
    close_and_refresh: 'Close & Refresh',
    file_duplicate_warning: '⚠️ 文件重复',
    file_exists_message: '文件 "{filename}" 已存在。',
    file_size_label: '大小',
    last_modified_label: '最后修改',
    overwrite_confirm: '是否覆盖现有文件？',
    file_skipped: '已跳过文件: {filename}',
    no_files_to_upload: '没有文件需要上传',

    // Knowledge Base
    kb_management_title: '知识库管理',
    knowledge_base_list: '知识库列表',
    knowledge_base_name: '知识库名称',
    create_knowledge_base: '创建知识库',
    edit_knowledge_base: '编辑知识库',
    delete_knowledge_base: '删除知识库',
    knowledge_base_description: '描述',
    file_count: '文件数量',
    created_time: '创建时间',
    no_knowledge_bases: '暂无知识库',
    create_first_kb: '创建第一个知识库开始使用',
    kb_search_placeholder: 'Search by keyword in kb name or description...',
    select: '选择',
    kb_name_label: '知识库名称 *',
    kb_name_en_label: '(Knowledge Base Name)',
    kb_description_label: '描述',
    kb_description_en_label: '(Description)',
    enter_kb_name: '输入知识库名称',
    enter_kb_description: '输入知识库描述',
    max_characters: '最多{max}个字符',
    max_characters_en: '(Max {max} characters)',
    loading_kb_list: '加载知识库列表...',
    build_kb: '构建知识库',
    build_kb_progress: '正在构建知识库，请稍候...',

    // Chat
    chat_title: '智能问答',
    send_message: '发送消息',
    type_message: '输入您的问题...',
    type_message_hint: '输入您的问题... (Shift+Enter 换行，Enter 发送)',
    select_knowledge_base: '选择知识库',
    chat_history: '聊天历史',
    new_chat: '新对话',
    clear_history: '清除记录',
    clear_history_confirm: '确定要清除所有聊天记录吗？',
    upload_file_btn: '上传文件',
    agent_select: 'Agent选择',
    kb_select: '知识库选择',
    file_select: '文件选择',
    memory: '记忆',
    ai_generated_disclaimer: '回答内容均由AI生成，仅供参考',
    please_select_agent_kb: '请先选择Agent和知识库',
    send: '发送',
    stop: '停止',

    // Chat - KB Selector Hints
    kb_required_hint: '请选择知识库',
    kb_optional_hint: '选择知识库',
    kb_none_hint: '选择知识库...',

    // Chat - Agent & KB Selection
    select_agent_placeholder: '选择Agent...',
    select_kb_placeholder: '选择知识库...',
    select_file_placeholder: '选择文件...',
    no_files_available: '无可用文件',

    // Chat - Toast Messages
    toast_load_agent_failed: '加载Agent列表失败',
    toast_switch_agent_success: '已切换至 {name}',
    toast_switch_agent_failed: '切换Agent失败: {error}',
    toast_switch_kb_success: '已切换知识库',
    toast_switch_kb_failed: '切换知识库失败: {error}',
    toast_load_files_failed: '加载文件列表失败: {error}',
    toast_file_already_selected: '该文件已被选择',
    toast_file_removed: '已移除: {name}',
    toast_enter_message: '请输入消息',
    toast_select_agent: '请选择Agent',
    toast_agent_requires_kb: '{name} 需要选择知识库才能使用',
    toast_send_failed: '发送消息失败: {error}',
    toast_execution_stopped: '已停止执行',
    toast_files_added: '已添加 {count} 个文件',
    toast_chat_cleared: '聊天记录已清空',
    toast_cannot_copy: '无法复制内容',
    toast_all_files_exist: '所有文件都已存在，已取消上传',
    toast_no_files_to_upload: '没有文件需要上传',
    toast_file_removed_simple: '文件已移除',
    toast_files_associated: '文件已关联到知识库: {name}',
    toast_associate_failed: '关联知识库失败: {error}',

    // Chat - Error Messages
    error_sorry: '抱歉，发生了错误: {error}',
    error_kb_not_selected: '未选择知识库',
    error_get_kb_info_failed: '获取知识库信息失败',
    error_update_kb_config_failed: '更新知识库配置失败',
    error_upload_failed: '上传失败',
    error_upload_timeout: '上传超时',

    // Chat - UI Elements
    grid_view: '网格',
    tab_view: '标签页',
    grid_view_title: '网格视图',
    tab_view_title: '标签页视图',
    executing: '执行中...',
    completed: '完成',
    failed: '失败',
    processing: '处理中...',
    upload_complete: '上传完成！',
    remove: '移除',
    copy_content: '复制内容',
    current_kb: '当前知识库',

    // Chat - File Upload
    upload_file_title: '上传文件',
    please_select_agent_kb_first: '请先选择Agent和知识库',
    file_overwrite_confirm: '以下文件已存在：\n\n• {files}\n\n是否要覆盖这些文件？\n\n点击"确定"覆盖，点击"取消"跳过这些文件。',

    // Knowledge Base Detail
    kb_detail_title: '知识库详情',
    back: '返回',
    view_config: '查看配置',
    file_association: '文件关联',
    database_association: '关系数据库关联',
    qa_association: '示例关联与学习',
    save_association: '保存关联',
    build_knowledge_base: '构建知识库',
    select_files: '选择文件',
    search_selected_files: '搜索已选文件...',
    delete_selected: '删除选中({count})',
    processing_status: '处理状态',
    database_type: '数据库类型',
    host_address: '主机地址',
    port: '端口',
    database_name: '数据库名',
    username: '用户名',
    password: '密码',
    sqlite_file_path: 'SQLite文件路径',
    sqlite_file_path_hint: '请输入SQLite文件的完整路径',
    test_connection_load_tables: '测试连接并加载表',
    search_table_name: '搜索表名...',
    select_tables_to_include: '选择要包含的表',
    add_selected_tables: '➕ 添加选中的表到知识库',
    table_name: '表名',
    database_type_col: '数据库类型',
    database_name_col: '数据库名',
    select_qa_files: '选择Q&A文件',
    search_selected_qa_files: '搜索已选Q&A文件...',
    excel_format_requirement: 'Excel格式要求: Sheet名称"example"，列头：question, answer, howtofind',
    select_files_modal_title: '选择文件',
    search_files: '搜索文件...',
    select_all: '✓ 全选',
    select_current_page: '✓ 当前页',
    deselect_all: '✗ 取消',
    select_qa_excel_files: '选择Q&A Excel文件',
    only_show_excel_files: 'ℹ️ 只显示Excel文件（.xls, .xlsx）',
    confirm_selection: '确认选择',
    config_view: '⚙️ 配置查看',
    kb_config: '📦 知识库配置',
    default_yaml_config: '📄 默认YAML配置',
    build_kb_modal_title: '构建知识库',
    building_kb_please_wait: '正在构建知识库，请稍候...',

    // KB Detail - Toast Messages
    toast_kb_id_not_found: '未找到知识库ID',
    toast_kb_deleted: '知识库删除成功',
    toast_kb_delete_failed: '删除知识库失败: {error}',
    toast_load_kb_failed: '加载知识库详情失败: {error}',
    toast_load_files_failed_kb: '加载文件列表失败: {error}',
    toast_files_selected: '已选择 {count} 个文件',
    toast_file_removed_kb: '文件已移除',
    toast_select_files_first: '请先选择要删除的文件',
    toast_files_deleted: '已删除 {count} 个文件',
    toast_db_connection_added: '已添加数据库连接（{count}个表）',
    toast_db_connection_removed: '数据库连接已移除',
    toast_invalid_table_index: '无效的表索引',
    toast_db_connection_not_found: '找不到对应的数据库连接',
    toast_table_removed: '表已移除',
    toast_select_tables_first: '请先选择要删除的表',
    toast_tables_deleted: '已删除 {count} 个表',
    toast_select_at_least_one_table: '请至少选择一个表',
    toast_validating_excel: '正在验证Excel文件格式...',
    toast_qa_files_selected: '已选择 {count} 个符合格式的Q&A文件',
    toast_no_valid_qa_files: '没有选择符合格式要求的文件',
    toast_validate_failed: '验证文件格式失败: {error}',
    toast_selection_cancelled: '已取消选择',
    toast_qa_file_removed: 'Q&A文件已移除',
    toast_select_qa_files_first: '请先选择要删除的文件',
    toast_qa_files_deleted: '已删除 {count} 个Q&A文件',
    toast_kb_id_invalid: '知识库ID无效',
    toast_saving_config: '正在保存配置...',
    toast_config_saved: '配置保存成功',
    toast_config_save_failed: '保存配置失败: {error}',
    toast_all_files_selected: '已选择所有 {count} 个文件',
    toast_current_page_selected: '已选择当前页的 {count} 个文件',
    toast_all_deselected: '已取消所有选择',

    // KB Detail - UI Elements
    loading_kb_detail: '加载中...',
    getting_kb_info: '正在获取知识库信息',
    no_files_found: '没有找到文件',
    no_files_selected_yet: '还未选择任何文件',
    no_db_tables_yet: '还未添加任何数据库表',
    no_qa_files_yet: '还未选择任何Q&A文件',
    testing_connection: '测试连接中...',
    connection_success: '✓ 连接成功！找到 {count} 个表',
    connection_failed: '✕ 连接失败: {error}',
    no_config: '暂无配置',
    loading_config: '加载中...',
    load_config_failed: '无法加载配置文件',
    load_error: '加载失败\n\n错误信息：{error}',
    selected_count_display: '已选择 {selected} / {total}',
    selected_count_zero: '0 / {total}',

    // Q&A Detail Page
    qa_detail_title: 'Q&A详情',
    total_qa_count: '共 {count} 条',
    knowledge_base: '知识库',
    search_qa: '搜索问题或答案...',
    question: '问题',
    answer: '答案',
    how_to_find: '如何查找',
    source_file: '源文件',
    created_at: '创建时间',
    learning_status: '学习状态',
    updated_at: '更新时间',
    execute: '执行',
    batch_execute: '批量执行({count})',
    no_qa_found: '未找到Q&A数据',
    qa_detail_modal_title: 'Q&A详情',
    status_pending: '待处理',
    status_learning: '学习中',
    status_completed: '已完成',
    status_failed: '失败',
    toast_invalid_url: 'URL格式无效',
    toast_load_qa_failed: '加载Q&A数据失败: {error}',
    toast_qa_execution_started: 'Q&A #{id} 开始执行',
    toast_qa_execution_completed: 'Q&A #{id} 执行完成',
    toast_select_qa_first: '请先选择要执行的Q&A',
    toast_batch_execution_started: '开始批量执行 {count} 条Q&A',
    toast_batch_execution_completed: '批量执行完成，共 {count} 条Q&A',
    toast_memory_not_enabled: 'Memory功能未开启，请先开启Memory',

    // Version
    version: 'Youtu-RAG v1.0.0',
    version_number: 'v1.0.0',

    // Language
    language: '语言',
    chinese: '中文',
    english: 'English'
  },

  en: {
    // Navigation
    nav_files: 'Files',
    nav_knowledge: 'Knowledge Base',
    nav_chat: 'Chat AI',
    nav_documents: 'Documents',

    // Common
    loading: 'Loading...',
    confirm: 'Confirm',
    cancel: 'Cancel',
    save: 'Save',
    delete: 'Delete',
    edit: 'Edit',
    search: 'Search',
    upload: 'Upload',
    download: 'Download',
    close: 'Close',
    back: 'Back',
    next: 'Next',
    previous: 'Previous',
    submit: 'Submit',
    reset: 'Reset',
    refresh: 'Refresh',

    // Time
    just_now: 'Just now',
    minutes_ago: ' minutes ago',
    hours_ago: ' hours ago',
    days_ago: ' days ago',

    // Toast messages
    toast_copy_success: 'Copied to clipboard',
    toast_copy_failed: 'Copy failed',
    toast_operation_success: 'Operation successful',
    toast_operation_failed: 'Operation failed',
    toast_select_excel_file: 'Please select an Excel file (.xlsx or .xls)',

    // File Manager
    file_management_title: 'Source Files',
    file_list: 'File List',
    file_name: 'File Name',
    file_size: 'File Size',
    file_type: 'File Type',
    upload_time: 'Upload Time',
    update_time: 'Update Time',
    actions: 'Actions',
    upload_file: 'Upload File',
    batch_delete: 'Batch Delete',
    batch_select: 'Batch Select',
    select_mode: 'Select Mode',
    exit_select_mode: 'Exit Select',
    selected_count: '{count} file(s) selected',
    delete_selected: 'Delete Selected ({count})',
    confirm_delete: 'Confirm Delete',
    confirm_delete_message: 'Are you sure you want to delete the selected files?',
    confirm_delete_file: '⚠️ Confirm File Deletion',
    load_file_list_failed: 'Failed to load file list',
    loading_file_list: 'Loading file list...',
    no_files: 'No files',
    upload_files_prompt: 'Click upload button to add files',
    search_placeholder: 'Search by keyword in filename or metadata...',
    export_metadata: 'Export Metadata',
    import_metadata: 'Import Metadata',
    items_per_page: 'Items per page:',
    edit_metadata: 'Edit File Metadata',
    metadata_filename: 'File Name',
    metadata_title: 'Title',
    metadata_description: 'Description',
    metadata_tags: 'Tags',
    metadata_tags_help: 'e.g.: Technical Doc, Python, Tutorial',
    enter_file_title: 'Enter file title',
    enter_file_description: 'Enter file description',
    enter_tags_comma_separated: 'Enter tags, separated by commas',
    import_result_title: '📥 Metadata Import Results',
    file_details_title: '📋 File Details',
    file_content: 'File Content',
    loading_file_content: 'Loading file content...',
    close_and_refresh: 'Close & Refresh',
    file_duplicate_warning: '⚠️ File Duplicate',
    file_exists_message: 'File "{filename}" already exists.',
    file_size_label: 'Size',
    last_modified_label: 'Last Modified',
    overwrite_confirm: 'Do you want to overwrite the existing file?',
    file_skipped: 'File skipped: {filename}',
    no_files_to_upload: 'No files to upload',

    // Knowledge Base
    kb_management_title: 'Knowledge Base Management',
    knowledge_base_list: 'Knowledge Base List',
    knowledge_base_name: 'Knowledge Base Name',
    create_knowledge_base: 'Create Knowledge Base',
    edit_knowledge_base: 'Edit Knowledge Base',
    delete_knowledge_base: 'Delete Knowledge Base',
    knowledge_base_description: 'Description',
    file_count: 'File Count',
    created_time: 'Created Time',
    no_knowledge_bases: 'No knowledge bases',
    create_first_kb: 'Create your first knowledge base to get started',
    kb_search_placeholder: 'Search by keyword in kb name or description...',
    select: 'Select',
    kb_name_label: 'Knowledge Base Name *',
    kb_name_en_label: '(Knowledge Base Name)',
    kb_description_label: 'Description',
    kb_description_en_label: '(Description)',
    enter_kb_name: 'Enter knowledge base name',
    enter_kb_description: 'Enter knowledge base description',
    max_characters: 'Max {max} characters',
    max_characters_en: '(Max {max} characters)',
    loading_kb_list: 'Loading knowledge base list...',
    build_kb: 'Build Knowledge Base',
    build_kb_progress: 'Building knowledge base, please wait...',

    // Chat
    chat_title: 'AI Chat',
    send_message: 'Send Message',
    type_message: 'Type your question...',
    type_message_hint: 'Type your question... (Shift+Enter for new line, Enter to send)',
    select_knowledge_base: 'Select Knowledge Base',
    chat_history: 'Chat History',
    new_chat: 'New Chat',
    clear_history: 'Clear History',
    clear_history_confirm: 'Are you sure you want to clear all chat history?',
    upload_file_btn: 'Upload File',
    agent_select: 'Select Agent',
    kb_select: 'Select Knowledge Base',
    file_select: 'Select File',
    memory: 'Memory',
    ai_generated_disclaimer: 'All responses are AI-generated and for reference only',
    please_select_agent_kb: 'Please select Agent and Knowledge Base first',
    send: 'Send',
    stop: 'Stop',

    // Chat - KB Selector Hints
    kb_required_hint: 'Please select a knowledge base',
    kb_optional_hint: 'Select knowledge base',
    kb_none_hint: 'Select knowledge base...',

    // Chat - Agent & KB Selection
    select_agent_placeholder: 'Select Agent...',
    select_kb_placeholder: 'Select knowledge base...',
    select_file_placeholder: 'Select file...',
    no_files_available: 'No files available',

    // Chat - Toast Messages
    toast_load_agent_failed: 'Failed to load agent list',
    toast_switch_agent_success: 'Switched to {name}',
    toast_switch_agent_failed: 'Failed to switch agent: {error}',
    toast_switch_kb_success: 'Knowledge base switched',
    toast_switch_kb_failed: 'Failed to switch knowledge base: {error}',
    toast_load_files_failed: 'Failed to load file list: {error}',
    toast_file_already_selected: 'This file has already been selected',
    toast_file_removed: 'Removed: {name}',
    toast_enter_message: 'Please enter a message',
    toast_select_agent: 'Please select an agent',
    toast_agent_requires_kb: '{name} requires a knowledge base to be selected',
    toast_send_failed: 'Failed to send message: {error}',
    toast_execution_stopped: 'Execution stopped',
    toast_files_added: 'Added {count} file(s)',
    toast_chat_cleared: 'Chat history cleared',
    toast_cannot_copy: 'Cannot copy content',
    toast_all_files_exist: 'All files already exist, upload cancelled',
    toast_no_files_to_upload: 'No files to upload',
    toast_file_removed_simple: 'File removed',
    toast_files_associated: 'Files associated with knowledge base: {name}',
    toast_associate_failed: 'Failed to associate with knowledge base: {error}',

    // Chat - Error Messages
    error_sorry: 'Sorry, an error occurred: {error}',
    error_kb_not_selected: 'No knowledge base selected',
    error_get_kb_info_failed: 'Failed to get knowledge base information',
    error_update_kb_config_failed: 'Failed to update knowledge base configuration',
    error_upload_failed: 'Upload failed',
    error_upload_timeout: 'Upload timeout',

    // Chat - UI Elements
    grid_view: 'Grid',
    tab_view: 'Tabs',
    grid_view_title: 'Grid View',
    tab_view_title: 'Tab View',
    executing: 'Executing...',
    completed: 'Completed',
    failed: 'Failed',
    processing: 'Processing...',
    upload_complete: 'Upload complete!',
    remove: 'Remove',
    copy_content: 'Copy content',
    current_kb: 'Current knowledge base',

    // Chat - File Upload
    upload_file_title: 'Upload File',
    please_select_agent_kb_first: 'Please select Agent and Knowledge Base first',
    file_overwrite_confirm: 'The following files already exist:\n\n• {files}\n\nDo you want to overwrite these files?\n\nClick "OK" to overwrite, or "Cancel" to skip these files.',

    // Knowledge Base Detail
    kb_detail_title: 'Knowledge Base Details',
    back: 'Back',
    view_config: 'View Config',
    file_association: 'File Association',
    database_association: 'Database Association',
    qa_association: 'Q&A Association & Learning',
    save_association: 'Save Association',
    build_knowledge_base: 'Build Knowledge Base',
    select_files: 'Select Files',
    search_selected_files: 'Search selected files...',
    delete_selected: 'Delete Selected ({count})',
    processing_status: 'Processing Status',
    database_type: 'Database Type',
    host_address: 'Host Address',
    port: 'Port',
    database_name: 'Database Name',
    username: 'Username',
    password: 'Password',
    sqlite_file_path: 'SQLite File Path',
    sqlite_file_path_hint: 'Please enter the full path to the SQLite file',
    test_connection_load_tables: 'Test Connection & Load Tables',
    search_table_name: 'Search table name...',
    select_tables_to_include: 'Select tables to include',
    add_selected_tables: '➕ Add Selected Tables to Knowledge Base',
    table_name: 'Table Name',
    database_type_col: 'Database Type',
    database_name_col: 'Database Name',
    select_qa_files: 'Select Q&A Files',
    search_selected_qa_files: 'Search selected Q&A files...',
    excel_format_requirement: 'Excel format: Sheet name "example", columns: question, answer, howtofind',
    select_files_modal_title: 'Select Files',
    search_files: 'Search files...',
    select_all: '✓ Select All',
    select_current_page: '✓ Current Page',
    deselect_all: '✗ Deselect',
    select_qa_excel_files: 'Select Q&A Excel Files',
    only_show_excel_files: 'ℹ️ Only showing Excel files (.xls, .xlsx)',
    confirm_selection: 'Confirm Selection',
    config_view: '⚙️ Configuration View',
    kb_config: '📦 Knowledge Base Config',
    default_yaml_config: '📄 Default YAML Config',
    build_kb_modal_title: 'Build Knowledge Base',
    building_kb_please_wait: 'Building knowledge base, please wait...',

    // KB Detail - Toast Messages
    toast_kb_id_not_found: 'Knowledge base ID not found',
    toast_kb_deleted: 'Knowledge base deleted successfully',
    toast_kb_delete_failed: 'Failed to delete knowledge base: {error}',
    toast_load_kb_failed: 'Failed to load knowledge base details: {error}',
    toast_load_files_failed_kb: 'Failed to load file list: {error}',
    toast_files_selected: 'Selected {count} file(s)',
    toast_file_removed_kb: 'File removed',
    toast_select_files_first: 'Please select files to delete first',
    toast_files_deleted: 'Deleted {count} file(s)',
    toast_db_connection_added: 'Database connection added ({count} table(s))',
    toast_db_connection_removed: 'Database connection removed',
    toast_invalid_table_index: 'Invalid table index',
    toast_db_connection_not_found: 'Database connection not found',
    toast_table_removed: 'Table removed',
    toast_select_tables_first: 'Please select tables to delete first',
    toast_tables_deleted: 'Deleted {count} table(s)',
    toast_select_at_least_one_table: 'Please select at least one table',
    toast_validating_excel: 'Validating Excel file format...',
    toast_qa_files_selected: 'Selected {count} valid Q&A file(s)',
    toast_no_valid_qa_files: 'No files matching the format requirements were selected',
    toast_validate_failed: 'Failed to validate file format: {error}',
    toast_selection_cancelled: 'Selection cancelled',
    toast_qa_file_removed: 'Q&A file removed',
    toast_select_qa_files_first: 'Please select files to delete first',
    toast_qa_files_deleted: 'Deleted {count} Q&A file(s)',
    toast_kb_id_invalid: 'Invalid knowledge base ID',
    toast_saving_config: 'Saving configuration...',
    toast_config_saved: 'Configuration saved successfully',
    toast_config_save_failed: 'Failed to save configuration: {error}',
    toast_all_files_selected: 'Selected all {count} file(s)',
    toast_current_page_selected: 'Selected {count} file(s) on current page',
    toast_all_deselected: 'All selections cancelled',

    // KB Detail - UI Elements
    loading_kb_detail: 'Loading...',
    getting_kb_info: 'Getting knowledge base information',
    no_files_found: 'No files found',
    no_files_selected_yet: 'No files selected yet',
    no_db_tables_yet: 'No database tables added yet',
    no_qa_files_yet: 'No Q&A files selected yet',
    testing_connection: 'Testing connection...',
    connection_success: '✓ Connection successful! Found {count} table(s)',
    connection_failed: '✕ Connection failed: {error}',
    no_config: 'No configuration',
    loading_config: 'Loading...',
    load_config_failed: 'Unable to load configuration file',
    load_error: 'Load failed\n\nError: {error}',
    selected_count_display: 'Selected {selected} / {total}',
    selected_count_zero: '0 / {total}',

    // Q&A Detail Page
    qa_detail_title: 'Q&A Details',
    total_qa_count: 'Total {count} items',
    knowledge_base: 'Knowledge Base',
    search_qa: 'Search questions or answers...',
    question: 'Question',
    answer: 'Answer',
    how_to_find: 'How to Find',
    source_file: 'Source File',
    created_at: 'Created At',
    learning_status: 'Learning Status',
    updated_at: 'Updated At',
    execute: 'Execute',
    batch_execute: 'Batch Execute ({count})',
    no_qa_found: 'No Q&A data found',
    qa_detail_modal_title: 'Q&A Details',
    status_pending: 'Pending',
    status_learning: 'Learning',
    status_completed: 'Completed',
    status_failed: 'Failed',
    toast_invalid_url: 'Invalid URL format',
    toast_load_qa_failed: 'Failed to load Q&A data: {error}',
    toast_qa_execution_started: 'Q&A #{id} execution started',
    toast_qa_execution_completed: 'Q&A #{id} execution completed',
    toast_select_qa_first: 'Please select Q&A items to execute first',
    toast_batch_execution_started: 'Batch execution started for {count} Q&A item(s)',
    toast_batch_execution_completed: 'Batch execution completed for {count} Q&A item(s)',
    toast_memory_not_enabled: 'Memory feature is not enabled. Please enable it first',

    // Version
    version: 'Youtu-RAG v1.0.0',
    version_number: 'v1.0.0',

    // Language
    language: 'Language',
    chinese: '中文',
    english: 'English'
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
