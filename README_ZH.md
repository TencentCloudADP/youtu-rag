<div align="center">
  <img src="docs/assets/Youtu-RAG-title.png" alt="Youtu-RAG" width="800">
</div>

<div align="center">

[![Python Version](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-online-brightgreen.svg)](docs/)
[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](CHANGELOG.md)

[English](README.md) | 简体中文 

</div>

<p align="center">
  <a href="#核心特性">✨ 核心特性</a> •
  <a href="#使用示例">📖 使用示例</a> •
  <a href="#快速开始">🚀 快速开始</a> •
  <a href="#评测基准">📊 评测基准</a>
</p>

---

`Youtu-RAG` 是新一代智能体驱动的检索增强生成系统，基于 **"本地部署 · 自主决策 · 记忆驱动"** 范式构建。具备自主决策与记忆学习能力，是个人本地知识库管理和问答系统的最佳实践。

**核心理念**：
- **本地部署**：所有组件支持本地部署，数据不出域，集成 MinIO 对象存储实现大规模文件本地化管理
- **自主决策**：智能体自主判断是否检索、如何检索、何时调用记忆，根据问题类型和历史经验选择最优策略
- **记忆驱动**：双层记忆机制（短期会话记忆 + 长期知识沉淀），实现 QA 经验的持续学习与自我进化

传统 RAG 系统遵循"离线切块-向量检索-拼接生成"的固定流程，长期面临**隐私风险、记忆缺失与检索僵化**等核心瓶颈。Youtu-RAG 旨在将系统从被动的检索工具升级为**具备自主决策与记忆学习能力的智能检索增强生成系统**。

## ✨ 核心特性

<table>
<tr>
<td width="50%" valign="top">

### 📁 文件中心化架构

以文件为核心的知识组织，支持 PDF、Excel、图片、数据库等多源异构数据接入

**支持格式**: `PDF/Word/MD` `Excel` `IMAGE` `Database` `+12种格式`

</td>
<td width="50%" valign="top">

### 🎯 智能检索引擎

自主决策最优检索策略。支持向量检索、数据库检索、元数据过滤等多种模式

**检索模式**: `向量检索` `SQL查询` `元数据过滤`

</td>
</tr>

<tr>
<td width="50%" valign="top">

### 🧠 双层记忆机制

短期会话内信息记忆 + 长期跨会话知识沉淀，实现QA经验的记忆与学习

**记忆类型**: `短期记忆` `长期记忆` `QA学习`

</td>
<td width="50%" valign="top">

### 🤖 8种开箱即用Agent

从简单对话到复杂编排，覆盖多种应用级场景。支持Web Search、KB Search、Meta Retrieval、Excel Research、Text2SQL等智能体

**应用场景**: `结构化检索` `阅读理解` `元数据检索`

</td>
</tr>

<tr>
<td width="50%" valign="top">

### 🎨 轻量级WebUI

纯原生 HTML + CSS + JavaScript 实现，无框架依赖。支持文件上传、知识库管理、AI对话、文档预览等完整功能

**技术特点**: `零依赖` `流式响应` `操作便捷`

</td>
<td width="50%" valign="top">

### 🔐 安全可控

相关组件均支持本地部署，数据不出域。集成MinIO对象存储，支持大规模文件本地化管理

**安全保障**: `本地部署` `数据隔离` `MinIO存储`

</td>
</tr>
</table>

<p align="center" style="margin-top: 40px;">
  <img src="docs/assets/Youtu-RAG.png" alt="Youtu-RAG Architecture" width="90%">
</p>

## 📖 使用示例

### 1️⃣ 文件管理

<!-- #### 文件管理配置项

修改 `configs/rag/file_management.yaml` 以应用文件管理配置，需要配置开关状态的项目如下：

```yaml
# File Management Configuration

ocr: # OCR 接入的情况下配置为 true，可对 PDF/PNG/JPG 等文件进行解析。
  enabled: true 
  model: "${UTU_OCR_MODEL}"
  base_url: "${UTU_OCR_BASE_URL}"

chunk: # Chunk 接入的情况下配置为 true，可对文本内容进行智能分块。
  enabled: true 
  model: "${UTU_CHUNK_MODEL}"
  base_url: "${UTU_CHUNK_BASE_URL}"

metadata_extraction: # 元数据提取，默认开启
  enabled: true  
  preview_length: 500  # 预览长度，默认500字符
``` -->

#### 文件上传和预览

1. 访问前端界面 `http://localhost:8000`
2. 点击左侧边栏的 **"文件管理"**
3. 点击 **"上传文件"**
4. 根据文件类型和文件管理配置，文件会经过不同路径处理并生成可预览内容


<table>
  <tr>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <strong>文件上传示例</strong><br>自动提取Metadata和摘要生成
    </td>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <strong>PDF文件处理后预览</strong><br>需要OCR配置支持
    </td>
  </tr>
  <tr>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <video src="https://github.com/miraclefish/Youtu-RAG-User-Guide/releases/download/v0.1.0/01_upload_file.mov" 
             controls muted preload="metadata" 
             width="100%" 
             style="height: 100%; max-height: 300px; object-fit: cover; border-radius: 8px; display: block;"></video>
    </td>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <video src="https://github.com/miraclefish/Youtu-RAG-User-Guide/releases/download/v0.1.0/02_pdf_file_example.mov" 
             controls muted preload="metadata" 
             width="100%" 
             style="height: 100%; max-height: 300px; object-fit: cover; border-radius: 8px; display: block;"></video>
    </td>
  </tr>
  <tr>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <strong>PNG文件处理后预览</strong><br>需要OCR配置支持
    </td>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <strong>HiChunk解析预览</strong><br>需要HiChunk配置支持
    </td>
  </tr>
  <tr>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <video src="https://github.com/miraclefish/Youtu-RAG-User-Guide/releases/download/v0.1.0/03_png_file_example.mov" 
             controls muted preload="metadata" 
             width="100%" 
             style="height: 100%; max-height: 300px; object-fit: cover; border-radius: 8px; display: block;"></video>
    </td>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <video src="https://github.com/miraclefish/Youtu-RAG-User-Guide/releases/download/v0.1.0/04_hichunk_example.mov" 
             controls muted preload="metadata" 
             width="100%" 
             style="height: 100%; max-height: 300px; object-fit: cover; border-radius: 8px; display: block;"></video>
    </td>
  </tr>
</table>

#### 文件批量管理

在 OCR 和 HiChunk 配置开启时，文档上传的解析环节将产生额外时间消耗，建议该类文件使用单文件导入方式（批量导入将会产生较长等待时间）。

<table>
  <tr>
    <td style="border: 1px solid black; padding: 10px; width: 33%; vertical-align: top;">
      <strong>文件批量删除和上传</strong><br>建议单次批量导入同类型文件
    </td>
    <td style="border: 1px solid black; padding: 10px; width: 33%; vertical-align: top;">
      <strong>文件Metadata批量编辑</strong><br>支持批量导出、编辑、导入
    </td>
    <td style="border: 1px solid black; padding: 10px; width: 33%; vertical-align: top;">
      <strong>文件搜索</strong><br>支持文件名、Metadata、摘要等
    </td>
  </tr>
  <tr style="height: 320px;">
    <td style="border: 1px solid black; padding: 10px; width: 33%; vertical-align: middle; height: 320px;">
      <video src="https://github.com/miraclefish/Youtu-RAG-User-Guide/releases/download/v0.1.0/06_batch_delete_upload.mov" 
             controls muted preload="metadata" 
             width="100%" 
             style="height: 100%; max-height: 300px; object-fit: cover; border-radius: 8px; display: block;"></video>
    </td>
    <td style="border: 1px solid black; padding: 10px; width: 33%; vertical-align: middle; height: 320px;">
      <video src="https://github.com/miraclefish/Youtu-RAG-User-Guide/releases/download/v0.1.0/05_metadata_export_import.mov" 
             controls muted preload="metadata" 
             width="100%" 
             style="height: 100%; max-height: 300px; object-fit: cover; border-radius: 8px; display: block;"></video>
    </td>
    <td style="border: 1px solid black; padding: 10px; width: 33%; vertical-align: middle; height: 320px;">
      <video src="https://github.com/miraclefish/Youtu-RAG-User-Guide/releases/download/v0.1.0/07_file_search.mov" 
             controls muted preload="metadata" 
             width="100%" 
             style="height: 100%; max-height: 300px; object-fit: cover; border-radius: 8px; display: block;"></video>
    </td>
  </tr>
</table>


### 2️⃣ 知识库管理

#### 知识库创建和删除

1. 访问前端界面 `http://localhost:8000`
2. 点击左侧边栏的 **"知识库"**
3. 点击 **"创建知识库"** 按钮
4. 填写知识库名称（如：`技术文档`）
5. 点击确认创建

<table>
  <tr>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <strong>知识库创建和删除</strong><br>仅支持单个知识库操作
    </td>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <strong>知识库搜索</strong><br>支持知识库名称、Description搜索
    </td>
  </tr>
  <tr>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <video src="https://github.com/miraclefish/Youtu-RAG-User-Guide/releases/download/v0.1.0/08_kb_create_delete.mov" 
             controls muted preload="metadata" 
             width="100%" 
             style="height: 100%; max-height: 300px; object-fit: cover; border-radius: 8px; display: block;"></video>
    </td>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <video src="https://github.com/miraclefish/Youtu-RAG-User-Guide/releases/download/v0.1.0/09_kb_search.mov" 
             controls muted preload="metadata" 
             width="100%" 
             style="height: 100%; max-height: 300px; object-fit: cover; border-radius: 8px; display: block;"></video>
    </td>
  </tr>
</table>

#### 知识库内容关联和向量化构建

1. **文件关联**: 关联已上传文件到知识库
2. **数据库关联**: 关联本地数据库到知识库
3. **示例关联**: 关联示例问答对到知识库（作为经验信息）

> 💡 **提示**：每种关联配置完都需要点击**保存关联**按钮进行关联配置保存，避免丢失之前的选中

<table>
  <tr>
    <td style="border: 1px solid black; padding: 10px; width: 33%; vertical-align: top;">
      <strong>文件关联</strong><br>单次可多选文件进行关联
    </td>
    <td style="border: 1px solid black; padding: 10px; width: 33%; vertical-align: top;">
      <strong>数据库关联</strong><br>支持Sqlite和MySQL
    </td>
    <td style="border: 1px solid black; padding: 10px; width: 33%; vertical-align: top;">
      <strong>示例关联</strong><br>支持关联示例问答对
    </td>
  </tr>
  <tr>
    <td style="border: 1px solid black; padding: 10px; width: 33%; vertical-align: top;">
      <video src="https://github.com/miraclefish/Youtu-RAG-User-Guide/releases/download/v0.1.0/10_kb_file_association.mov" 
             controls muted preload="metadata" 
             width="100%" 
             style="height: 100%; max-height: 300px; object-fit: cover; border-radius: 8px; display: block;"></video>
    </td>
    <td style="border: 1px solid black; padding: 10px; width: 33%; vertical-align: top;">
      <video src="https://github.com/miraclefish/Youtu-RAG-User-Guide/releases/download/v0.1.0/11_kb_db_association.mov" 
             controls muted preload="metadata" 
             width="100%" 
             style="height: 100%; max-height: 300px; object-fit: cover; border-radius: 8px; display: block;"></video>
    </td>
    <td style="border: 1px solid black; padding: 10px; width: 33%; vertical-align: top;">
      <video src="https://github.com/miraclefish/Youtu-RAG-User-Guide/releases/download/v0.1.0/12_kb_qa_example.mov" 
             controls muted preload="metadata" 
             width="100%" 
             style="height: 100%; max-height: 300px; object-fit: cover; border-radius: 8px; display: block;"></video>
    </td>
  </tr>
  <tr>
    <td style="border: 1px solid black; padding: 10px; width: 33%; vertical-align: top;">
      <strong>知识库配置查看</strong><br>查看关联配置和构建配置
    </td>
    <td style="border: 1px solid black; padding: 10px; width: 33%; vertical-align: top;">
      <strong>知识库向量化构建</strong><br>统一构建不同类型的关联内容
    </td>
    <td style="border: 1px solid black; padding: 10px; width: 33%; vertical-align: top;">
      <strong>知识库关联编辑</strong><br>支持关联内容的编辑和更新
    </td>
  </tr>
  <tr>
    <td style="border: 1px solid black; padding: 10px; width: 33%; vertical-align: top;">
      <video src="https://github.com/miraclefish/Youtu-RAG-User-Guide/releases/download/v0.1.0/13_kb_config_show.mov" 
             controls muted preload="metadata" 
             width="100%" 
             style="height: 100%; max-height: 300px; object-fit: cover; border-radius: 8px; display: block;"></video>
    </td>
    <td style="border: 1px solid black; padding: 10px; width: 33%; vertical-align: top;">
      <video src="https://github.com/miraclefish/Youtu-RAG-User-Guide/releases/download/v0.1.0/14_kb_build.mov" 
             controls muted preload="metadata" 
             width="100%" 
             style="height: 100%; max-height: 300px; object-fit: cover; border-radius: 8px; display: block;"></video>
    </td>
    <td style="border: 1px solid black; padding: 10px; width: 33%; vertical-align: top;">
      <video src="https://github.com/miraclefish/Youtu-RAG-User-Guide/releases/download/v0.1.0/15_kb_modify.mov" 
             controls muted preload="metadata" 
             width="100%" 
             style="height: 100%; max-height: 300px; object-fit: cover; border-radius: 8px; display: block;"></video>
    </td>
  </tr>
</table>


### 3️⃣ 智能对话

1. 可针对不同任务选择已配置的 Agent 进行对话或问答：
   - 部分智能体必须选定知识库或文件才可用
   - 提供临时上传文件按钮，支持临时上传文件进行问答，但该文件只会自动关联当前知识库，并不会进行向量构建
  
2. 在前端对话界面中, 打开右下角 **“记忆”** 开关，即可启用双层记忆机制。开启记忆后，Agent将具备：
   - **短期记忆**：记住对话上下文，避免重复提问
   - **长期记忆**：沉淀成功经验，下次遇到相似问题时优先复用


<table>
  <tr>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <strong>💬 Chat Agent</strong><ul style="margin: 5px 0 0 0; padding-left: 20px;">
        <li>闲聊 Agent</li>
        <li>建议开启“记忆”以支持多轮对话</li>
      </ul>
    </td>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <strong>🔍 Web Search Agent</strong><ul style="margin: 5px 0 0 0; padding-left: 20px;">
        <li>支持网页搜索</li>
        <li>可访问链接探查详细内容并回答</li>
      </ul>
    </td>
  </tr>
  <tr>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <video src="https://github.com/miraclefish/Youtu-RAG-User-Guide/releases/download/v0.1.0/16_chat_agent.mov" 
             controls muted preload="metadata" 
             width="100%" 
             style="height: 100%; max-height: 300px; object-fit: cover; border-radius: 8px; display: block;"></video>
    </td>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <video src="https://github.com/miraclefish/Youtu-RAG-User-Guide/releases/download/v0.1.0/17_websearch_agent.mov" 
             controls muted preload="metadata" 
             width="100%" 
             style="height: 100%; max-height: 300px; object-fit: cover; border-radius: 8px; display: block;"></video>
    </td>
  </tr>
  <tr>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <strong>📚 KB Search Agent</strong><ul style="margin: 5px 0 0 0; padding-left: 20px;">
        <li>必须选择知识库</li>
        <li>支持向量检索和重排</li>
      </ul>
    </td>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <strong>📚 Meta Retrieval Agent</strong><ul style="margin: 5px 0 0 0; padding-left: 20px;">
        <li>必须选择知识库</li>
        <li>支持向量检索和重排</li>
        <li>支持问题意图解析和元数据过滤</li>
      </ul>
    </td>
  </tr>
  <tr>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <video src="https://github.com/miraclefish/Youtu-RAG-User-Guide/releases/download/v0.1.0/18_kbsearch_agent.mov" 
             controls muted preload="metadata" 
             width="100%" 
             style="height: 100%; max-height: 300px; object-fit: cover; border-radius: 8px; display: block;"></video>
    </td>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <video src="https://github.com/miraclefish/Youtu-RAG-User-Guide/releases/download/v0.1.0/19_meta_retrieval_agent.mov" 
             controls muted preload="metadata" 
             width="100%" 
             style="height: 100%; max-height: 300px; object-fit: cover; border-radius: 8px; display: block;"></video>
    </td>
  </tr>
  <tr>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <strong>📄 File QA Agent</strong><ul style="margin: 5px 0 0 0; padding-left: 20px;">
        <li>必须选择知识库和文件</li>
        <li>支持Python读取和处理文件内容</li>
        <li>支持向量检索和重排</li>
      </ul>
    </td>
    </td>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <strong>📊 Excel Agent</strong><ul style="margin: 5px 0 0 0; padding-left: 20px;">
        <li>必须选择知识库和文件</li>
        <li>问题拆解和数据处理步骤拆解</li>
        <li>Python代码执行和反思</li>
      </ul>
    </td>
  </tr>
  <tr>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <video src="https://github.com/miraclefish/Youtu-RAG-User-Guide/releases/download/v0.1.0/20_file_qa_agent.mp4" 
             controls muted preload="metadata" 
             width="100%" 
             style="height: 100%; max-height: 300px; object-fit: cover; border-radius: 8px; display: block;"></video>
    </td>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <video src="https://github.com/miraclefish/Youtu-RAG-User-Guide/releases/download/v0.1.0/22_excel_agent.mp4" 
             controls muted preload="metadata" 
             width="100%" 
             style="height: 100%; max-height: 300px; object-fit: cover; border-radius: 8px; display: block;"></video>
    </td>
  </tr>
  <tr>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <strong>💻 Text2SQL Agent</strong><ul style="margin: 5px 0 0 0; padding-left: 20px;">
        <li>必须选择已关联数据库的知识库</li>
        <li>问题拆解与SQL代码生成和执行</li>
        <li>SQL查询结果展示和反思</li>
      </ul>
    </td>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <strong>🧠 Short and Long-Term Memory</strong><ul style="margin: 5px 0 0 0; padding-left: 20px;">
        <li>Short-term memory: Session内生效，用于支持多轮对话</li>
        <li>Long-term memory: 长期生效，用于沉淀成功经验</li>
      </ul>
    </td>
  </tr>
  <tr>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <video src="https://github.com/miraclefish/Youtu-RAG-User-Guide/releases/download/v0.1.0/21_text2sql_agent.mov" 
             controls muted preload="metadata" 
             width="100%" 
             style="height: 100%; max-height: 300px; object-fit: cover; border-radius: 8px; display: block;"></video>
    </td>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <video src="https://github.com/miraclefish/Youtu-RAG-User-Guide/releases/download/v0.1.0/23_memory_chat_2.mov" 
             controls muted preload="metadata" 
             width="100%" 
             style="height: 100%; max-height: 300px; object-fit: cover; border-radius: 8px; display: block;"></video>
    </td>
  </tr>
  <tr>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <strong>🧐 Text2SQL Agent with Memory</strong><ul style="margin: 5px 0 0 0; padding-left: 20px;">
        <li>短期记忆在Session内生效</li>
        <li>长期记忆可避免相似问题的额外token消耗</li>
      </ul>
    </td>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <strong>🎯 QA Learning</strong><ul style="margin: 5px 0 0 0; padding-left: 20px;">
        <li>记录 QA 示例</li>
        <li>自动学习 Agent 路由策略</li>
      </ul>
    </td>
  </tr>
  <tr>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <video src="https://github.com/miraclefish/Youtu-RAG-User-Guide/releases/download/v0.1.0/25_text2sql_memory.mov" 
             controls muted preload="metadata" 
             width="100%" 
             style="height: 100%; max-height: 300px; object-fit: cover; border-radius: 8px; display: block;"></video>
    </td>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <video src="https://github.com/miraclefish/Youtu-RAG-User-Guide/releases/download/v0.1.0/26_qa_learning.mov" 
             controls muted preload="metadata" 
             width="100%" 
             style="height: 100%; max-height: 300px; object-fit: cover; border-radius: 8px; display: block;"></video>
    </td>
  </tr>
</table>

## 🚀 快速开始

### 环境要求

- **Python**：3.12+
- **包管理器**：推荐使用 [uv](https://github.com/astral-sh/uv)
- **操作系统**：Linux桌面版 / macOS / Windows

### 📦 对象存储 (MinIO) 配置

MinIO是一个高性能对象存储服务，用于存储上传的文档文件（依然是本地管理）。

安装说明请参考官方 [MinIO 仓库](https://github.com/minio/minio)。支持两种安装方式：
- **从源码安装**：从源码构建并安装MinIO
- **构建Docker镜像**：使用Docker容器部署MinIO

### ⚙️ 模型部署

| 模型 | HuggingFace | 部署方法 | 是否必选 |
|:---|:---|:---|:---:|
| [Youtu-Embedding ](https://github.com/TencentCloudADP/youtu-embedding) | [HuggingFace](https://huggingface.co/tencent/Youtu-Embedding) | [部署文档](https://xxxxxx) | ✅ 必选，或其他API服务 |
| [Youtu-Parsing](https://github.com/TencentCloudADP/youtu-parsing) | [HuggingFace](https://huggingface.co/spaces/tencent/Youtu-Parsing) | [部署文档](https://xxxxxx) | ⭕ 可选 |
| [Youtu-HiChunk](https://github.com/TencentCloudADP/hichunk) | [HuggingFace](https://huggingface.co/spaces/tencent/HiChunk) | [部署文档](https://xxxx) | ⭕ 可选 |


### 一键安装 Youtu-RAG 系统

```bash
git clone https://github.com/TencentCloudADP/youtu-rag.git
cd youtu-rag
uv sync
source .venv/bin/activate
cp .env.example .env
```

### 配置必要的环境变量

编辑 `.env` 文件，填写以下核心配置：

```bash
# =============================================
# LLM配置（必填）
# =============================================
UTU_LLM_TYPE=chat.completions
UTU_LLM_MODEL=deepseek-chat
UTU_LLM_BASE_URL=https://api.deepseek.com/v1
UTU_LLM_API_KEY=your_deepseek_api_key  # 替换为你的API Key

# =============================================
# Embedding配置（必填）
# =============================================
# Option 1: 本地服务（Youtu-Embedding）
UTU_EMBEDDING_URL=http://localhost:8081
UTU_EMBEDDING_MODEL=youtu-embedding-2B

# Option 2: 其他API服务
# UTU_EMBEDDING_URL=https://api.your-embedding-service.com
# UTU_EMBEDDING_API_KEY=your_api_key
# UTU_EMBEDDING_MODEL=model_name

# =============================================
# Reranker配置（可选，提升检索精度）
# =============================================
UTU_RERANKER_MODEL=jina-reranker-v3
UTU_RERANKER_URL=https://api.jina.ai/v1/rerank
UTU_RERANKER_API_KEY=your_jina_api_key 

# =============================================
# OCR配置（可选，可本地部署Youtu-Parsing）
# =============================================
UTU_OCR_BASE_URL=https://api.ocr.com/ocr
UTU_OCR_MODEL=ocr

# =============================================
# Chunk配置（可选，可本地部署Youtu-HiChunk）
# =============================================
UTU_CHUNK_BASE_URL=https://api.hichunk.com/chunk
UTU_CHUNK_MODEL=hichunk

# =============================================
# 记忆功能（可选）
# =============================================
memoryEnabled=false  # 设置为true启用双层记忆机制
```

> **提示**：如果不需要使用 OCR 和 Chunk 功能，可以在 [configs/rag/file_management.yaml](configs/rag/file_management.yaml) 中设置 `ocr enabled: false` 和 `chunk enabled: false` 来禁用这些功能。

### 启动服务

```bash
# 方式1：使用启动脚本（推荐）
bash start.sh

# 方式2：直接使用uvicorn
uv run uvicorn utu.rag.api.main:app --reload --host 0.0.0.0 --port 8000
```

启动成功后，访问以下地址：

- 📱 前端界面: http://localhost:8000
- 📊 监控面板: http://localhost:8000/monitor

---

## 📊 评测基准

Youtu-RAG提供完整的评测体系，支持多维度能力验证。

### 🗄️ 结构化检索（Text2SQL）

**能力**：自然语言转SQL、Schema理解、SQL执行
**数据集**：自建Text2SQL数据集 (Multi-table、Complex excel、Domain table)
**指标**：Accuracy (LLM Judge)

<div align="center">

| 数据集概况 | 数据集 | Multi-table-mini | Complex Excel | Multi-table | Domain Table |
|:---:|:---|:---:|:---:|:---:|:---:|
| | **数据量** | 245 | 931 | 1,390 | 100 |
| | **类型** | 多表 | 复杂问题 | 多表全量 | 专业知识 |
| **Baseline** | Vanna | 45.71% | 38.64% | 35.11% | 9.00% |
| **🎯 Youtu-RAG** | **Text2SQL Agent** | **69.39%** ↑ | **57.36%** ↑ | **67.27%** ↑ | **27.00%** ↑ |

</div>

---

### 📊 半结构化检索（Excel）

**能力**：表格理解、数据分析、非标准表格解析
**数据集**：自建Excel问答数据集（500条测试问题）
**指标**：LLM Judge
- **Accuracy**: 答案的事实正确性
- **Analysis Depth**: 答案的分析质量和洞察力
- **Feasibility**: 生成的代码/方案是否可执行
- **Aesthetics**: 可视化图表的视觉质量

<div align="center">

| 类别 | 方法 | Accuracy | Analysis Depth | Feasibility | Aesthetics |
|:---:|:---|:---:|:---:|:---:|:---:|
| **Baselines** | TableGPT2-7B | 8.4 | 5.1 | 4.3 | 6.2 |
| | StructGPT | 6.22 | 3.84 | 3.12 | 4.5 |
| | TableLLM-7B | 4.1 | 2.1 | 1.8 | 2.3 |
| | ST-Raptor | 22.4 | 6.0 | 7.4 | 12.4 |
| | TreeThinker | 31.0 | 22.8 | 21.4 | 36.8 |
| | Code Loop | 27.5 | 9.5 | 14.9 | 20.4 |
| **🎯 Youtu-RAG** | **Excel Agent** | **37.5** ↑ | **30.2** ↑ | **27.6** ↑ | **42.6** ↑ |

</div>

---

### 📖 阅读理解（长文本）

**[FactGuard](https://arxiv.org/pdf/2504.05607)**：长文档单点事实核查、信息抽取、推理验证
**[Sequential-NIAH](https://aclanthology.org/2025.emnlp-main.1497.pdf)**：长文档多点信息抽取、顺序信息提取

<div align="center">

| 数据集概况 | 数据集 | FactGuard | Sequential-NIAH |
|:---:|:---|:---:|:---:|
| | **数据量** | 700 | 2,000 |
| | **类型** | 长文本问答（单点） | 长文本问答（多点） |
| **Baselines** | Naive Retrieval Top3 | 79.86% | 14.20% |
| | Naive Retrieval Top5 | 80.71% | 29.75% |
| | Naive Retrieval Top10 | 82.71% | 57.25% |
| | Naive Retrieval Top15 | 83.00% | 70.15% |
| **🎯 Youtu-RAG** | **KB Search Agent** | **88.27%** ↑ | **85.05%** ↑ |
| | **File QA Agent** | **88.29%** ↑ | **60.80%** * |

</div>

> **说明**：*长上下文环境下阅读全文是LLM的已知弱点，这与Sequential-NIAH的实验发现一致。File QA Agent在多点提取任务上的性能反映了这一局限性，而KB Search Agent基于检索的方法取得了显著更好的结果。

---

### 🏷️ 元数据检索

**能力**：问题偏好理解、元数据过滤和重排、向量检索
**数据集**：自建元数据检索数据集
**指标**：
- **Weighted NDCG@5**: 在前5个检索结果中，按准确顺序召回真实相关文档的能力指标
- **Recall@all**: 所有的真实的相关文档中有多少被准确召回

<div align="center">

| 数据集 | 数据量 | 指标 | Baseline<br/>(Naive Retrieval) | Youtu-RAG<br/>(Meta Retrieval Agent) | 提升幅度 |
|:---|:---:|:---|:---:|:---:|:---:|
| **时效性偏好** | 183 | Recall@all | 34.52% | **41.92%** | +7.40% ↑ |
| | | NDCG_w@5 | 29.91% | **43.57%** | +13.66% ↑ |
| **热度偏好** | 301 | Recall@all | 26.19% | **47.20%** | +21.01% ↑ |
| | | NDCG_w@5 | 29.86% | **54.31%** | +24.45% ↑ |
| **平均** | 483 | Recall@all | 29.34% | **45.21%** | +15.87% ↑ |
| | | NDCG_w@5 | 29.88% | **50.25%** | +20.37% ↑ |

</div>
  

### Memoria-Bench（审核中，待发布）

**Memoria-Bench** 是业内首个区分**语义记忆、情节记忆、程序记忆**，并适配**深度研究、表格问答、复杂代码分析补全**等高信息密度场景的智能体记忆评估基准。

**核心特性**：
- 📚 **语义记忆评测**：知识理解与应用
- 📖 **情节记忆评测**：历史对话回溯
- 🔧 **程序记忆评测**：技能学习与复用
- 🎯 **场景覆盖**：研究报告生成、数据分析、代码补全

> 💡 **提示**：Memoria-Bench评测基准正在审核中，敬请期待！


## 🤝 贡献指南

我们欢迎任何形式的贡献！包括但不限于：
<ul>
<li>🐛 报告Bug和问题</li>
<li>💡 提出新功能建议</li>
<li>📝 改进文档</li>
<li>🔧 提交代码改进</li>
</ul>

详细的开发流程和规范请参考 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 📄 许可证

本项目采用 [MIT License](LICENSE) 开源协议。

## 🙏 致谢

Youtu-RAG 基于多个开源项目的卓越成果构建而成：

- **[Youtu-Agent](https://github.com/TencentCloudADP/youtu-agent)**：智能体框架
- **[Youtu-LLM](https://github.com/TencentCloudADP/youtu-tip/tree/master/youtu-llm)**：LLM基座
- **[Youtu-Embedding](https://github.com/TencentCloudADP/youtu-embedding)**：中文向量编码器
- **[Youtu-Parsing](https://github.com/TencentCloudADP/youtu-parsing)**：文档解析模型
- **[Youtu-HiChunk](https://github.com/TencentCloudADP/hichunk)**：文档分层模型

特别感谢所有为本项目贡献代码、提出建议和报告问题的开发者！

## 📞 联系我们

- **GitHub Issues**：[提交Issue](https://github.com/TencentCloudADP/youtu-rag/issues)
- **讨论区**：[GitHub Discussions](https://github.com/TencentCloudADP/youtu-rag/discussions)
- **文档**：[在线文档](docs/)

## 📚 引用

如果本项目对您的研究或工作有帮助，欢迎引用：

```bibtex
@software{Youtu-RAG,
  author = {Tencent Youtu Lab},
  title = {Youtu-RAG: Next-Generation Agentic Intelligent Retrieval-Augmented Generation System},
  year = {2026},
  url = {https://github.com/TencentCloudADP/youtu-rag}
}
```

---

<div align="center">

**⭐ 如果这个项目对您有帮助，请给我们一个Star！**

</div>
