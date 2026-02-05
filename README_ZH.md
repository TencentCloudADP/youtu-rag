<div align="center">
  <a href="https://youtu-rag-docs.vercel.app" target="_blank">
    <img src="docs/assets/Youtu-RAG-title.png" alt="Youtu-RAG" width="800">
  </a>
</div>

<div align="center">

[![Python Version](https://img.shields.io/badge/python-3.12+-blue.svg)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-online-brightgreen.svg)](https://youtu-rag-docs.vercel.app)
[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)]()

[English](README.md) | 简体中文 

</div>

<p align="center">
  <a href="#核心特性">✨ 核心特性</a> •
  <a href="#使用示例">📖 使用示例</a> •
  <a href="#快速开始">🚀 快速开始</a> •
  <a href="#评测基准">📊 评测基准</a>
</p>

---

**[Youtu-RAG](https://youtu-rag-docs.vercel.app)** 是新一代智能体驱动的检索增强生成系统，基于 **"本地部署 · 自主决策 · 记忆驱动"** 范式构建。具备自主决策与记忆学习能力，是个人本地知识库管理和问答系统的最佳实践。

**核心理念**：
- **本地部署**：所有组件支持本地部署，数据不出域，集成 MinIO 对象存储实现大规模文件本地化管理
- **自主决策**：智能体自主判断是否检索、如何检索、何时调用记忆，根据问题类型和历史经验选择最优策略
- **记忆驱动**：双层记忆机制（短期会话记忆 + 长期知识沉淀），实现 QA 经验的持续学习与自我进化

传统 RAG 系统遵循"离线切块-向量检索-拼接生成"的固定流程，长期面临**隐私风险、记忆缺失与检索僵化**等核心瓶颈。Youtu-RAG 旨在将系统从被动的检索工具升级为**具备自主决策与记忆学习能力的智能检索增强生成系统**。

<p align="center" style="margin-top: 40px;">
  <img src="docs/assets/Youtu-RAG.png" alt="Youtu-RAG Architecture" width="100%">
</p>

<a id="核心特性"></a>
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

自主决策最优检索策略，支持网络搜索、向量检索、元数据过滤、数据库查询、代码执行等丰富的工具调用

**检索模式**: `自主决策` `工具调用` `多样化数据源`

</td>
</tr>

<tr>
<td width="50%" valign="top">

### 🧠 双层记忆机制

短期会话内信息记忆 + 长期跨会话知识沉淀，实现QA经验的记忆与学习

**记忆类型**: `短期记忆` `长期记忆` `QA学习`

</td>
<td width="50%" valign="top">

### 🤖 开箱即用Agent

从简单对话到复杂编排，覆盖多种应用级场景。支持Web Search、KB Search、Meta Retrieval、Excel Agent、Text2SQL等8+智能体

**应用场景**: `开箱即用` `多样化场景` `复杂任务协同`

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


<a id="使用示例"></a>
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
      <video src="https://github.com/user-attachments/assets/d8247dc2-f134-46da-9fd1-15c990445011" 
             controls muted preload="metadata" 
             width="100%" 
             style="height: 100%; max-height: 300px; object-fit: cover; border-radius: 8px; display: block;"></video>
    </td>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <video src="https://github.com/user-attachments/assets/e5568428-990a-4008-8808-dcbe80cb2757" 
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
      <video src="https://github.com/user-attachments/assets/7701152f-b55e-46c1-af33-ebafd1b2341e" 
             controls muted preload="metadata" 
             width="100%" 
             style="height: 100%; max-height: 300px; object-fit: cover; border-radius: 8px; display: block;"></video>
    </td>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <video src="https://github.com/user-attachments/assets/815cd47a-5137-483b-b4e7-f5205d7d0b03" 
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
      <video src="https://github.com/user-attachments/assets/60e01dc6-58db-4f8d-bb3d-4a259f34f741" 
             controls muted preload="metadata" 
             width="100%" 
             style="height: 100%; max-height: 300px; object-fit: cover; border-radius: 8px; display: block;"></video>
    </td>
    <td style="border: 1px solid black; padding: 10px; width: 33%; vertical-align: middle; height: 320px;">
      <video src="https://github.com/user-attachments/assets/30de1091-68d4-4306-99da-e64e9e87329c" 
             controls muted preload="metadata" 
             width="100%" 
             style="height: 100%; max-height: 300px; object-fit: cover; border-radius: 8px; display: block;"></video>
    </td>
    <td style="border: 1px solid black; padding: 10px; width: 33%; vertical-align: middle; height: 320px;">
      <video src="https://github.com/user-attachments/assets/b0bd0b20-0be9-4dbc-9c11-a558229b2e45" 
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
      <video src="https://github.com/user-attachments/assets/162ef8e4-ae3f-44dd-8389-9bbdb9640bc2" 
             controls muted preload="metadata" 
             width="100%" 
             style="height: 100%; max-height: 300px; object-fit: cover; border-radius: 8px; display: block;"></video>
    </td>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <video src="https://github.com/user-attachments/assets/9fafc311-6333-4759-844e-47ba1054a66a" 
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
      <video src="https://github.com/user-attachments/assets/ae1d0bb9-080f-4813-b9f0-32c30cf2e84c" 
             controls muted preload="metadata" 
             width="100%" 
             style="height: 100%; max-height: 300px; object-fit: cover; border-radius: 8px; display: block;"></video>
    </td>
    <td style="border: 1px solid black; padding: 10px; width: 33%; vertical-align: top;">
      <video src="https://github.com/user-attachments/assets/f9d013e6-3e8f-46a6-95ac-43b66bb389f0" 
             controls muted preload="metadata" 
             width="100%" 
             style="height: 100%; max-height: 300px; object-fit: cover; border-radius: 8px; display: block;"></video>
    </td>
    <td style="border: 1px solid black; padding: 10px; width: 33%; vertical-align: top;">
      <video src="https://github.com/user-attachments/assets/404cbdc9-a053-423c-859a-4f28c3fbabfb" 
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
      <video src="https://github.com/user-attachments/assets/500b3b31-42c6-491a-846f-e6b23ad19dc4" 
             controls muted preload="metadata" 
             width="100%" 
             style="height: 100%; max-height: 300px; object-fit: cover; border-radius: 8px; display: block;"></video>
    </td>
    <td style="border: 1px solid black; padding: 10px; width: 33%; vertical-align: top;">
      <video src="https://github.com/user-attachments/assets/69118200-2f94-4d89-945c-1d20aac4d2a6" 
             controls muted preload="metadata" 
             width="100%" 
             style="height: 100%; max-height: 300px; object-fit: cover; border-radius: 8px; display: block;"></video>
    </td>
    <td style="border: 1px solid black; padding: 10px; width: 33%; vertical-align: top;">
      <video src="https://github.com/user-attachments/assets/d7f96a7a-2c6c-488d-93b8-ddc93eaf662f" 
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
      <video src="https://github.com/user-attachments/assets/f65b6989-1af8-4304-b5b9-95fdd1cb217e" 
             controls muted preload="metadata" 
             width="100%" 
             style="height: 100%; max-height: 300px; object-fit: cover; border-radius: 8px; display: block;"></video>
    </td>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <video src="https://github.com/user-attachments/assets/e16935de-0e1b-4b46-922b-32d588c58939" 
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
      <video src="https://github.com/user-attachments/assets/33d7d44c-289f-47fe-881b-ceb237932218" 
             controls muted preload="metadata" 
             width="100%" 
             style="height: 100%; max-height: 300px; object-fit: cover; border-radius: 8px; display: block;"></video>
    </td>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <video src="https://github.com/user-attachments/assets/e3ffbddd-4638-4e94-8112-38458c847a83" 
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
      <video src="https://github.com/user-attachments/assets/0d1d3b35-4fad-4122-aafc-3209c4cd6efd" 
             controls muted preload="metadata" 
             width="100%" 
             style="height: 100%; max-height: 300px; object-fit: cover; border-radius: 8px; display: block;"></video>
    </td>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <video src="https://github.com/user-attachments/assets/da9be8d8-fdd8-4481-91bd-fe08c3a6bbdb" 
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
      <video src="https://github.com/user-attachments/assets/30fff32d-066b-4f0b-a444-7e9b9e3932fa" 
             controls muted preload="metadata" 
             width="100%" 
             style="height: 100%; max-height: 300px; object-fit: cover; border-radius: 8px; display: block;"></video>
    </td>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <video src="https://github.com/user-attachments/assets/fd98f130-7d8d-457e-96a5-1cf43c8daf81" 
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
      <video src="https://github.com/user-attachments/assets/93403424-82e8-4cc9-b035-bec977822a1f" 
             controls muted preload="metadata" 
             width="100%" 
             style="height: 100%; max-height: 300px; object-fit: cover; border-radius: 8px; display: block;"></video>
    </td>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <video src="https://github.com/user-attachments/assets/59d74642-5d0b-4f00-acd0-d2d346035348" 
             controls muted preload="metadata" 
             width="100%" 
             style="height: 100%; max-height: 300px; object-fit: cover; border-radius: 8px; display: block;"></video>
    </td>
  </tr>
</table>

<a id="快速开始"></a>
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
| [Youtu-Embedding ](https://github.com/TencentCloudADP/youtu-embedding) | [HuggingFace](https://huggingface.co/tencent/Youtu-Embedding) | [部署文档](https://youtu-rag-docs.vercel.app/docs/zh/youtu-embedding/deploying-locally) | ✅ 必选，或其他 Embedding API 服务 |
| [Youtu-Parsing](https://github.com/TencentCloudADP/youtu-parsing) | [HuggingFace](https://huggingface.co/tencent/Youtu-Parsing) | [部署文档](https://youtu-rag-docs.vercel.app/docs/zh/youtu-parsing/deploying-locally) | ⭕ 可选 |
| [Youtu-HiChunk](https://github.com/TencentCloudADP/hichunk) | [HuggingFace](https://huggingface.co/tencent/Youtu-HiChunk) | [部署文档](https://youtu-rag-docs.vercel.app/docs/zh/hichunk/deploying-locally) | ⭕ 可选 |


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

# Option 2: 其他Embedding API服务
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

<a id="评测基准"></a>
## 📊 评测基准

Youtu-RAG提供完整的评测体系，支持多维度能力验证。

### 🗄️ 结构化检索（Text2SQL）

- **能力**：自然语言转SQL、Schema理解、SQL执行
- **数据集**：自建Text2SQL数据集 (Multi-table、Complex excel、Domain table)
- **指标**：Accuracy (LLM Judge)

<div align="center">

<table>
<thead>
<tr>
<th rowspan="3" align="center">数据集概况</th>
<th align="left">数据集</th>
<th align="center">Multi-table-mini</th>
<th align="center">Complex Excel</th>
<th align="center">Multi-table</th>
<th align="center">Domain Table</th>
</tr>
<tr>
<th align="left"><strong>数据量</strong></th>
<th align="center">245</th>
<th align="center">931</th>
<th align="center">1,390</th>
<th align="center">100</th>
</tr>
<tr>
<th align="left"><strong>类型</strong></th>
<th align="center">多表</th>
<th align="center">复杂问题</th>
<th align="center">多表全量</th>
<th align="center">专业知识</th>
</tr>
</thead>
<tbody>
<tr>
<td align="center"><strong>Baseline</strong></td>
<td align="left">Vanna</td>
<td align="center">45.71%</td>
<td align="center">38.64%</td>
<td align="center">35.11%</td>
<td align="center">9.00%</td>
</tr>
<tr>
<td align="center"><strong>🎯 Youtu-RAG</strong></td>
<td align="left"><strong>Text2SQL Agent</strong></td>
<td align="center"><strong>69.39%</strong> ↑</td>
<td align="center"><strong>57.36%</strong> ↑</td>
<td align="center"><strong>67.27%</strong> ↑</td>
<td align="center"><strong>27.00%</strong> ↑</td>
</tr>
</tbody>
</table>

</div>

---

### 📊 半结构化检索（Excel）

- **能力**：表格理解、数据分析、非标准表格解析
- **数据集**：自建Excel问答数据集（500条测试问题）
- **指标**：LLM Judge
  - **Accuracy**: 答案的事实正确性
  - **Analysis Depth**: 答案的分析质量和洞察力
  - **Feasibility**: 生成的代码/方案是否可执行
  - **Aesthetics**: 可视化图表的视觉质量

<div align="center">

<table>
<thead>
<tr>
<th align="center">类别</th>
<th align="left">方法</th>
<th align="center">Accuracy</th>
<th align="center">Analysis Depth</th>
<th align="center">Feasibility</th>
<th align="center">Aesthetics</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="6" align="center"><strong>Baselines</strong></td>
<td align="left">TableGPT2-7B</td>
<td align="center">8.4</td>
<td align="center">5.1</td>
<td align="center">4.3</td>
<td align="center">6.2</td>
</tr>
<tr>
<td align="left">StructGPT</td>
<td align="center">6.22</td>
<td align="center">3.84</td>
<td align="center">3.12</td>
<td align="center">4.5</td>
</tr>
<tr>
<td align="left">TableLLM-7B</td>
<td align="center">4.1</td>
<td align="center">2.1</td>
<td align="center">1.8</td>
<td align="center">2.3</td>
</tr>
<tr>
<td align="left">ST-Raptor</td>
<td align="center">22.4</td>
<td align="center">6.0</td>
<td align="center">7.4</td>
<td align="center">12.4</td>
</tr>
<tr>
<td align="left">TreeThinker</td>
<td align="center">31.0</td>
<td align="center">22.8</td>
<td align="center">21.4</td>
<td align="center">36.8</td>
</tr>
<tr>
<td align="left">Code Loop</td>
<td align="center">27.5</td>
<td align="center">9.5</td>
<td align="center">14.9</td>
<td align="center">20.4</td>
</tr>
<tr>
<td align="center"><strong>🎯 Youtu-RAG</strong></td>
<td align="left"><strong>Excel Agent</strong></td>
<td align="center"><strong>37.5</strong> ↑</td>
<td align="center"><strong>30.2</strong> ↑</td>
<td align="center"><strong>27.6</strong> ↑</td>
<td align="center"><strong>42.6</strong> ↑</td>
</tr>
</tbody>
</table>

</div>

---

### 📖 阅读理解（长文本）

- **[FactGuard](https://arxiv.org/pdf/2504.05607)**：长文档单点事实核查、信息抽取、推理验证
- **[Sequential-NIAH](https://aclanthology.org/2025.emnlp-main.1497.pdf)**：长文档多点信息抽取、顺序信息提取

<div align="center">

<table>
<thead>
<tr>
<th rowspan="3" align="center">数据集概况</th>
<th align="left">数据集</th>
<th align="center">FactGuard</th>
<th align="center">Sequential-NIAH</th>
</tr>
<tr>
<th align="left"><strong>数据量</strong></th>
<th align="center">700</th>
<th align="center">2,000</th>
</tr>
<tr>
<th align="left"><strong>类型</strong></th>
<th align="center">长文本问答（单点）</th>
<th align="center">长文本问答（多点）</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="4" align="center"><strong>Baselines</strong></td>
<td align="left">Naive Retrieval Top3</td>
<td align="center">79.86%</td>
<td align="center">14.20%</td>
</tr>
<tr>
<td align="left">Naive Retrieval Top5</td>
<td align="center">80.71%</td>
<td align="center">29.75%</td>
</tr>
<tr>
<td align="left">Naive Retrieval Top10</td>
<td align="center">82.71%</td>
<td align="center">57.25%</td>
</tr>
<tr>
<td align="left">Naive Retrieval Top15</td>
<td align="center">83.00%</td>
<td align="center">70.15%</td>
</tr>
<tr>
<td rowspan="2" align="center"><strong>🎯 Youtu-RAG</strong></td>
<td align="left"><strong>KB Search Agent</strong></td>
<td align="center"><strong>88.27%</strong> ↑</td>
<td align="center"><strong>85.05%</strong> ↑</td>
</tr>
<tr>
<td align="left"><strong>File QA Agent</strong></td>
<td align="center"><strong>88.29%</strong> ↑</td>
<td align="center"><strong>60.80%</strong> *</td>
</tr>
</tbody>
</table>

</div>

> **说明**：*长上下文环境下阅读全文是LLM的已知弱点，这与Sequential-NIAH的实验发现一致。File QA Agent在多点提取任务上的性能反映了这一局限性，而KB Search Agent基于检索的方法取得了显著更好的结果。

---

### 🏷️ 元数据检索

- **能力**：问题偏好理解、元数据过滤和重排、向量检索
- **数据集**：自建元数据检索数据集
- **指标**：
  - **Weighted NDCG@5**: 在前5个检索结果中，按准确顺序召回真实相关文档的能力指标
  - **Recall@all**: 所有的真实的相关文档中有多少被准确召回

<div align="center">

<table>
<thead>
<tr>
<th align="left">数据集</th>
<th align="center">数据量</th>
<th align="left">指标</th>
<th align="center">Baseline<br/>(Naive Retrieval)</th>
<th align="center">Youtu-RAG<br/>(Meta Retrieval Agent)</th>
<th align="center">提升幅度</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="2" align="left"><strong>时效性偏好</strong></td>
<td rowspan="2" align="center">183</td>
<td align="left">Recall@all</td>
<td align="center">34.52%</td>
<td align="center"><strong>41.92%</strong></td>
<td align="center">+7.40% ↑</td>
</tr>
<tr>
<td align="left">NDCG_w@5</td>
<td align="center">29.91%</td>
<td align="center"><strong>43.57%</strong></td>
<td align="center">+13.66% ↑</td>
</tr>
<tr>
<td rowspan="2" align="left"><strong>热度偏好</strong></td>
<td rowspan="2" align="center">301</td>
<td align="left">Recall@all</td>
<td align="center">26.19%</td>
<td align="center"><strong>47.20%</strong></td>
<td align="center">+21.01% ↑</td>
</tr>
<tr>
<td align="left">NDCG_w@5</td>
<td align="center">29.86%</td>
<td align="center"><strong>54.31%</strong></td>
<td align="center">+24.45% ↑</td>
</tr>
<tr>
<td rowspan="2" align="left"><strong>平均</strong></td>
<td rowspan="2" align="center">483</td>
<td align="left">Recall@all</td>
<td align="center">29.34%</td>
<td align="center"><strong>45.21%</strong></td>
<td align="center">+15.87% ↑</td>
</tr>
<tr>
<td align="left">NDCG_w@5</td>
<td align="center">29.88%</td>
<td align="center"><strong>50.25%</strong></td>
<td align="center">+20.37% ↑</td>
</tr>
</tbody>
</table>

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
- **[FactGuard](https://arxiv.org/pdf/2504.05607)**：（Benchmark）长文档单点事实核查、信息抽取、推理验证
- **[Sequential-NIAH](https://aclanthology.org/2025.emnlp-main.1497.pdf)**：（Benchmark）长文档多点信息抽取、顺序信息提取

特别感谢所有为本项目贡献代码、提出建议和报告问题的开发者！

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
