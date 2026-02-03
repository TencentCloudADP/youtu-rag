# Quickstart

This guide will walk you through setting up Youtu-RAG, configuring necessary services, and starting the system.

## Environment Requirements

- **Python**: 3.12+
- **Package Manager**: Recommended to use [uv](https://github.com/astral-sh/uv)
- **Operating System**: Linux Desktop / macOS / Windows

---

## Installation & Setup

### 1. Clone the Repository

First, clone the repository and set up the Python environment.

```bash
# Clone the project repository
git clone https://github.com/TencentCloudADP/youtu-rag.git
cd youtu-rag

# Install dependencies using uv
uv sync

# Activate the virtual environment
source .venv/bin/activate

# Create your environment configuration file from the example
cp .env.example .env
```

---

## Service Configuration

### 2. Object Storage (MinIO) Configuration

MinIO is a high-performance object storage service used to store uploaded document files (still locally managed).

For installation instructions, please refer to the official [MinIO repository](https://github.com/minio/minio). Two installation methods are supported:

- **Install from Source**: Build and install MinIO from source code
- **Build Docker Image**: Deploy MinIO using Docker containers

**Quick Start with Docker**:

```bash
# Pull MinIO Docker image
docker pull minio/minio

# Run MinIO server
docker run -p 9000:9000 -p 9001:9001 \
  --name minio \
  -v ~/minio/data:/data \
  -e "MINIO_ROOT_USER=minioadmin" \
  -e "MINIO_ROOT_PASSWORD=minioadmin" \
  minio/minio server /data --console-address ":9001"
```

After starting, access MinIO console at `http://localhost:9001` with credentials `minioadmin/minioadmin`.

---

### 3. Model Deployment

The following models need to be deployed for Youtu-RAG to function properly:

| Model | HuggingFace | Deployment Method | Required |
|:---|:---|:---|:---:|
| [Youtu-Embedding](https://github.com/TencentCloudADP/youtu-embedding) | [HuggingFace](https://huggingface.co/tencent/Youtu-Embedding) | Local or API | ✅ Required |
| [Youtu-Parsing](https://github.com/TencentCloudADP/youtu-parsing) | [HuggingFace](https://huggingface.co/spaces/tencent/Youtu-Parsing) | Local or API | ⭕ Optional |
| [Youtu-HiChunk](https://github.com/TencentCloudADP/hichunk) | [HuggingFace](https://huggingface.co/spaces/tencent/HiChunk) | Local or API | ⭕ Optional |

**Note**: You can also use other compatible API services for Embedding, OCR, and Chunking instead of deploying these models locally.

---

## Environment Variables Configuration

### 4. Configure Core Environment Variables

Edit the `.env` file and fill in the following core configurations:

#### LLM Configuration (Required)

```bash
# LLM Configuration
UTU_LLM_TYPE=chat.completions
UTU_LLM_MODEL=deepseek-chat
UTU_LLM_BASE_URL=https://api.deepseek.com/v1
UTU_LLM_API_KEY=your_deepseek_api_key  # Replace with your API Key
```

#### Embedding Configuration (Required)

**Option 1: Local Service (Youtu-Embedding-2B)**

```bash
UTU_EMBEDDING_URL=http://localhost:8081
UTU_EMBEDDING_MODEL=youtu-embedding-2B
```

**Option 2: Other API Services**

```bash
UTU_EMBEDDING_URL=https://api.your-embedding-service.com
UTU_EMBEDDING_API_KEY=your_api_key
UTU_EMBEDDING_MODEL=model_name
```

#### MinIO Configuration (Required)

```bash
# MinIO Configuration
ENABLE_MINIO_MONITOR=true
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET=ufile
MINIO_BUCKET_SYS=sysfile
MINIO_SECURE=false
MINIO_LOCAL_TMP_DIR=/tmp/minio
```

#### Optional Services

**Reranker Configuration (Optional, improves retrieval accuracy)**

```bash
UTU_RERANKER_MODEL=jina-reranker-v3
UTU_RERANKER_URL=https://api.jina.ai/v1/rerank
UTU_RERANKER_API_KEY=your_jina_api_key
```

**OCR Configuration (Optional, locally deployable Youtu-Parsing)**

```bash
UTU_OCR_BASE_URL=https://api.ocr.com/ocr
UTU_OCR_MODEL=youtu-ocr
```

**Chunk Configuration (Optional, locally deployable Youtu-HiChunk)**

```bash
UTU_CHUNK_BASE_URL=https://api.hichunk.com/chunk
UTU_CHUNK_MODEL=hichunk
```

**Memory Function (Optional)**

```bash
memoryEnabled=false  # Set to true to enable dual-layer memory mechanism
```

> **Note**: If you don't need OCR and Chunk features, you can disable them by setting `ocr enabled: false` and `chunk enabled: false` in `configs/rag/file_management.yaml`.

For complete environment variable configuration details, please refer to [Environment Variables Documentation](./environment_variables.md).

---

## Start the Service

### 5. Launch Youtu-RAG

```bash
# Method 1: Using startup script (Recommended)
bash start.sh

# Method 2: Directly using uvicorn
uv run uvicorn utu.rag.api.main:app --reload --host 0.0.0.0 --port 8000
```

After successful startup, access the following addresses:

- 📱 **Frontend Interface**: http://localhost:8000
- 📊 **Monitoring Dashboard**: http://localhost:8000/monitor

---

## Using the System

### 6. Getting Started with WebUI

Once the service is running, you can start using Youtu-RAG through the web interface:

#### File Management
1. Click **"File Management"** in the left sidebar
2. Upload files (PDF, Excel, Images, etc.)
3. View file processing status and preview content

#### Knowledge Base Management
1. Click **"Knowledge Base"** in the left sidebar
2. Create a new knowledge base
3. Associate files, databases, or Q&A examples
4. Build vector indices

#### Intelligent Dialogue
1. Click **"Dialogue"** in the left sidebar
2. Select an agent (Chat, KB Search, Text2SQL, etc.)
3. Select a knowledge base if required
4. Enable **"Memory"** switch for better context retention
5. Start chatting!

For detailed usage instructions, please refer to [WebUI Guide](./frontend.md).

---

## Advanced Configuration

### Tracing & Monitoring

Youtu-RAG integrates [Phoenix](https://arize.com/docs/phoenix) for tracing and monitoring agent execution flow.

```bash
# Phoenix/OpenTelemetry Configuration
PHOENIX_ENDPOINT=http://127.0.0.1:6006/v1/traces
PHOENIX_PROJECT_NAME=youtu_rag
PHOENIX_API_KEY=YOUR_PHOENIX_API_KEY  # Required for Phoenix Cloud
```

### Database Configuration

By default, Youtu-RAG uses SQLite for storing system data. You can configure a different database:

```bash
# SQLite (default)
UTU_DB_URL=sqlite:///./rag_data/relational_database/rag_demo.sqlite

# PostgreSQL
UTU_DB_URL=postgresql://user:password@host:port/database

# MySQL
UTU_DB_URL=mysql://user:password@host:port/database
```

### Vector Store Configuration

```bash
# ChromaDB Configuration
ENABLE_VECTOR_MONITOR=true
VECTOR_STORE_PATH=./rag_data/vector_store
```

---

## Troubleshooting

### Common Issues

**Issue 1: MinIO connection failed**
- Ensure MinIO service is running: `docker ps | grep minio`
- Verify MinIO endpoint and credentials in `.env` file

**Issue 2: Embedding service unavailable**
- Check if Youtu-Embedding service is running
- Verify `UTU_EMBEDDING_URL` is correct
- Test the endpoint: `curl http://localhost:8081/health`

**Issue 3: OCR/HiChunk not working**
- Check if these services are properly deployed
- Or disable them in `configs/rag/file_management.yaml`:
  ```yaml
  ocr:
    enabled: false
  chunk:
    enabled: false
  ```

**Issue 4: Port already in use**
- Change the server port in `.env`:
  ```bash
  SERVER_PORT=8001
  ```

---

## Next Steps

- **Explore Agents**: Try different agents for various tasks in the WebUI
- **Upload Documents**: Build your knowledge base with your own documents
- **Enable Memory**: Turn on the memory feature for better conversation experience
- **Check Benchmarks**: Learn about Youtu-RAG's performance in the [README](../README.md#benchmarks)

For more detailed documentation:
- [WebUI Guide](./frontend.md) - Complete guide to using the web interface
- [Environment Variables](./environment_variables.md) - All configuration options
- [README](../README.md) - Project overview and features
