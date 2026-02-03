# Environment Variables

This document outlines the configuration of key environment variables in Youtu-RAG.

## Overview

Before running Youtu-RAG, you need to configure the necessary environment variables. A recommended practice is to start by copying the `.env.example` file from the project root:

```sh
cp .env.example .env
```

Then, edit the `.env` file to fill in the required API Keys and other configurations.

## Web UI

The Youtu-RAG web interface can be configured with the following settings:

```sh
# Server host (0.0.0.0 to listen on all interfaces)
SERVER_HOST=0.0.0.0
# Server port (frontend and backend use the same port)
SERVER_PORT=8000
```

## LLM API Keys

The core capabilities of Youtu-RAG rely on Large Language Models (LLMs). You need to configure the appropriate models for different functional modules. This is the primary text generation LLM that the Agent relies on for its operations.

```sh
# LLM type (e.g., chat.completions, responses)
UTU_LLM_TYPE=chat.completions
# LLM model name (e.g., deepseek-chat, gpt-4-turbo)
UTU_LLM_MODEL=deepseek-chat
# LLM provider's API base URL (e.g., https://api.deepseek.com/v1)
UTU_LLM_BASE_URL=https://api.deepseek.com/v1
# LLM provider's API Key
UTU_LLM_API_KEY=YOUR_API_KEY
```

## Tools

Some Toolkits require their own API Keys or specific configurations.

### Search Toolkit (`SearchToolkit`)

The search toolkit integrates the following two services by default:

1. **Web Search**: Uses the efficient Google Search API provided by [Serper](https://serper.dev/). You will need to register and obtain an API Key.
2. **Web Content Extraction**: Uses the [Jina AI Reader](https://jina.ai/reader/) to convert web page content into an LLM-friendly Markdown format. This also requires registration and an API Key.

Configure them in your `.env` file as follows:

```sh
# Get from https://serper.dev/
SERPER_API_KEY=YOUR_SERPER_API_KEY
# Get from https://jina.ai/reader/
JINA_API_KEY=YOUR_JINA_API_KEY
```

### OCR Toolkit (`Youtu-Parsing`)

Youtu-RAG integrates an OCR service for extracting text from images and documents:

```sh
# OCR service base URL
UTU_OCR_BASE_URL=YOUR_OCR_SERVICE_URL
# OCR model name
UTU_OCR_MODEL=youtu-ocr
```

### Embedding Toolkit (`Youtu-Embedding`)

Youtu-RAG supports two embedding service options:

**Option 1: Local Youtu-Embedding Service (2048 dimensions)**

```sh
# Youtu-Embedding-2B Local Service URL
UTU_EMBEDDING_URL=YOUR_EMBEDDING_SERVICE_URL
# Model name
UTU_EMBEDDING_MODEL=youtu-embedding-2B
```

**Option 2: Other Embedding Service**

```sh
# API URL (OpenAI API Standard)
UTU_EMBEDDING_URL=YOUR_API_URL
# API Key
UTU_EMBEDDING_API_KEY=YOUR_API_KEY
# Model name
UTU_EMBEDDING_MODEL=MODEL_NAME
```


### Reranker Toolkit (`Jina-Reranker`)

The reranker service is used to improve retrieval accuracy by re-ranking search results:

```sh
# Reranker model name
UTU_RERANKER_MODEL=jina-reranker-v3
# Reranker API URL
UTU_RERANKER_URL=https://api.jina.ai/v1/rerank
# Reranker API Key (get from https://jina.ai/reranker)
UTU_RERANKER_API_KEY=YOUR_JINA_RERANKER_API_KEY
```

## Database

### Vector Database

Youtu-RAG uses ChromaDB as the vector database for storing document embeddings:

```sh
# Enable vector store monitoring
ENABLE_VECTOR_MONITOR=true
# ChromaDB vector store path
VECTOR_STORE_PATH=./rag_data/vector_store
```

### Relational Database

A relational database is used for system administration, storing tracing data, and evaluation results:

```sh
# Database URL (SQLite example)
# For local development, the SQLite database will be created in the specified directory
UTU_DB_URL=sqlite:///./rag_data/relational_database/rag_demo.sqlite
```

You can also use other database systems by modifying the connection URL accordingly (e.g., PostgreSQL, MySQL).

### MinIO

MinIO is used as object storage for managing RAG-related files:

```sh
# Enable MinIO object storage monitoring
ENABLE_MINIO_MONITOR=true
# MinIO endpoint
MINIO_ENDPOINT=localhost:9000
# MinIO access credentials
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
# MinIO bucket name for RAG user file storage
MINIO_BUCKET=ufile
# MinIO bucket name for RAG system file storage
MINIO_BUCKET_SYS=sysfile
# Use secure connection (true for https, false for http)
MINIO_SECURE=false
# Local temporary directory for MinIO operations
MINIO_LOCAL_TMP_DIR=/tmp/minio
```

## Tracing & Monitoring

The framework integrates [OpenTelemetry](https://opentelemetry.io/) and [Phoenix](https://arize.com/docs/phoenix) for tracing and monitoring the Agent's execution flow.

```sh
# Phoenix/OpenTelemetry trace receiver endpoint
PHOENIX_ENDPOINT=http://127.0.0.1:6006/v1/traces
# Project name displayed in the Phoenix UI
PHOENIX_PROJECT_NAME=youtu_rag
# Phoenix API Key (required for Phoenix Cloud service)
# Get from https://app.phoenix.arize.com/s/_space_name_/settings/general
PHOENIX_API_KEY=YOUR_PHOENIX_API_KEY
```

If you're using Phoenix locally, only the `PHOENIX_ENDPOINT` is required. For Phoenix Cloud service, you'll also need to configure the `PHOENIX_API_KEY`.


## Memory

Youtu-RAG supports a memory feature for maintaining context across conversations:

```sh
# Enable/disable memory feature (true/false)
memoryEnabled=false
```

Set to `true` to enable the memory feature, which allows the system to remember previous interactions.

## Logging

Configure the logging level for the application:

```sh
# Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
UTU_LOG_LEVEL=INFO
```

Available log levels:
- `DEBUG`: Detailed information for diagnosing problems
- `INFO`: General informational messages (recommended for production)
- `WARNING`: Warning messages for potentially harmful situations
- `ERROR`: Error messages for serious problems
- `CRITICAL`: Critical messages for very serious errors