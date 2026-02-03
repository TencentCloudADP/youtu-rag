#!/bin/bash

# Youtu-RAG unified startup script
# Provides frontend and backend services

if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

HOST=${SERVER_HOST:-0.0.0.0}
PORT=${SERVER_PORT:-8000}

echo "========================================"
echo "  Starting Youtu-RAG..."
echo "========================================"
echo ""
echo "📦 Loading environment variables..."
echo "   Host: $HOST"
echo "   Port: $PORT"
echo ""
echo "🚀 Starting services..."
echo ""
echo "========================================"
echo "  Access URLs"
echo "========================================"
echo "📱 Frontend: http://localhost:$PORT"
echo "📊 Monitor: http://localhost:$PORT/monitor"
echo "========================================"
echo ""
echo "Tip: Press Ctrl+C to stop the service"
echo ""

uv run uvicorn utu.rag.api.main:app --reload --host $HOST --port $PORT

