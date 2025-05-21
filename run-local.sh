#!/bin/bash
# Script to run the FlashDNA application locally

set -e  # Exit on error

# Create log directory if it doesn't exist
mkdir -p tmp

# Kill any existing uvicorn processes
pkill -f uvicorn 2>/dev/null || echo "No uvicorn processes found"

# Sleep to ensure processes are terminated
sleep 2

# Clean up compiled Python files
find flashcamp -name '*.pyc' -delete

# Find an available port starting from 8000
PORT=8000
while lsof -i :$PORT >/dev/null 2>&1; do
  echo "Port $PORT is in use, trying next port..."
  PORT=$((PORT+1))
done

echo "=== Starting FlashDNA API on port $PORT ==="
echo "Logs will be saved to tmp/server.log"

# Start the server
nohup uvicorn flashcamp.backend.main:app --reload --host 0.0.0.0 --port $PORT > tmp/server.log 2>&1 & 
echo $! > tmp/server.pid

# Wait a moment for the server to start
sleep 3

# Check if server started successfully
if curl -s http://localhost:$PORT/ > /dev/null; then
    echo "✅ FlashDNA API is running!"
    echo "API is available at: http://localhost:$PORT"
    echo "API documentation is available at: http://localhost:$PORT/api/docs"
    echo "Server PID: $(cat tmp/server.pid)"
else
    echo "❌ API service didn't start properly. Check logs at tmp/server.log"
    tail -n 20 tmp/server.log
fi

echo ""
echo "=== Commands you may need: ==="
echo "- View logs: tail -f tmp/server.log"
echo "- Stop server: kill -9 \$(cat tmp/server.pid)" 