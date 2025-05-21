#!/bin/bash
# Script to stop the locally running FlashDNA API

# Check if server PID file exists
if [ -f tmp/server.pid ]; then
    PID=$(cat tmp/server.pid)
    echo "Stopping FlashDNA API (PID: $PID)..."
    kill -9 $PID 2>/dev/null || echo "Process already stopped"
    rm tmp/server.pid
    echo "Server stopped successfully"
else
    echo "No server PID file found"
    
    # Try to find and kill any uvicorn processes
    PIDS=$(pgrep -f "uvicorn flashcamp.backend.main:app")
    if [ -n "$PIDS" ]; then
        echo "Found uvicorn processes: $PIDS"
        echo "Stopping them..."
        kill -9 $PIDS
        echo "Processes stopped"
    else
        echo "No running uvicorn processes found"
    fi
fi

# Clean up compiled Python files
find flashcamp -name '*.pyc' -delete
echo "Cleaned up .pyc files" 