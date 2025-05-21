#!/bin/bash
# Script to build and run the FlashDNA Docker container

set -e  # Exit on error

echo "=== Building FlashDNA Docker image ==="
docker-compose build

echo "=== Starting FlashDNA services ==="
docker-compose up -d

echo "=== Waiting for services to start ==="
sleep 5

echo "=== Checking service health ==="
if curl -s http://localhost:8000/ > /dev/null; then
    echo "✅ FlashDNA API is running!"
    echo "API is available at: http://localhost:8000"
    echo "API documentation is available at: http://localhost:8000/api/docs"
else
    echo "❌ API service didn't start properly. Check logs with: docker-compose logs api"
fi

echo ""
echo "=== Commands you may need: ==="
echo "- View logs: docker-compose logs -f api"
echo "- Stop services: docker-compose down"
echo "- Rebuild and restart: ./docker-build.sh" 