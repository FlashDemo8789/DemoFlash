#!/bin/bash
# Script to stop and clean up FlashDNA Docker services

set -e  # Exit on error

echo "=== Stopping FlashDNA services ==="
docker-compose down

echo "=== Checking for orphaned containers ==="
orphaned=$(docker ps -a --filter "name=flashcamp" --format "{{.ID}}")
if [ -n "$orphaned" ]; then
    echo "Found orphaned containers, removing them..."
    docker rm -f $orphaned
else
    echo "No orphaned containers found."
fi

echo "=== Cleanup complete! ==="
echo "To restart services, run: ./docker-build.sh" 