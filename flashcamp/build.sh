#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting optimized Docker build for FLASH project...${NC}"
echo -e "${YELLOW}Note: This build is optimized for Intel MacBook Pro${NC}"

# Make sure Docker BuildKit is enabled
export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1

# Set platform explicitly
export DOCKER_DEFAULT_PLATFORM=linux/amd64

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
  echo -e "${YELLOW}Docker is not running. Starting Docker...${NC}"
  open -a Docker
  
  # Wait for Docker to start
  echo "Waiting for Docker to start..."
  until docker info > /dev/null 2>&1; do
    sleep 1
  done
fi

# Run database migrations
echo -e "${GREEN}Setting up database...${NC}"
if ! command -v python -m alembic > /dev/null 2>&1; then
  echo -e "${YELLOW}Installing alembic...${NC}"
  pip install alembic
fi

# Run migrations
echo -e "${GREEN}Running database migrations...${NC}"
export PYTHONPATH=.
python -m alembic upgrade head || {
  echo -e "${RED}Error running migrations. Please check your database configuration.${NC}"
  exit 1
}

# Build images with cache optimization
echo -e "${GREEN}Building Docker images...${NC}"
docker-compose build --parallel --build-arg BUILDKIT_INLINE_CACHE=1

echo -e "${GREEN}Build completed successfully!${NC}"
echo -e "${YELLOW}To start the services, run: docker-compose up${NC}"

# Optional: start services immediately
read -p "Start services now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
  echo -e "${GREEN}Starting services...${NC}"
  docker-compose up
fi 