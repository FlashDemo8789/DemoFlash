#!/usr/bin/env bash
set -euo pipefail

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting FLASH application...${NC}"

# Set the model path
export FLASHDNA_MODEL=${FLASHDNA_MODEL:-$(pwd)/models/success_xgb.joblib}

echo -e "${YELLOW}Environment variables:${NC}"
echo "FLASHDNA_MODEL: $FLASHDNA_MODEL"

# Start the backend service
echo -e "${GREEN}Starting backend service...${NC}"
exec uvicorn flashcamp.backend.app:app --host 0.0.0.0 --port "${PORT:-8000}" "$@" 