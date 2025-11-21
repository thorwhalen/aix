#!/bin/bash
# Simple test runner script for AIX

set -e

echo "================================"
echo "AIX Test Runner"
echo "================================"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo -e "${RED}pytest not found. Installing...${NC}"
    pip install pytest pytest-cov pytest-mock
fi

# Run unit tests
echo ""
echo -e "${YELLOW}Running unit tests...${NC}"
echo "--------------------------------"
pytest tests/ -v --cov=aix --cov-report=term-missing

# Check exit code
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Unit tests passed${NC}"
else
    echo -e "${RED}✗ Unit tests failed${NC}"
    exit 1
fi

# Run doctests (may fail without API keys, so don't exit on error)
echo ""
echo -e "${YELLOW}Running doctests...${NC}"
echo "--------------------------------"
pytest --doctest-modules aix/ --ignore=aix/gen_ai/ -v || echo -e "${YELLOW}⚠ Some doctests failed (may require API keys)${NC}"

echo ""
echo "================================"
echo -e "${GREEN}Tests completed!${NC}"
echo "================================"
