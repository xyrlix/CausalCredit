#!/usr/bin/env bash
set -euo pipefail

echo "=== CausalCredit Environment Setup ==="

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | grep -oP '\d+\.\d+')
echo "Detected Python: $PYTHON_VERSION"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate 2>/dev/null || source venv/Scripts/activate 2>/dev/null

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -e ".[dev]"

# Create data directories
mkdir -p data/home-credit-default-risk
mkdir -p data/lending-club
mkdir -p data/processed
mkdir -p models
mkdir -p logs

echo ""
echo "=== Setup complete! ==="
echo "Run 'make run-api' to start the API server"
echo "Run 'make run-demo' to start the Streamlit demo"
echo "Run 'make test' to run tests"
