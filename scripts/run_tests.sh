#!/usr/bin/env bash
set -euo pipefail

echo "Running CausalCredit test suite..."
pytest tests/ -v --cov=src --cov-report=term-missing
