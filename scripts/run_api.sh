#!/usr/bin/env bash
set -euo pipefail

echo "Starting CausalCredit API server..."
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
