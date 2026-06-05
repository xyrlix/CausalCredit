#!/usr/bin/env bash
set -euo pipefail

echo "Starting CausalCredit Streamlit Demo..."
streamlit run src/frontend/app.py --server.port 8501
