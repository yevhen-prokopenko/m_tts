#!/usr/bin/env bash
set -e

# Navigate to script directory
cd "$(dirname "$0")"

# Check for virtual environment
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    uv venv --python 3.12 .venv
    uv pip install --python .venv/bin/python -r requirements.txt
fi

# Run Telegram bot
.venv/bin/python bot.py
