#!/usr/bin/env bash
# Check service status and view recent logs
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=============================================="
echo "📊 TELEGRAM BOT SERVICE STATUS"
echo "=============================================="

PID=$(pgrep -f "m_tts.*bot.py" || true)
if [ -n "$PID" ]; then
    echo "🟢 Bot is active (PID: $PID)"
else
    echo "🔴 Bot is not running."
fi

echo ""
echo "Recent 15 log lines (bot.log):"
echo "----------------------------------------------"
if [ -f "$SCRIPT_DIR/bot.log" ]; then
    tail -n 15 "$SCRIPT_DIR/bot.log"
else
    echo "Log file not created yet."
fi
echo "----------------------------------------------"
