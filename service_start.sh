#!/usr/bin/env bash
# Start / Enable background launchd service
PLIST="$HOME/Library/LaunchAgents/com.mtts.bot.plist"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ ! -f "$PLIST" ]; then
    echo "❌ Service file not found: $PLIST"
    exit 1
fi

launchctl unload "$PLIST" 2>/dev/null || true
launchctl load "$PLIST"
echo "✅ Telegram bot background service started!"
echo "Logs: tail -f $SCRIPT_DIR/bot.log"
