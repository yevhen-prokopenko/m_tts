#!/usr/bin/env bash
# Stop / Disable background launchd service
PLIST="$HOME/Library/LaunchAgents/com.mtts.bot.plist"

launchctl unload "$PLIST" 2>/dev/null || true
echo "⏹ Telegram bot background service stopped."
