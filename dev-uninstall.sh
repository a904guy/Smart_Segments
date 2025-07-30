#!/bin/bash
# Development uninstall script for Smart Segments Krita Plugin

set -e

echo "🗑️  Smart Segments Plugin Uninstall"
echo "==================================="

# Find and remove plugin files
REMOVED=false

# Check Windows path (if running in WSL)
if [ -L "$HOME/AppData/Roaming/krita/pykrita/smart_segments" ]; then
    rm -f "$HOME/AppData/Roaming/krita/pykrita/smart_segments"
    rm -f "$HOME/AppData/Roaming/krita/pykrita/smart_segments.desktop"
    echo "✅ Plugin removed from $HOME/AppData/Roaming/krita/pykrita/"
    REMOVED=true
fi

# Check Linux path
if [ -L "$HOME/.local/share/krita/pykrita/smart_segments" ]; then
    rm -f "$HOME/.local/share/krita/pykrita/smart_segments"
    rm -f "$HOME/.local/share/krita/pykrita/smart_segments.desktop"
    echo "✅ Plugin removed from $HOME/.local/share/krita/pykrita/"
    REMOVED=true
fi

if [ "$REMOVED" = false ]; then
    echo "ℹ️  Plugin not found in any of the standard Krita plugin directories."
    echo "No action needed."
else
    echo ""
    echo "📋 Complete removal:"
    echo "1. Restart Krita if it's running"
    echo "2. The plugin will no longer appear in the Python Plugin Manager"
fi
