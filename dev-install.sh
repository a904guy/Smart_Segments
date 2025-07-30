#!/bin/bash
# Development installation script for Smart Segments Krita Plugin
# This script creates symbolic links for active development with APT Krita

set -e

echo "🚀 Smart Segments Development Installation"
echo "=========================================="

# Check if we're in the right directory
if [ ! -f "smart_segments.desktop" ] || [ ! -d "smart_segments" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    exit 1
fi

# Check if Krita is installed
if ! command -v krita > /dev/null 2>&1; then
    echo "❌ Error: Krita not found. Installing Krita with Python support..."
    echo "Running: sudo apt update && sudo apt install krita python3-pyqt5"
    
    if command -v sudo > /dev/null 2>&1; then
        sudo apt update && sudo apt install -y krita python3-pyqt5
    else
        echo "Please install Krita manually: apt update && apt install krita python3-pyqt5"
        exit 1
    fi
fi

# Verify we have the APT version (not snap)
KRITA_PATH=$(which krita)
if [[ "$KRITA_PATH" == *"/snap/"* ]]; then
    echo "⚠️  Warning: Found snap version of Krita at $KRITA_PATH"
    echo "The snap version doesn't support Python plugins. Please install the APT version:"
    echo "   sudo snap remove krita"
    echo "   sudo apt install krita python3-pyqt5"
    exit 1
fi

echo "✅ Found APT Krita at: $KRITA_PATH"

# Install plugin directly
echo "📦 Installing plugin with symbolic links..."

# Determine Krita plugin directory
KRITA_DIR=""
if [ -d "$HOME/AppData/Roaming/krita/pykrita" ]; then
    # Windows path (if running in WSL)
    KRITA_DIR="$HOME/AppData/Roaming/krita/pykrita"
elif [ -d "$HOME/.local/share/krita/pykrita" ]; then
    # Linux path
    KRITA_DIR="$HOME/.local/share/krita/pykrita"
else
    # Create the directory
    mkdir -p "$HOME/.local/share/krita/pykrita"
    KRITA_DIR="$HOME/.local/share/krita/pykrita"
fi

# Create symbolic links
ln -sf "$(pwd)/smart_segments" "$KRITA_DIR/smart_segments"
ln -sf "$(pwd)/smart_segments.desktop" "$KRITA_DIR/smart_segments.desktop"

echo "✅ Plugin symlinked to $KRITA_DIR for development"

echo ""
echo "🎨 Krita Plugin Installation Complete!"
echo ""
echo "📋 Next steps:"
echo "1. Launch Krita: krita"
echo "2. Go to Settings → Configure Krita → Python Plugin Manager"
echo "3. Enable 'Smart Segments' plugin"
echo "4. Restart Krita"
echo "5. Find the plugin in Tools → Smart Segments"
echo ""
echo "🔧 Development benefits:"
echo "• Any changes to the code will be immediately available in Krita"
echo "• No need to reinstall after code changes"
echo "• Just restart Krita to pick up changes"
echo "• Full Python debugging support"
echo ""
echo "📁 Plugin location: $KRITA_DIR/smart_segments"
echo "📝 Krita logs: ~/.local/share/krita/krita.log"
echo "🗑️  To uninstall: rm -f $KRITA_DIR/smart_segments $KRITA_DIR/smart_segments.desktop"
