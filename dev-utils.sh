#!/bin/bash
# Development utilities for Smart Segments Krita Plugin

set -e

show_help() {
    echo "Smart Segments Development Utilities"
    echo ""
    echo "Usage: ./dev-utils.sh <command>"
    echo ""
    echo "Commands:"
    echo "  install      Install plugin to Krita (same as ./dev-install.sh)"
    echo "  uninstall    Remove plugin from Krita (same as ./dev-uninstall.sh)"
    echo "  bundle       Create distribution bundle"
    echo "  test         Run all tests"
    echo "  clean        Clean build artifacts and cache files"
    echo "  version      Show plugin version"
    echo "  help         Show this help message"
}

clean_artifacts() {
    echo "🧹 Cleaning build artifacts and cache files..."
    
    # Remove Python cache files
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true
    find . -type f -name "*.pyo" -delete 2>/dev/null || true
    find . -type f -name "*~" -delete 2>/dev/null || true
    
    # Remove build directories
    rm -rf build/ dist/ *.egg-info/ 2>/dev/null || true
    rm -rf temp_bundle/ temp_package/ 2>/dev/null || true
    
    # Remove coverage files
    rm -f .coverage 2>/dev/null || true
    rm -rf htmlcov/ 2>/dev/null || true
    
    echo "✅ Clean complete"
}

case "${1:-help}" in
    "install")
        ./dev-install.sh
        ;;
    "uninstall")
        ./dev-uninstall.sh
        ;;
    "bundle")
        echo "📦 Creating distribution bundle..."
        python3 create_plugin_package.py
        ;;
    "test")
        echo "🧪 Running tests..."
        cd tests && python3 run_tests.py
        ;;
    "clean")
        clean_artifacts
        ;;
    "version")
        python3 -c "
import sys
sys.path.insert(0, '.')
try:
    from smart_segments.version import __version__
    print(f'Smart Segments v{__version__}')
except ImportError:
    # Fallback if imports fail outside Krita
    import re
    with open('smart_segments/version.py') as f:
        version_content = f.read()
        match = re.search(r'__version__\s*=\s*[\"\'](.*?)[\"\']', version_content)
        if match:
            print(f'Smart Segments v{match.group(1)}')
        else:
            print('Smart Segments (version unknown)')
"
        ;;
    "help"|*)
        show_help
        ;;
esac
