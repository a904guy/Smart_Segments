"""
Test Suite for Smart Segments Krita Plugin

This package contains unit, integration, and performance tests for the Smart Segments plugin.
"""

import sys
import os
from pathlib import Path

# Add the plugin directory to the Python path for testing
plugin_root = Path(__file__).parent.parent / "pykrita" / "smart_segments"
sys.path.insert(0, str(plugin_root))

# Test configuration
TEST_CONFIG = {
    'verbose': True,
    'capture_logs': True,
    'temp_dir': Path(__file__).parent / "temp",
    'test_data_dir': Path(__file__).parent / "test_data",
    'mock_krita': True  # Use mocked Krita API for testing
}

# Ensure test directories exist
TEST_CONFIG['temp_dir'].mkdir(exist_ok=True)
TEST_CONFIG['test_data_dir'].mkdir(exist_ok=True)
