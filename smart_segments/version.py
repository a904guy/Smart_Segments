"""
Version information for Smart Segments Krita Plugin
"""

__version__ = "1.0.2"
__version_info__ = (1, 0, 2)

# Build information
__build__ = "dev"
__build_date__ = "2024-01-01"

# Compatibility information
MIN_KRITA_VERSION = "5.0.0"
MIN_PYTHON_VERSION = "3.10"

# Plugin metadata
PLUGIN_NAME = "Smart Segments"
PLUGIN_ID = "smart_segments"
PLUGIN_DESCRIPTION = "AI-powered intelligent segmentation tool for Krita"
PLUGIN_AUTHOR = "Hawkins.Tech Inc"
PLUGIN_LICENSE = "MIT"

def get_version_string():
    """Get formatted version string"""
    if __build__ == "dev":
        return f"{__version__}-dev"
    return __version__

def get_full_version_info():
    """Get complete version information"""
    return {
        "version": __version__,
        "version_info": __version_info__,
        "build": __build__,
        "build_date": __build_date__,
        "plugin_name": PLUGIN_NAME,
        "plugin_id": PLUGIN_ID,
        "min_krita_version": MIN_KRITA_VERSION,
        "min_python_version": MIN_PYTHON_VERSION,
    }
