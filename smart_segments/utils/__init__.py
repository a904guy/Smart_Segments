"""Utility functions for Smart Segments plugin"""

from .environment import EnvironmentUtils, DependencyChecker
from .platform_utils import (
    PlatformType,
    CrossPlatformPathHandler,
    PythonDetector,
    PortableFileOperations,
    path_handler,
    python_detector,
    file_ops
)
from .system_utils import (
    SystemResourceManager,
    PlatformSpecificOperations,
    resource_manager,
    platform_ops
)
from .krita_bridge import KritaImageBridge

__all__ = [
    'EnvironmentUtils',
    'DependencyChecker',
    'PlatformType',
    'CrossPlatformPathHandler',
    'PythonDetector',
    'PortableFileOperations',
    'SystemResourceManager',
    'PlatformSpecificOperations',
    'KritaImageBridge',
    'path_handler',
    'python_detector',
    'file_ops',
    'resource_manager',
    'platform_ops'
]
