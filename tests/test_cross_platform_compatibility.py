#!/usr/bin/env python3
"""
Test script for Cross-Platform Compatibility Layer

This script tests all the cross-platform utilities to ensure they work
correctly across Windows, Linux, and macOS.
"""

import sys
import tempfile
from pathlib import Path

# Add the pykrita module to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "pykrita" / "smart_segments"))

# Import our cross-platform utilities directly from modules
try:
    from smart_segments.utils.platform_utils import (
        path_handler, python_detector, file_ops,
        PlatformType, CrossPlatformPathHandler, PythonDetector, PortableFileOperations
    )
    from smart_segments.utils.system_utils import (
        resource_manager, platform_ops,
        SystemResourceManager, PlatformSpecificOperations
    )
except ImportError as e:
    print(f"Import error: {e}")
    print("Trying direct module imports...")
    
    # Fallback: import modules directly
    import smart_segments.utils.platform_utils as platform_utils
    import smart_segments.utils.system_utils as system_utils
    
    path_handler = platform_utils.path_handler
    python_detector = platform_utils.python_detector
    file_ops = platform_utils.file_ops
    resource_manager = system_utils.resource_manager
    platform_ops = system_utils.platform_ops
    
    PlatformType = platform_utils.PlatformType
    CrossPlatformPathHandler = platform_utils.CrossPlatformPathHandler
    PythonDetector = platform_utils.PythonDetector
    PortableFileOperations = platform_utils.PortableFileOperations
    SystemResourceManager = system_utils.SystemResourceManager
    PlatformSpecificOperations = system_utils.PlatformSpecificOperations


def test_platform_detection():
    """Test platform detection functionality"""
    print("=== Testing Platform Detection ===")
    
    print(f"Detected platform: {path_handler.platform}")
    print(f"Platform type: {path_handler.platform.value}")
    print(f"Path separator: '{path_handler.separator}'")
    print(f"Line ending: {repr(path_handler.line_ending)}")
    print(f"Case sensitive filesystem: {path_handler.is_case_sensitive_filesystem()}")
    print(f"Executable extension: '{path_handler.get_executable_extension()}'")
    print()


def test_path_operations():
    """Test cross-platform path operations"""
    print("=== Testing Path Operations ===")
    
    # Test path normalization
    test_paths = [
        "test/path/file.txt",
        "/absolute/path/file.txt",
        "..\\windows\\style\\path.exe",
        "~/user/home/file.py"
    ]
    
    for test_path in test_paths:
        normalized = path_handler.normalize_path(test_path)
        print(f"'{test_path}' -> '{normalized}'")
    
    # Test path joining
    joined = path_handler.join_paths("base", "sub", "file.txt")
    print(f"Joined path: {joined}")
    
    # Test directory methods
    print(f"Home directory: {path_handler.get_home_directory()}")
    print(f"Temp directory: {path_handler.get_temp_directory()}")
    print(f"App data directory: {path_handler.get_app_data_directory()}")
    print(f"Cache directory: {path_handler.get_cache_directory()}")
    print()


def test_python_detection():
    """Test Python detection functionality"""
    print("=== Testing Python Detection ===")
    
    pythons = python_detector.find_python_executables()
    print(f"Found {len(pythons)} Python installations:")
    
    for python in pythons[:3]:  # Show first 3
        print(f"  - {python['executable']}")
        print(f"    Version: {python['version']}")
        print(f"    Architecture: {python['architecture']}")
        print(f"    Is current: {python['is_current']}")
        print(f"    Virtual env: {python['is_virtual_env']}")
        if python['virtual_env_path']:
            print(f"    Venv path: {python['virtual_env_path']}")
        print()
    
    # Test recommended Python
    recommended = python_detector.get_recommended_python()
    if recommended:
        print(f"Recommended Python: {recommended['executable']} (v{recommended['version']})")
    else:
        print("No compatible Python found")
    print()


def test_file_operations():
    """Test portable file operations"""
    print("=== Testing File Operations ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Test text file operations
        test_file = temp_path / "test.txt"
        test_content = "Hello, cross-platform world!\nLine 2\nLine 3"
        
        success = file_ops.write_text_file(test_file, test_content)
        print(f"Write text file: {'✓' if success else '✗'}")
        
        read_content = file_ops.read_text_file(test_file)
        print(f"Read text file: {'✓' if read_content == test_content else '✗'}")
        
        # Test directory operations
        test_dir = temp_path / "test_directory"
        success = file_ops.create_directory(test_dir)
        print(f"Create directory: {'✓' if success else '✗'}")
        
        # Test file copying
        copy_file = temp_path / "copy.txt"
        success = file_ops.copy_file(test_file, copy_file)
        print(f"Copy file: {'✓' if success else '✗'}")
        
        # Test file permissions (Unix only)
        if path_handler.platform != PlatformType.WINDOWS:
            permissions = file_ops.get_file_permissions(test_file)
            print(f"File permissions: {oct(permissions) if permissions else 'N/A'}")
            
            is_executable = file_ops.is_executable(test_file)
            print(f"Is executable: {is_executable}")
        
        # Test file deletion
        success = file_ops.delete_file(copy_file)
        print(f"Delete file: {'✓' if success else '✗'}")
        
        success = file_ops.delete_directory(test_dir)
        print(f"Delete directory: {'✓' if success else '✗'}")
    
    print()


def test_system_resources():
    """Test system resource monitoring"""
    print("=== Testing System Resources ===")
    
    # Get system info
    sys_info = resource_manager.get_system_info()
    print(f"OS: {sys_info.os_name} {sys_info.os_version}")
    print(f"Architecture: {sys_info.architecture}")
    print(f"CPU count: {sys_info.cpu_count}")
    print(f"Memory total: {sys_info.memory_total} MB")
    print(f"Memory available: {sys_info.memory_available} MB")
    print(f"Disk space free: {sys_info.disk_space_free} MB")
    print(f"Python: {sys_info.python_version} ({sys_info.python_executable})")
    print()
    
    # Get resource usage
    memory_usage = resource_manager.get_memory_usage()
    print(f"Memory usage: {memory_usage['percent']:.1f}%")
    
    cpu_usage = resource_manager.get_cpu_usage(interval=0.1)
    print(f"CPU usage: {cpu_usage['overall']:.1f}%")
    print(f"CPU cores: {cpu_usage['count']}")
    
    disk_usage = resource_manager.get_disk_usage()
    print(f"Disk usage: {disk_usage['percent']:.1f}%")
    print()


def test_platform_operations():
    """Test platform-specific operations"""
    print("=== Testing Platform Operations ===")
    
    # Test environment variables
    env_vars = platform_ops.get_environment_variables()
    print(f"Environment variables count: {len(env_vars)}")
    
    # Test PATH manipulation
    system_path = platform_ops.get_system_path()
    print(f"PATH entries: {len(system_path)}")
    print(f"First PATH entry: {system_path[0] if system_path else 'None'}")
    
    # Test setting environment variable (non-persistent)
    test_var_name = "SMART_SEGMENTS_TEST"
    success = platform_ops.set_environment_variable(test_var_name, "test_value")
    print(f"Set env variable: {'✓' if success else '✗'}")
    
    # Test getting default browser
    browser = platform_ops.get_default_browser()
    print(f"Default browser: {browser or 'Unknown'}")
    
    # Test file associations
    py_associations = platform_ops.get_file_associations('.py')
    print(f"Python file associations: {len(py_associations)}")
    if py_associations:
        print(f"  First: {py_associations[0]}")
    
    # Test installed applications (limited output)
    applications = platform_ops.get_installed_applications()
    print(f"Installed applications found: {len(applications)}")
    if applications:
        print(f"  Example: {applications[0]['name']}")
    
    print()


def test_process_monitoring():
    """Test process monitoring functionality"""
    print("=== Testing Process Monitoring ===")
    
    # Test monitoring current process
    current_pid = __import__('os').getpid()
    process_info = resource_manager.monitor_process(current_pid)
    
    if process_info:
        print(f"Current process (PID {current_pid}):")
        print(f"  Name: {process_info.name}")
        print(f"  Status: {process_info.status}")
        print(f"  Memory: {process_info.memory_mb:.1f} MB")
        print(f"  CPU: {process_info.cpu_percent:.1f}%")
    else:
        print(f"Failed to get info for current process (PID {current_pid})")
    
    # Test finding processes by name
    python_processes = resource_manager.find_processes_by_name("python")
    print(f"Python processes found: {len(python_processes)}")
    
    print()


def main():
    """Run all tests"""
    print("Cross-Platform Compatibility Layer Test")
    print("=" * 50)
    print()
    
    try:
        test_platform_detection()
        test_path_operations()
        test_python_detection()
        test_file_operations()
        test_system_resources()
        test_platform_operations()
        test_process_monitoring()
        
        print("✓ All tests completed successfully!")
        
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
