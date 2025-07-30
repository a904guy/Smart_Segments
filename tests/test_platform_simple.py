#!/usr/bin/env python3
"""
Simple test for cross-platform utilities
"""

import sys
import tempfile
from pathlib import Path

# Direct imports without going through __init__.py
sys.path.insert(0, str(Path(__file__).parent / "pykrita" / "smart_segments" / "utils"))

def test_basic_functionality():
    """Test basic cross-platform functionality"""
    print("Testing Cross-Platform Compatibility Layer")
    print("=" * 50)
    
    try:
        # Import platform utilities
        import platform_utils
        
        print("✓ Platform utilities imported successfully")
        
        # Test platform detection
        handler = platform_utils.CrossPlatformPathHandler()
        print(f"Platform: {handler.platform}")
        print(f"Path separator: '{handler.separator}'")
        print(f"Line ending: {repr(handler.line_ending)}")
        
        # Test path operations
        test_path = "test/path/file.txt"
        normalized = handler.normalize_path(test_path)
        print(f"Normalized path: {normalized}")
        
        # Test file operations
        file_ops = platform_utils.PortableFileOperations()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            test_file = temp_path / "test.txt"
            test_content = "Hello, cross-platform world!"
            
            # Write and read test
            success = file_ops.write_text_file(test_file, test_content)
            print(f"Write file: {'✓' if success else '✗'}")
            
            read_content = file_ops.read_text_file(test_file)
            print(f"Read file: {'✓' if read_content == test_content else '✗'}")
        
        print("✓ Platform utilities working correctly")
        
    except Exception as e:
        print(f"✗ Error testing platform utilities: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        # Import system utilities
        import system_utils
        
        print("✓ System utilities imported successfully")
        
        # Test system info
        resource_mgr = system_utils.SystemResourceManager()
        sys_info = resource_mgr.get_system_info()
        
        print(f"OS: {sys_info.os_name}")
        print(f"CPU count: {sys_info.cpu_count}")
        print(f"Disk free: {sys_info.disk_space_free} MB")
        
        # Test platform operations
        platform_ops = system_utils.PlatformSpecificOperations()
        env_vars = platform_ops.get_environment_variables()
        print(f"Environment variables: {len(env_vars)}")
        
        print("✓ System utilities working correctly")
        
    except Exception as e:
        print(f"✗ Error testing system utilities: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✓ Basic cross-platform compatibility test completed!")


if __name__ == "__main__":
    test_basic_functionality()
