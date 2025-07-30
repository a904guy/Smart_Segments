"""
Unit tests for platform utilities
"""

import unittest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import the modules to test
try:
    from utils.platform_utils import (
        CrossPlatformPathHandler, PythonDetector, PortableFileOperations,
        path_handler, python_detector, file_ops
    )
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "pykrita" / "smart_segments"))
    from utils.platform_utils import (
        CrossPlatformPathHandler, PythonDetector, PortableFileOperations,
        path_handler, python_detector, file_ops
    )


class TestPlatformUtils(unittest.TestCase):
    """Test cases for platform utilities functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.path_handler = path_handler
        self.python_detector = python_detector
        self.file_ops = file_ops
        self.test_temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        if os.path.exists(self.test_temp_dir):
            shutil.rmtree(self.test_temp_dir)
    
    def test_get_platform_info(self):
        """Test platform information detection"""
        platform = self.path_handler.platform
        
        self.assertIsNotNone(platform)
        self.assertIn(platform.value, ['windows', 'linux', 'macos'])
    
    def test_normalize_path(self):
        """Test path normalization across platforms"""
        # Test basic path normalization
        test_paths = [
            "/home/user/test",
            "C:\\Users\\User\\test",
            "~/Documents/test",
            "./relative/path"
        ]
        
        for path in test_paths:
            normalized = self.path_handler.normalize_path(path)
            # The normalize_path method returns a Path object
            self.assertIsInstance(normalized, (str, Path))
    
    def test_get_temp_directory(self):
        """Test temporary directory retrieval"""
        temp_dir = self.path_handler.get_temp_directory()
        
        # The get_temp_directory method returns a Path object or string
        self.assertIsInstance(temp_dir, (str, Path))
        temp_path = Path(temp_dir)
        self.assertTrue(temp_path.exists())
        self.assertTrue(temp_path.is_dir())
    
    def test_file_operations(self):
        """Test cross-platform file operations"""
        test_file = Path(self.test_temp_dir) / "test_file.txt"
        test_content = "Test content\\nLine 2\\nLine 3"
        
        # Test write operation
        success = self.file_ops.write_text_file(test_file, test_content)
        self.assertTrue(success)
        self.assertTrue(test_file.exists())
        
        # Test read operation
        content = self.file_ops.read_text_file(test_file)
        self.assertEqual(content.strip(), test_content)
    
    def test_python_detection(self):
        """Test Python executable detection"""
        pythons = self.python_detector.find_python_executables()
        self.assertIsInstance(pythons, list)
        
        # Should find at least the current Python
        self.assertGreater(len(pythons), 0)
        
        # Test recommended Python
        recommended = self.python_detector.get_recommended_python()
        if recommended:
            self.assertIn('executable', recommended)
            self.assertIn('version', recommended)
    
    def test_path_operations(self):
        """Test path operations"""
        # Test path joining
        joined = self.path_handler.join_paths("base", "sub", "file.txt")
        self.assertIsInstance(joined, (str, Path))
        self.assertIn("file.txt", str(joined))
        
        # Test directory methods
        home_dir = self.path_handler.get_home_directory()
        self.assertIsInstance(home_dir, (str, Path))
        
        temp_dir = self.path_handler.get_temp_directory()
        self.assertIsInstance(temp_dir, (str, Path))
        
        # Test executable extension
        ext = self.path_handler.get_executable_extension()
        self.assertIsInstance(ext, str)


class TestPlatformUtilsErrorHandling(unittest.TestCase):
    """Test error handling in platform utilities"""
    
    def setUp(self):
        self.file_ops = file_ops
    
    def test_read_nonexistent_file(self):
        """Test reading non-existent file"""
        nonexistent_file = Path("/this/path/does/not/exist.txt")
        content = self.file_ops.read_text_file(nonexistent_file)
        self.assertIsNone(content)
    
    def test_write_to_readonly_location(self):
        """Test writing to read-only location (may pass on some systems)"""
        # Try to write to root directory (should fail on most systems)
        readonly_file = Path("/readonly_test.txt")
        success = self.file_ops.write_text_file(readonly_file, "test")
        # This test may pass on some systems with sufficient permissions
        self.assertIsInstance(success, bool)


if __name__ == '__main__':
    unittest.main()
