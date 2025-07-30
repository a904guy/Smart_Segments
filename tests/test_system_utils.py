"""
Unit tests for system utilities
"""

import unittest
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import the modules to test
try:
    from utils.system_utils import (
        SystemResourceManager, PlatformSpecificOperations,
        resource_manager, platform_ops
    )
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "pykrita" / "smart_segments"))
    from utils.system_utils import (
        SystemResourceManager, PlatformSpecificOperations,
        resource_manager, platform_ops
    )


class TestSystemUtils(unittest.TestCase):
    """Test cases for system utilities functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.resource_manager = resource_manager
        self.platform_ops = platform_ops
    
    def test_get_system_info(self):
        """Test system information retrieval"""
        info = self.resource_manager.get_system_info()
        
        self.assertIsNotNone(info)
        self.assertIsInstance(info.os_name, str)
        self.assertIsInstance(info.architecture, str)
        self.assertGreater(info.cpu_count, 0)
        self.assertGreater(info.memory_total, 0)
    
    def test_get_memory_usage(self):
        """Test memory usage retrieval"""
        memory_info = self.resource_manager.get_memory_usage()
        
        self.assertIsInstance(memory_info, dict)
        self.assertIn('total', memory_info)
        self.assertIn('available', memory_info)
        self.assertIn('percent', memory_info)
        
        # Verify values are reasonable
        self.assertGreater(memory_info['total'], 0)
        self.assertGreaterEqual(memory_info['available'], 0)
        self.assertGreaterEqual(memory_info['percent'], 0)
        self.assertLessEqual(memory_info['percent'], 100)
    
    def test_get_cpu_usage(self):
        """Test CPU usage retrieval"""
        cpu_info = self.resource_manager.get_cpu_usage()
        
        self.assertIsInstance(cpu_info, dict)
        self.assertIn('overall', cpu_info)
        self.assertIn('count', cpu_info)
        self.assertGreater(cpu_info['count'], 0)
        self.assertGreaterEqual(cpu_info['overall'], 0)
    
    def test_get_environment_variables(self):
        """Test environment variable retrieval"""
        # Test getting all environment variables
        env_vars = self.platform_ops.get_environment_variables()
        self.assertIsInstance(env_vars, dict)
        self.assertIn('PATH', env_vars)
        
        # Test getting system path
        system_path = self.platform_ops.get_system_path()
        self.assertIsInstance(system_path, list)
        self.assertGreater(len(system_path), 0)
    
    def test_set_environment_variable(self):
        """Test environment variable setting"""
        test_var = 'SMART_SEGMENTS_TEST_VAR'
        test_value = 'test_value_12345'
        
        # Set the variable
        success = self.platform_ops.set_environment_variable(test_var, test_value)
        self.assertTrue(success)
        
        # Verify it was set
        env_vars = self.platform_ops.get_environment_variables()
        self.assertEqual(env_vars.get(test_var), test_value)
        
        # Clean up
        if test_var in os.environ:
            del os.environ[test_var]
    
    def test_platform_operations(self):
        """Test platform-specific operations"""
        # Test getting default browser
        browser = self.platform_ops.get_default_browser()
        if browser:  # May be None if not detectable
            self.assertIsInstance(browser, str)
        
        # Test file associations
        py_associations = self.platform_ops.get_file_associations('.py')
        self.assertIsInstance(py_associations, list)
    
    def test_process_monitoring(self):
        """Test process monitoring functionality"""
        # Test finding processes by name
        python_processes = self.resource_manager.find_processes_by_name("python")
        self.assertIsInstance(python_processes, list)
        
        # Test monitoring current process
        current_pid = os.getpid()
        process_info = self.resource_manager.monitor_process(current_pid)
        if process_info:  # May be None if psutil not available
            self.assertIsNotNone(process_info.name)
            self.assertIsInstance(process_info.memory_mb, (int, float))


class TestResourceManagerFallback(unittest.TestCase):
    """Test resource manager fallback behavior"""
    
    def setUp(self):
        self.resource_manager = resource_manager
    
    def test_disk_usage(self):
        """Test disk usage retrieval"""
        disk_info = self.resource_manager.get_disk_usage()
        
        self.assertIsInstance(disk_info, dict)
        self.assertIn('total', disk_info)
        self.assertIn('free', disk_info)
        self.assertIn('percent', disk_info)
        
        # Verify values are reasonable
        self.assertGreater(disk_info['total'], 0)
        self.assertGreaterEqual(disk_info['free'], 0)
        self.assertGreaterEqual(disk_info['percent'], 0)
        self.assertLessEqual(disk_info['percent'], 100)


if __name__ == '__main__':
    unittest.main()
