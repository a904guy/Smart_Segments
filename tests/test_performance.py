"""
Performance tests for Smart Segments plugin
"""

import unittest
import time
from pathlib import Path
from unittest.mock import MagicMock

# Mock numpy for performance testing without the dependency
class MockNumPy:
    @staticmethod
    def array(data, dtype=None):
        return MockArray(data, dtype)
    
    @staticmethod
    def zeros(shape, dtype=None):
        return MockArray([0] * (shape[0] * shape[1] if len(shape) == 2 else shape[0] * shape[1] * shape[2]), dtype)
    
    @staticmethod
    def random():
        return MockRandom()
    
    uint8 = 'uint8'
    float32 = 'float32'

class MockArray:
    def __init__(self, data, dtype=None):
        self.data = data
        self.dtype = dtype or 'float32'
        if isinstance(data, list):
            self.shape = (len(data),)
        else:
            self.shape = (100, 100, 3)  # Default shape
    
    def reshape(self, shape):
        new_array = MockArray(self.data, self.dtype)
        new_array.shape = shape
        return new_array
    
    def tobytes(self):
        return b'mock_data' * 100
    
    def astype(self, dtype):
        return MockArray(self.data, dtype)
    
    def __mul__(self, other):
        return MockArray(self.data, self.dtype)
    
    def mean(self, axis=None):
        return MockArray([128] * 100, self.dtype)

class MockRandom:
    @staticmethod
    def randint(low, high, size, dtype=None):
        return MockArray([50] * (size[0] * size[1] if len(size) == 2 else size[0] * size[1] * size[2]), dtype)

# Mock numpy module
import sys
sys.modules['numpy'] = MockNumPy()

# Import plugin modules
try:
    from utils.krita_bridge import KritaImageBridge
    from utils.platform_utils import path_handler, file_ops
    from utils.system_utils import resource_manager, platform_ops
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent / "pykrita" / "smart_segments"))
    from utils.krita_bridge import KritaImageBridge
    from utils.platform_utils import path_handler, file_ops
    from utils.system_utils import resource_manager, platform_ops


class MockLayer:
    """Mock Krita layer for performance testing"""
    def __init__(self, width=1000, height=1000):
        self._width = width
        self._height = height
        self._visible = True
        self._name = "Performance Test Layer"
    
    def bounds(self):
        return type('', (), {
            'x': lambda: 0, 
            'y': lambda: 0, 
            'width': lambda: self._width, 
            'height': lambda: self._height
        })()
    
    def visible(self):
        return self._visible
    
    def name(self):
        return self._name
    
    def type(self):
        return "paintLayer"
    
    def opacity(self):
        return 255
    
    def colorModel(self):
        return "RGBA"
    
    def colorDepth(self):
        return "U8"
    
    def colorProfile(self):
        return "sRGB"
    
    def projectionPixelData(self, x, y, width, height):
        return b'test_data' * (width * height * 4)
    
    def setPixelData(self, data, x, y, width, height):
        pass  # Mock implementation
    
    def document(self):
        return MagicMock()


class TestKritaBridgePerformance(unittest.TestCase):
    """Performance tests for KritaImageBridge"""
    
    def setUp(self):
        self.bridge = KritaImageBridge()
        self.performance_results = {}
    
    def tearDown(self):
        """Print performance results"""
        print("\\nPerformance Results:")
        for test_name, duration in self.performance_results.items():
            print(f"  {test_name}: {duration:.4f} seconds")
    
    def measure_time(self, test_name, func, *args, **kwargs):
        """Measure execution time of a function"""
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        duration = end_time - start_time
        self.performance_results[test_name] = duration
        
        return result, duration
    
    def test_layer_conversion_performance(self):
        """Test performance of layer to numpy conversion"""
        # Test with different layer sizes
        layer_sizes = [(100, 100), (500, 500), (1000, 1000)]
        
        for width, height in layer_sizes:
            mock_layer = MockLayer(width, height)
            test_name = f"layer_to_numpy_{width}x{height}"
            
            result, duration = self.measure_time(
                test_name,
                self.bridge.layer_to_numpy,
                mock_layer
            )
            
            # Performance assertion: should complete within reasonable time
            # Adjust these thresholds based on expected performance
            if width * height <= 100 * 100:
                self.assertLess(duration, 0.1, f"Small layer conversion too slow: {duration}s")
            elif width * height <= 500 * 500:
                self.assertLess(duration, 0.5, f"Medium layer conversion too slow: {duration}s")
            else:
                self.assertLess(duration, 2.0, f"Large layer conversion too slow: {duration}s")
    
    def test_mask_application_performance(self):
        """Test performance of mask application"""
        mock_layer = MockLayer(500, 500)
        mock_mask = MockNumPy.random().randint(0, 256, (500, 500), dtype=MockNumPy.uint8)
        
        result, duration = self.measure_time(
            "apply_mask_500x500",
            self.bridge.apply_mask_to_layer,
            mock_mask,
            mock_layer
        )
        
        # Should complete within reasonable time
        self.assertLess(duration, 1.0, f"Mask application too slow: {duration}s")
    
    def test_batch_processing_performance(self):
        """Test performance of processing multiple layers"""
        layers = [MockLayer(200, 200) for _ in range(5)]
        
        start_time = time.time()
        results = []
        
        for i, layer in enumerate(layers):
            result = self.bridge.layer_to_numpy(layer)
            results.append(result)
        
        end_time = time.time()
        batch_duration = end_time - start_time
        
        self.performance_results["batch_processing_5x200x200"] = batch_duration
        
        # Batch processing should be reasonably fast
        self.assertLess(batch_duration, 2.0, f"Batch processing too slow: {batch_duration}s")
        
        # Should process all layers
        self.assertEqual(len(results), len(layers))


class TestPlatformUtilsPerformance(unittest.TestCase):
    """Performance tests for platform utilities"""
    
    def setUp(self):
        self.platform_utils = PlatformUtils()
        self.performance_results = {}
    
    def tearDown(self):
        print("\\nPlatform Utils Performance Results:")
        for test_name, duration in self.performance_results.items():
            print(f"  {test_name}: {duration:.4f} seconds")
    
    def measure_time(self, test_name, func, *args, **kwargs):
        """Measure execution time of a function"""
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        duration = end_time - start_time
        self.performance_results[test_name] = duration
        
        return result, duration
    
    def test_file_operations_performance(self):
        """Test performance of file operations"""
        import tempfile
        temp_dir = tempfile.mkdtemp()
        test_file = Path(temp_dir) / "performance_test.txt"
        
        # Test write performance
        test_content = "Test content\\n" * 1000  # 1000 lines
        
        result, write_duration = self.measure_time(
            "file_write_1000_lines",
            self.platform_utils.write_file,
            test_file,
            test_content
        )
        
        # Test read performance
        result, read_duration = self.measure_time(
            "file_read_1000_lines",
            self.platform_utils.read_file,
            test_file
        )
        
        # Performance assertions
        self.assertLess(write_duration, 0.1, f"File write too slow: {write_duration}s")
        self.assertLess(read_duration, 0.1, f"File read too slow: {read_duration}s")
        
        # Clean up
        import shutil
        shutil.rmtree(temp_dir)
    
    def test_platform_detection_performance(self):
        """Test performance of platform detection"""
        # Platform detection should be very fast
        result, duration = self.measure_time(
            "platform_detection",
            self.platform_utils.get_platform_info
        )
        
        self.assertLess(duration, 0.01, f"Platform detection too slow: {duration}s")
    
    def test_path_operations_performance(self):
        """Test performance of path operations"""
        test_paths = [
            "/home/user/test",
            "C:\\\\Users\\\\User\\\\test",
            "~/Documents/test",
            "./relative/path"
        ] * 100  # Test with 400 paths
        
        start_time = time.time()
        normalized_paths = []
        
        for path in test_paths:
            normalized = self.platform_utils.normalize_path(path)
            normalized_paths.append(normalized)
        
        end_time = time.time()
        duration = end_time - start_time
        
        self.performance_results["path_normalization_400_paths"] = duration
        
        # Should normalize all paths quickly
        self.assertLess(duration, 0.1, f"Path normalization too slow: {duration}s")
        self.assertEqual(len(normalized_paths), len(test_paths))


class TestSystemUtilsPerformance(unittest.TestCase):
    """Performance tests for system utilities"""
    
    def setUp(self):
        self.system_utils = SystemUtils()
        self.performance_results = {}
    
    def tearDown(self):
        print("\\nSystem Utils Performance Results:")
        for test_name, duration in self.performance_results.items():
            print(f"  {test_name}: {duration:.4f} seconds")
    
    def measure_time(self, test_name, func, *args, **kwargs):
        """Measure execution time of a function"""
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        duration = end_time - start_time
        self.performance_results[test_name] = duration
        
        return result, duration
    
    def test_system_info_performance(self):
        """Test performance of system information gathering"""
        result, duration = self.measure_time(
            "system_info_gathering",
            self.system_utils.get_system_info
        )
        
        # System info should be gathered quickly
        self.assertLess(duration, 1.0, f"System info gathering too slow: {duration}s")
    
    def test_environment_variable_performance(self):
        """Test performance of environment variable operations"""
        # Test multiple environment variable operations
        start_time = time.time()
        
        for i in range(100):
            var_name = f"TEST_VAR_{i}"
            self.system_utils.set_environment_variable(var_name, f"value_{i}")
            retrieved = self.system_utils.get_environment_variable(var_name)
        
        end_time = time.time()
        duration = end_time - start_time
        
        self.performance_results["env_vars_100_operations"] = duration
        
        # Environment variable operations should be fast
        self.assertLess(duration, 0.5, f"Environment variable operations too slow: {duration}s")
        
        # Clean up
        import os
        for i in range(100):
            var_name = f"TEST_VAR_{i}"
            if var_name in os.environ:
                del os.environ[var_name]


if __name__ == '__main__':
    # Run performance tests with increased verbosity
    unittest.main(verbosity=2)
