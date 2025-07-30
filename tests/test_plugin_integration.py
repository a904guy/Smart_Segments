"""
Integration tests for Smart Segments plugin components
"""

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    # Mock numpy for testing when not available
    class MockNumpy:
        @staticmethod
        def random():
            class Random:
                @staticmethod
                def randint(low, high, size, dtype):
                    if isinstance(size, tuple) and len(size) == 3:
                        return [[[0, 0, 0, 255] for _ in range(size[1])] for _ in range(size[0])]
                    return [[0] * size[1] for _ in range(size[0])]
            return Random()
        
        @staticmethod
        def frombuffer(data, dtype):
            class MockArray:
                def reshape(self, *args):
                    return self
            return MockArray()
        
        @staticmethod
        def sum(arr):
            return 100  # Mock sum
        
        @staticmethod
        def logical_or():
            class LogicalOr:
                @staticmethod
                def reduce(arrays):
                    return arrays[0] if arrays else None
            return LogicalOr()
        
        uint8 = 'uint8'
    
    np = MockNumpy()

# Mock Krita API for testing
class MockKrita:
    @staticmethod
    def instance():
        return MagicMock()

class MockNode:
    def __init__(self, width=100, height=100):
        self._width = width
        self._height = height
        self._data = np.random.randint(0, 256, (height, width, 4), dtype=np.uint8)
        self._visible = True
        self._name = "Test Layer"
    
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
        return self._data.tobytes()
    
    def setPixelData(self, data, x, y, width, height):
        self._data = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 4)
    
    def document(self):
        return MagicMock()

# Mock krita module
import sys
sys.modules['krita'] = MockKrita()

# Now import our modules
try:
    from utils.krita_bridge import KritaImageBridge
    from utils.platform_utils import path_handler, file_ops
    from utils.system_utils import resource_manager, platform_ops
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent / "pykrita" / "smart_segments"))
    from utils.krita_bridge import KritaImageBridge
    from utils.platform_utils import path_handler, file_ops
    from utils.system_utils import resource_manager, platform_ops


class TestPluginIntegration(unittest.TestCase):
    """Test integration between plugin components"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.krita_bridge = KritaImageBridge()
        self.path_handler = path_handler
        self.file_ops = file_ops
        self.resource_manager = resource_manager
        self.platform_ops = platform_ops
        
        # Create mock layer
        self.mock_layer = MockNode(100, 100)
    
    def test_krita_bridge_platform_integration(self):
        """Test integration between KritaImageBridge and platform utilities"""
        # Convert layer to numpy array
        image_array = self.krita_bridge.layer_to_numpy(self.mock_layer)
        self.assertIsNotNone(image_array)
        
        # Use path handler to get temp directory
        temp_dir_str = self.path_handler.get_temp_directory()
        temp_dir = Path(temp_dir_str) / "test_integration"
        temp_dir.mkdir(exist_ok=True)
        
        # Save image data info using file operations
        info_file = temp_dir / "image_info.txt"
        image_info = f"Shape: {image_array.shape}\\nDtype: {image_array.dtype}"
        success = self.file_ops.write_text_file(info_file, image_info)
        self.assertTrue(success)
        
        # Read back the info
        read_info = self.file_ops.read_text_file(info_file)
        self.assertIn("Shape:", read_info)
        
        # Clean up
        import shutil
        shutil.rmtree(temp_dir)
    
    def test_system_bridge_integration(self):
        """Test integration between system utilities and image processing"""
        # Get system info
        system_info = self.resource_manager.get_system_info()
        self.assertIsNotNone(system_info)
        
        # Process image based on system capabilities
        image_array = self.krita_bridge.layer_to_numpy(self.mock_layer)
        
        # Simulate processing based on available memory
        memory_info = self.resource_manager.get_memory_usage()
        if memory_info.get('available', 0) > 1000:  # 1MB in KB
            # Process full image
            processed_array = image_array * 0.8  # Simple processing
            self.assertEqual(processed_array.shape, image_array.shape)
        else:
            # Process in chunks (fallback)
            processed_array = image_array[:50, :50] * 0.8
            self.assertEqual(processed_array.shape, (50, 50, 3))
    
    def test_cross_platform_file_handling(self):
        """Test cross-platform file handling in integration"""
        # Get platform info
        platform_type = self.path_handler.platform
        
        # Create mask array
        mask = np.random.randint(0, 2, (100, 100), dtype=np.uint8) * 255
        
        # Apply mask using platform-appropriate methods
        if platform_type.value == 'windows':
            # Windows-specific handling
            mask_layer = self.krita_bridge.apply_mask_to_layer(
                mask, self.mock_layer, create_new_layer=True
            )
        else:
            # Unix-like systems
            mask_layer = self.krita_bridge.apply_mask_to_layer(
                mask, self.mock_layer, create_new_layer=True
            )
        
        # Should work regardless of platform
        self.assertIsNotNone(mask_layer)
    
    def test_environment_aware_processing(self):
        """Test processing that adapts to environment"""
        # Set test environment variable
        self.platform_ops.set_environment_variable('SMART_SEGMENTS_TEST', 'true')
        
        # Process differently based on environment
        image_array = self.krita_bridge.layer_to_numpy(self.mock_layer)
        
        # Check environment variables
        env_vars = self.platform_ops.get_environment_variables()
        if env_vars.get('SMART_SEGMENTS_TEST') == 'true':
            # Test mode - use smaller processing
            small_array = image_array[:10, :10]
            self.assertEqual(small_array.shape, (10, 10, 3))
        
        # Clean up environment
        import os
        if 'SMART_SEGMENTS_TEST' in os.environ:
            del os.environ['SMART_SEGMENTS_TEST']
    
    def test_error_handling_integration(self):
        """Test error handling across integrated components"""
        # Test with invalid layer
        invalid_layer = None
        result = self.krita_bridge.layer_to_numpy(invalid_layer)
        self.assertIsNone(result)
        
        # Test with invalid file path
        invalid_path = Path("/invalid/path/that/does/not/exist.txt")
        content = self.file_ops.read_text_file(invalid_path)
        self.assertIsNone(content)
        
        # Test with invalid environment variable
        env_vars = self.platform_ops.get_environment_variables()
        invalid_env = env_vars.get('INVALID_VAR_NAME_12345')
        self.assertIsNone(invalid_env)


class TestComponentCommunication(unittest.TestCase):
    """Test communication between different plugin components"""
    
    def setUp(self):
        self.krita_bridge = KritaImageBridge()
        self.path_handler = path_handler
        self.file_ops = file_ops
        self.resource_manager = resource_manager
    
    def test_data_flow_pipeline(self):
        """Test complete data flow through the plugin pipeline"""
        # Step 1: Get image from Krita layer
        mock_layer = MockNode(50, 50)
        image_array = self.krita_bridge.layer_to_numpy(mock_layer)
        self.assertIsNotNone(image_array)
        
        # Step 2: Process using system-aware methods
        system_info = self.resource_manager.get_system_info()
        cpu_count = system_info.cpu_count
        
        # Simulate processing based on CPU count
        if cpu_count > 1:
            # Multi-core processing simulation
            processed_array = image_array * 1.2
        else:
            # Single-core processing
            processed_array = image_array * 1.0
        
        # Step 3: Create mask and apply back to layer
        mask = (processed_array.mean(axis=2) > 128).astype(np.uint8) * 255
        mask_layer = self.krita_bridge.apply_mask_to_layer(mask, mock_layer)
        self.assertIsNotNone(mask_layer)
        
        # Step 4: Save results using platform utilities
        temp_dir_str = self.path_handler.get_temp_directory()
        temp_dir = Path(temp_dir_str) / "pipeline_test"
        temp_dir.mkdir(exist_ok=True)
        result_file = temp_dir / "result.txt"
        
        result_info = f"Processing complete\\nMask pixels: {np.sum(mask > 0)}"
        success = self.file_ops.write_text_file(result_file, result_info)
        self.assertTrue(success)
        
        # Clean up
        import shutil
        shutil.rmtree(temp_dir)


if __name__ == '__main__':
    unittest.main()
