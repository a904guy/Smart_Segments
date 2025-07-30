import unittest
from pathlib import Path
import sys

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
                    # Return a mock array-like structure with shape and tobytes
                    class MockArray:
                        def __init__(self, shape):
                            self.shape = shape
                        
                        def tobytes(self):
                            # Return mock bytes data
                            total_size = 1
                            for dim in self.shape:
                                total_size *= dim
                            return bytes(total_size)
                        
                        def __getitem__(self, key):
                            return self
                        
                        def __setitem__(self, key, value):
                            pass
                        
                        def all(self):
                            return True
                    
                    return MockArray(size)
            return Random()
        
        @staticmethod
        def frombuffer(data, dtype):
            class MockArray:
                def __init__(self):
                    self.shape = (100, 100, 4)
                
                def reshape(self, *args):
                    self.shape = args
                    return self
                
                def __getitem__(self, key):
                    return self
                
                def __setitem__(self, key, value):
                    pass
            return MockArray()
        
        @staticmethod
        def zeros(shape, dtype=None):
            class MockArray:
                def __init__(self, shape):
                    self.shape = shape
                
                def __getitem__(self, key):
                    return self
                
                def __setitem__(self, key, value):
                    pass
                    
                def all(self):
                    return True
            return MockArray(shape)
        
        uint8 = 'uint8'
    
    np = MockNumpy()

# Add the plugin directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "pykrita" / "smart_segments"))

from utils.krita_bridge import KritaImageBridge

class TestKritaImageBridge(unittest.TestCase):
    def setUp(self):
        self.bridge = KritaImageBridge()
        self.sample_layer_data = np.random.randint(0, 256, (100, 100, 4), dtype=np.uint8)

    def test_layer_to_numpy(self):
        """Test conversion of layer data to numpy array"""
        # Mock a layer node and its method
        class MockNode:
            def __init__(self, width, height, data):
                self._width = width
                self._height = height
                self._data = data

            def bounds(self):
                return type('', (), {'x': lambda: 0, 'y': lambda: 0, 'width': lambda: self._width, 'height': lambda: self._height})()

            def projectionPixelData(self, x, y, width, height):
                if width == self._width and height == self._height:
                    return self._data.tobytes()
                return None

        mock_layer = MockNode(100, 100, self.sample_layer_data)
        result = self.bridge.layer_to_numpy(mock_layer)
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, (100, 100, 3))

    def test_apply_mask_to_layer(self):
        """Test applying mask to Krita layer node"""
        mask = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        target_layer_data = np.zeros((100, 100, 4), dtype=np.uint8)

        class MockNode:
            def __init__(self, data):
                self._data = data

            def bounds(self):
                return type('', (), {'x': lambda: 0, 'y': lambda: 0, 'width': lambda: self._data.shape[1], 'height': lambda: self._data.shape[0]})()

            def setPixelData(self, data, x, y, width, height):
                converted_data = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 4)
                self._data[y:y+height, x:x+width] = converted_data

            def document(self):
                return None

        target_layer = MockNode(target_layer_data)
        success = self.bridge.apply_mask_to_layer(mask, target_layer)
        self.assertTrue(success)
        self.assertTrue((target_layer._data[:, :, 3] == mask).all())

if __name__ == '__main__':
    unittest.main()
