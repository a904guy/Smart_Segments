"""
Krita Image Bridge Utilities

This module provides utilities for converting Krita document layers into numpy arrays and
for applying numpy-based segmentation masks back to Krita layers.
"""

import logging
import numpy as np
from typing import Optional, Tuple, Union
from krita import Node, Document, Selection

try:
    from krita import Krita
except ImportError:
    Krita = None


class KritaImageBridge:
    """Bridge between Krita's document/layer API and numpy arrays"""
    
    def __init__(self):
        """Initialize the Krita Image Bridge"""
        self.logger = logging.getLogger(__name__)
        
        # Color space mapping for different formats
        self.color_space_channels = {
            'RGBA': 4,
            'RGB': 3,
            'GRAYA': 2,
            'GRAY': 1,
            'CMYKA': 5,
            'CMYK': 4,
            'LABA': 4,
            'LAB': 3,
            'XYZA': 4,
            'XYZ': 3
        }
    
    def get_layer_info(self, layer: Node) -> dict:
        """
        Get detailed information about a Krita layer.
        
        Args:
            layer (Node): The layer to analyze.
            
        Returns:
            dict: Layer information including dimensions, color space, etc.
        """
        try:
            info = {
                'name': layer.name(),
                'type': layer.type(),
                'visible': layer.visible(),
                'width': layer.bounds().width(),
                'height': layer.bounds().height(),
                'x': layer.bounds().x(),
                'y': layer.bounds().y(),
                'opacity': layer.opacity(),
                'color_model': layer.colorModel(),
                'color_depth': layer.colorDepth(),
                'color_profile': layer.colorProfile()
            }
            
            # Determine number of channels
            color_space_key = f"{info['color_model']}{info['color_depth']}"
            info['channels'] = self.color_space_channels.get(
                info['color_model'], 4  # Default to 4 channels (RGBA)
            )
            
            return info
            
        except Exception as e:
            self.logger.error(f"Failed to get layer info: {e}")
            return {}
    
    def layer_to_numpy(self, layer: Node, convert_to_rgb: bool = True, 
                      normalize: bool = False) -> Optional[np.ndarray]:
        """
        Convert a Krita layer to a numpy array.
        
        Args:
            layer (Node): The layer to convert.
            convert_to_rgb (bool): Convert to RGB format for compatibility.
            normalize (bool): Normalize values to 0-1 range.
            
        Returns:
            Optional[np.ndarray]: The converted image data as a numpy array,
                                 or None if conversion fails.
        """
        try:
            if not layer or not layer.visible():
                self.logger.warning("Layer is None or not visible")
                return None
            
            # Get layer information
            layer_info = self.get_layer_info(layer)
            if not layer_info:
                return None
            
            width = layer_info['width']
            height = layer_info['height']
            channels = layer_info['channels']
            
            if width <= 0 or height <= 0:
                self.logger.warning(f"Invalid layer dimensions: {width}x{height}")
                return None
            
            self.logger.debug(f"Converting layer '{layer_info['name']}' "
                            f"({width}x{height}, {channels} channels)")
            
            # Get pixel data from layer bounds
            bounds = layer.bounds()
            layer_img = layer.projectionPixelData(
                bounds.x(), bounds.y(), width, height
            )
            
            if not layer_img:
                self.logger.error("Failed to get pixel data from layer")
                return None
            
            # Convert to numpy array
            try:
                # Handle different bit depths
                if layer_info['color_depth'] == 'U8':
                    dtype = np.uint8
                elif layer_info['color_depth'] == 'U16':
                    dtype = np.uint16
                elif layer_info['color_depth'] == 'F16':
                    dtype = np.float16
                elif layer_info['color_depth'] == 'F32':
                    dtype = np.float32
                else:
                    dtype = np.uint8  # Default fallback
                
                array_shape = (height, width, channels)
                numpy_array = np.frombuffer(layer_img, dtype=dtype).reshape(array_shape)
                
                # Convert to RGB if requested and layer has alpha channel
                if convert_to_rgb and channels == 4:  # RGBA
                    numpy_array = numpy_array[:, :, :3]  # Remove alpha channel
                elif convert_to_rgb and channels == 2:  # GRAYA
                    # Convert grayscale with alpha to RGB
                    gray = numpy_array[:, :, 0:1]
                    numpy_array = np.repeat(gray, 3, axis=2)
                elif convert_to_rgb and channels == 1:  # GRAY
                    # Convert grayscale to RGB
                    numpy_array = np.repeat(numpy_array, 3, axis=2)
                
                # Normalize if requested
                if normalize:
                    if dtype in [np.uint8]:
                        numpy_array = numpy_array.astype(np.float32) / 255.0
                    elif dtype in [np.uint16]:
                        numpy_array = numpy_array.astype(np.float32) / 65535.0
                    elif dtype in [np.float16, np.float32]:
                        # Assume already normalized or handle appropriately
                        numpy_array = numpy_array.astype(np.float32)
                
                self.logger.debug(f"Successfully converted layer to numpy array: "
                                f"shape={numpy_array.shape}, dtype={numpy_array.dtype}")
                return numpy_array
                
            except Exception as e:
                self.logger.error(f"Failed to convert pixel data to numpy array: {e}")
                return None
            
        except Exception as e:
            self.logger.error(f"Failed to convert layer to numpy: {e}")
            return None
    
    def numpy_to_krita_data(self, array: np.ndarray, 
                           target_color_model: str = 'RGBA',
                           target_depth: str = 'U8') -> Optional[bytes]:
        """
        Convert numpy array to Krita pixel data format.
        
        Args:
            array (np.ndarray): Input numpy array.
            target_color_model (str): Target color model (RGBA, RGB, etc.).
            target_depth (str): Target bit depth (U8, U16, F32).
            
        Returns:
            Optional[bytes]: Converted pixel data or None if conversion fails.
        """
        try:
            if array is None or array.size == 0:
                self.logger.error("Input array is None or empty")
                return None
            
            # Ensure array is in correct format
            if len(array.shape) != 3:
                self.logger.error(f"Array must be 3D, got shape: {array.shape}")
                return None
            
            height, width, channels = array.shape
            
            # Handle different target formats
            if target_color_model == 'RGBA' and channels == 3:
                # Add alpha channel
                if target_depth == 'U8':
                    alpha = np.full((height, width, 1), 255, dtype=np.uint8)
                else:
                    alpha = np.full((height, width, 1), 1.0, dtype=np.float32)
                array = np.concatenate([array, alpha], axis=2)
            elif target_color_model == 'RGB' and channels == 4:
                # Remove alpha channel
                array = array[:, :, :3]
            
            # Convert to target bit depth
            if target_depth == 'U8':
                if array.dtype != np.uint8:
                    if array.dtype in [np.float32, np.float64] and array.max() <= 1.0:
                        array = (array * 255).astype(np.uint8)
                    else:
                        array = array.astype(np.uint8)
            elif target_depth == 'U16':
                if array.dtype != np.uint16:
                    if array.dtype in [np.float32, np.float64] and array.max() <= 1.0:
                        array = (array * 65535).astype(np.uint16)
                    else:
                        array = array.astype(np.uint16)
            elif target_depth == 'F32':
                if array.dtype != np.float32:
                    if array.dtype in [np.uint8]:
                        array = array.astype(np.float32) / 255.0
                    elif array.dtype in [np.uint16]:
                        array = array.astype(np.float32) / 65535.0
                    else:
                        array = array.astype(np.float32)
            
            return array.tobytes()
            
        except Exception as e:
            self.logger.error(f"Failed to convert numpy array to Krita data: {e}")
            return None
    
    def extract_layer_content(self, mask: np.ndarray, source_layer: Node,
                             document: Document,
                             layer_name: str = "Extracted Segment") -> Optional[Node]:
        """
        Extract content from a layer using a mask and create a new layer with the extracted content.
        
        Args:
            mask (np.ndarray): The segmentation mask to use for extraction.
            source_layer (Node): The source layer to extract content from.
            document (Document): The document to create the new layer in.
            layer_name (str): Name for the new layer.
            
        Returns:
            Optional[Node]: The created layer with extracted content or None if operation fails.
        """
        try:
            if mask is None or mask.size == 0:
                self.logger.error("Mask is None or empty")
                return None
            
            if not source_layer or not document:
                self.logger.error("Source layer or document is None")
                return None
            
            self.logger.info(f"Extracting content from layer '{source_layer.name()}' using mask")
            
            # Store the currently active layer
            original_active_layer = document.activeNode()
            
            # Create a selection from the mask
            selection_created = self.create_selection_from_mask(mask, document)
            if not selection_created:
                self.logger.error("Failed to create selection from mask")
                return None
            
            # Make sure the source layer is active
            document.setActiveNode(source_layer)
            
            # Copy the selected content
            from krita import Krita
            app = Krita.instance()
            if app:
                copy_action = app.action('edit_copy')
                if copy_action and copy_action.isEnabled():
                    copy_action.trigger()
                    self.logger.debug("Copied selected content")
                else:
                    self.logger.error("Could not find or use 'edit_copy' action")
                    return None
            else:
                self.logger.error("Could not get Krita application instance")
                return None
            
            # Paste as a new layer
            if app:
                paste_new_layer_action = app.action('paste_as_new_layer')
                if paste_new_layer_action and paste_new_layer_action.isEnabled():
                    paste_new_layer_action.trigger()
                    self.logger.debug("Pasted content as a new layer")
                    new_layer = document.activeNode()  # The new layer becomes active
                    if new_layer and new_layer.name() != source_layer.name():
                        new_layer.setName(layer_name)  # Rename the new layer
                    else:
                        # Handle case where paste action did not create a new layer
                        self.logger.error("Paste action did not create a new layer as expected")
                        return None
                else:
                    self.logger.error("Could not find or use 'paste_as_new_layer' action")
                    return None
            
            # Clear the selection
            document.setSelection(None)
            
            # Restore the original active layer
            if original_active_layer:
                document.setActiveNode(original_active_layer)
            
            self.logger.info(f"Successfully extracted content to layer '{new_layer.name()}'")
            return new_layer
            
        except Exception as e:
            self.logger.error(f"Failed to extract layer content: {e}", exc_info=True)
            return None
    
    def create_selection_from_mask(self, mask: np.ndarray, document: Document) -> bool:
        """
        Create a Krita selection from a numpy mask using a direct data approach.

        Args:
            mask (np.ndarray): Binary or grayscale mask.
            document (Document): Krita document to apply selection to.

        Returns:
            bool: True if selection was created successfully.
        """
        try:
            if mask is None or document is None:
                self.logger.error("Mask or document is None for selection creation.")
                return False

            self.logger.info("Starting direct selection creation from mask.")

            # Convert mask to the format Krita's Selection expects (8-bit grayscale).
            if mask.dtype == np.bool_:
                selection_data_np = (mask * 255).astype(np.uint8)
            elif mask.dtype != np.uint8:
                # Normalize float masks (0.0-1.0) to uint8 (0-255)
                if mask.max() <= 1.0:
                    selection_data_np = (mask * 255).astype(np.uint8)
                else:
                    selection_data_np = mask.astype(np.uint8)
            else:
                selection_data_np = mask

            # Ensure the mask is single-channel.
            if len(selection_data_np.shape) == 3:
                self.logger.warning("Mask has multiple channels; using the first channel for selection.")
                selection_data_np = selection_data_np[:, :, 0]
            
            height, width = selection_data_np.shape
            doc_height = document.height()
            doc_width = document.width()

            if width != doc_width or height != doc_height:
                self.logger.error(f"Mask dimensions ({width}x{height}) do not match document dimensions ({doc_width}x{doc_height}).")
                # Note: In a real-world scenario, we might want to resize/pad the mask.
                return False

            # Create a new Selection object.
            selection = Selection()

            # Get the raw bytes from the numpy array.
            pixel_data = selection_data_np.tobytes()
            
            # Use setPixelData to populate the selection.
            self.logger.debug("Attempting to set pixel data on selection...")
            bounds = document.bounds()
            selection.setPixelData(pixel_data, bounds.x(), bounds.y(), bounds.width(), bounds.height())
            self.logger.debug("Pixel data set on selection object.")
            
            # Apply the selection to the document.
            document.setSelection(selection)
            
            # Verify that the selection was successfully applied.
            if document.selection():
                self.logger.info("Successfully created and applied selection using direct method.")
                return True
            else:
                self.logger.error("Failed to set selection on document after direct creation.")
                return False

        except Exception as e:
            self.logger.error(f"Failed to create selection from mask using direct method: {e}", exc_info=True)
            return False
    
    def get_document_as_numpy(self, document: Document, 
                            flatten: bool = True) -> Optional[np.ndarray]:
        """
        Convert entire Krita document to numpy array.
        
        Args:
            document (Document): Krita document to convert.
            flatten (bool): Whether to flatten all layers into single image.
            
        Returns:
            Optional[np.ndarray]: Document as numpy array or None if conversion fails.
        """
        try:
            if not document:
                return None
            
            if flatten:
                # Get flattened document
                width = document.width()
                height = document.height()
                
                # Get projection (flattened image)
                pixel_data = document.pixelData(0, 0, width, height)
                if not pixel_data:
                    return None
                
                # Convert to numpy array (assuming RGBA)
                array_shape = (height, width, 4)
                numpy_array = np.frombuffer(pixel_data, dtype=np.uint8).reshape(array_shape)
                
                return numpy_array
            else:
                # Return individual layers (not implemented for now)
                self.logger.warning("Individual layer extraction not implemented")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to convert document to numpy: {e}")
            return None

