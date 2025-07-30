"""
Selection & Layer Conversion Tools

This module provides tools for converting segmentation masks to Krita selections,
generating layer masks, extracting layers, and managing undo/redo operations.
"""

import logging
import numpy as np
from typing import Optional, List, Tuple, Dict, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import time

from krita import Krita, Document, Node, Selection
from PyQt5.QtCore import QObject, pyqtSignal

from .segmentation_api import SegmentationResult


class SelectionOperationType(Enum):
    """Types of selection operations"""
    REPLACE = "replace"          # Replace current selection
    ADD = "add"                  # Add to current selection
    SUBTRACT = "subtract"        # Subtract from current selection
    INTERSECT = "intersect"      # Intersect with current selection


class LayerExtractionMode(Enum):
    """Modes for layer extraction"""
    NEW_LAYER = "new_layer"      # Create new layer with extracted content
    DUPLICATE = "duplicate"      # Duplicate layer with mask applied
    MASK_ONLY = "mask_only"      # Create transparency mask only
    SELECTION_ONLY = "selection_only"  # Convert to selection only


@dataclass
class SelectionOptions:
    """Options for selection generation"""
    feather_radius: int = 0
    expand_pixels: int = 0
    contract_pixels: int = 0
    smooth_iterations: int = 0
    anti_alias: bool = True
    threshold: float = 0.5
    invert: bool = False


@dataclass
class LayerExtractionOptions:
    """Options for layer extraction"""
    extraction_mode: LayerExtractionMode = LayerExtractionMode.NEW_LAYER
    layer_name: Optional[str] = None
    preserve_original: bool = True
    blend_mode: str = "normal"
    opacity: int = 255
    create_group: bool = False
    group_name: Optional[str] = None
    apply_selection_options: bool = True
    selection_options: SelectionOptions = field(default_factory=SelectionOptions)


@dataclass
class SelectionOperation:
    """Represents a selection operation for undo/redo"""
    operation_id: str
    operation_type: str
    timestamp: float
    document: Document
    original_mask: Optional[np.ndarray]
    selection_options: SelectionOptions
    layer_options: Optional[LayerExtractionOptions]
    created_nodes: List[Node] = field(default_factory=list)
    modified_nodes: List[Node] = field(default_factory=list)
    original_selection: Optional[bytes] = None  # Serialized selection data


class SelectionManager(QObject):
    """
    Manages selection and layer conversion operations with undo/redo support
    """
    
    # Signals
    selection_created = pyqtSignal(object)  # Selection object
    layer_extracted = pyqtSignal(object)    # Node object
    operation_completed = pyqtSignal(str)   # Operation ID
    error_occurred = pyqtSignal(str)        # Error message
    
    def __init__(self, max_undo_levels: int = 50):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        
        # Undo/redo management
        self.max_undo_levels = max_undo_levels
        self.operation_history: List[SelectionOperation] = []
        self.current_operation_index = -1
        
        # Statistics
        self.total_operations = 0
        self.successful_operations = 0
        
        self.logger.info("Selection Manager initialized")
    
    def mask_to_selection(self, 
                         mask: np.ndarray,
                         document: Document,
                         operation_type: SelectionOperationType = SelectionOperationType.REPLACE,
                         options: Optional[SelectionOptions] = None) -> Optional[Selection]:
        """
        Convert a segmentation mask to a Krita selection
        
        Args:
            mask: Binary segmentation mask
            document: Target Krita document
            operation_type: How to combine with existing selection
            options: Selection generation options
            
        Returns:
            Created Selection object or None if failed
        """
        if options is None:
            options = SelectionOptions()
            
        operation_id = f"selection_{int(time.time() * 1000)}"
        self.logger.info(f"Starting selection creation process: {operation_id}")
        
        # Validate inputs
        if mask is None:
            error_msg = f"[{operation_id}] Input mask is None"
            self.logger.error(error_msg)
            self.error_occurred.emit(error_msg)
            return None
            
        if document is None:
            error_msg = f"[{operation_id}] Input document is None"
            self.logger.error(error_msg)
            self.error_occurred.emit(error_msg)
            return None
            
        self.logger.debug(f"[{operation_id}] Input validation passed - mask shape: {mask.shape}, document size: {document.width()}x{document.height()}")
        
        try:
            self.total_operations += 1
            
            # Store original selection for undo
            self.logger.debug(f"[{operation_id}] Storing original selection for undo")
            original_selection = None
            try:
                if document.selection():
                    original_selection = document.selection().toByteArray()
                    self.logger.debug(f"[{operation_id}] Original selection stored ({len(original_selection)} bytes)")
                else:
                    self.logger.debug(f"[{operation_id}] No existing selection to store")
            except Exception as e:
                self.logger.warning(f"[{operation_id}] Failed to store original selection: {e}")
            
            # Create operation record
            operation = SelectionOperation(
                operation_id=operation_id,
                operation_type="mask_to_selection",
                timestamp=time.time(),
                document=document,
                original_mask=mask.copy(),
                selection_options=options,
                layer_options=None,
                original_selection=original_selection
            )
            self.logger.debug(f"[{operation_id}] Operation record created")
            
            # Process mask
            self.logger.info(f"[{operation_id}] Processing mask for selection (threshold: {options.threshold}, expand: {options.expand_pixels}, contract: {options.contract_pixels})")
            processed_mask = self._process_mask_for_selection(mask, options)
            if processed_mask is None:
                error_msg = f"[{operation_id}] Failed to process mask for selection - check mask format and options"
                self.logger.error(error_msg)
                self.error_occurred.emit(error_msg)
                self._cleanup_failed_operation(operation)
                return None
            
            self.logger.debug(f"[{operation_id}] Mask processed successfully - output shape: {processed_mask.shape}, selected pixels: {np.sum(processed_mask)}")
            
            # Convert mask to selection
            self.logger.info(f"[{operation_id}] Converting processed mask to Krita selection")
            selection = self._create_selection_from_mask(processed_mask, document)
            if selection is None:
                error_msg = f"[{operation_id}] Failed to create selection from mask - check Krita API compatibility"
                self.logger.error(error_msg)
                self.error_occurred.emit(error_msg)
                self._cleanup_failed_operation(operation)
                return None
            
            self.logger.debug(f"[{operation_id}] Selection object created successfully")
            
            # Apply selection operation
            self.logger.info(f"[{operation_id}] Applying selection operation: {operation_type.value}")
            success = self._apply_selection_operation(
                document, selection, operation_type
            )
            
            if not success:
                error_msg = f"[{operation_id}] Failed to apply selection operation - check document state"
                self.logger.error(error_msg)
                self.error_occurred.emit(error_msg)
                self._cleanup_failed_operation(operation)
                return None
            
            self.logger.debug(f"[{operation_id}] Selection operation applied successfully")
            
            # Record operation for undo
            self.logger.debug(f"[{operation_id}] Recording operation for undo/redo")
            self._record_operation(operation)
            
            self.successful_operations += 1
            self.selection_created.emit(selection)
            self.operation_completed.emit(operation_id)
            
            self.logger.info(f"[{operation_id}] Selection created successfully - Total operations: {self.total_operations}, Success rate: {(self.successful_operations/self.total_operations)*100:.1f}%")
            return selection
            
        except Exception as e:
            error_msg = f"[{operation_id}] Unexpected error during selection creation: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            self.error_occurred.emit(error_msg)
            try:
                self._cleanup_failed_operation(operation)
            except:
                self.logger.warning(f"[{operation_id}] Failed to cleanup after error")
            return None
    
    def extract_layer(self,
                     mask: np.ndarray,
                     source_layer: Node,
                     document: Document,
                     options: Optional[LayerExtractionOptions] = None) -> Optional[Node]:
        """
        Extract a layer using a segmentation mask
        
        Args:
            mask: Segmentation mask
            source_layer: Source layer to extract from
            document: Target document
            options: Layer extraction options
            
        Returns:
            Created/modified Node or None if failed
        """
        if options is None:
            options = LayerExtractionOptions()
            
        try:
            self.total_operations += 1
            operation_id = f"extract_{int(time.time() * 1000)}"
            
            # Store original selection for undo
            original_selection = None
            if document.selection():
                original_selection = document.selection().toByteArray()
            
            # Create operation record
            operation = SelectionOperation(
                operation_id=operation_id,
                operation_type="extract_layer",
                timestamp=time.time(),
                document=document,
                original_mask=mask.copy(),
                selection_options=options.selection_options,
                layer_options=options,
                original_selection=original_selection
            )
            
            # Create selection if needed
            selection = None
            if options.extraction_mode != LayerExtractionMode.SELECTION_ONLY:
                if options.apply_selection_options:
                    selection = self.mask_to_selection(
                        mask, document, SelectionOperationType.REPLACE, 
                        options.selection_options
                    )
                else:
                    processed_mask = self._process_mask_for_selection(
                        mask, SelectionOptions()
                    )
                    selection = self._create_selection_from_mask(processed_mask, document)
                
                if selection is None:
                    self.logger.error("Failed to create selection for layer extraction")
                    return None
            
            # Perform extraction based on mode
            result_node = None
            if options.extraction_mode == LayerExtractionMode.NEW_LAYER:
                result_node = self._extract_to_new_layer(
                    source_layer, document, selection, options
                )
            elif options.extraction_mode == LayerExtractionMode.DUPLICATE:
                result_node = self._extract_as_duplicate(
                    source_layer, document, selection, options
                )
            elif options.extraction_mode == LayerExtractionMode.MASK_ONLY:
                result_node = self._apply_transparency_mask(
                    source_layer, document, selection, options
                )
            elif options.extraction_mode == LayerExtractionMode.SELECTION_ONLY:
                # Just create the selection
                selection = self.mask_to_selection(
                    mask, document, SelectionOperationType.REPLACE,
                    options.selection_options
                )
                result_node = source_layer  # Return source layer as reference
            
            if result_node is None:
                self.logger.error("Layer extraction failed")
                return None
            
            # Record created/modified nodes
            if result_node != source_layer:
                operation.created_nodes.append(result_node)
            else:
                operation.modified_nodes.append(result_node)
            
            # Record operation for undo
            self._record_operation(operation)
            
            self.successful_operations += 1
            self.layer_extracted.emit(result_node)
            self.operation_completed.emit(operation_id)
            
            self.logger.info(f"Layer extracted successfully: {operation_id}")
            return result_node
            
        except Exception as e:
            self.logger.error(f"Failed to extract layer: {e}")
            self.error_occurred.emit(str(e))
            return None
    
    def create_mask_layer(self,
                         mask: np.ndarray,
                         document: Document,
                         layer_name: Optional[str] = None,
                         options: Optional[SelectionOptions] = None) -> Optional[Node]:
        """
        Create a new layer containing the segmentation mask
        
        Args:
            mask: Segmentation mask
            document: Target document
            layer_name: Name for the new layer
            options: Selection options for mask processing
            
        Returns:
            Created mask layer or None if failed
        """
        if options is None:
            options = SelectionOptions()
            
        if layer_name is None:
            layer_name = f"Segment Mask {int(time.time())}"
            
        try:
            self.total_operations += 1
            operation_id = f"mask_layer_{int(time.time() * 1000)}"
            
            # Process mask
            processed_mask = self._process_mask_for_selection(mask, options)
            if processed_mask is None:
                self.logger.error("Failed to process mask")
                return None
            
            # Create new paint layer
            mask_layer = document.createNode(layer_name, "paintlayer")
            if mask_layer is None:
                self.logger.error("Failed to create mask layer")
                return None
            
            # Convert mask to image data
            mask_data = self._mask_to_image_data(processed_mask, document)
            if mask_data is None:
                self.logger.error("Failed to convert mask to image data")
                return None
            
            # Set layer data
            bounds = document.bounds()
            mask_layer.setPixelData(
                mask_data, 
                bounds.x(), bounds.y(), 
                bounds.width(), bounds.height()
            )
            
            # Add layer to document
            root_node = document.rootNode()
            root_node.addChildNode(mask_layer, None)
            
            # Record operation
            operation = SelectionOperation(
                operation_id=operation_id,
                operation_type="create_mask_layer",
                timestamp=time.time(),
                document=document,
                original_mask=mask.copy(),
                selection_options=options,
                layer_options=None,
                created_nodes=[mask_layer]
            )
            self._record_operation(operation)
            
            self.successful_operations += 1
            self.layer_extracted.emit(mask_layer)
            self.operation_completed.emit(operation_id)
            
            self.logger.info(f"Mask layer created successfully: {operation_id}")
            return mask_layer
            
        except Exception as e:
            self.logger.error(f"Failed to create mask layer: {e}")
            self.error_occurred.emit(str(e))
            return None
    
    def _process_mask_for_selection(self, 
                                  mask: np.ndarray, 
                                  options: SelectionOptions) -> Optional[np.ndarray]:
        """
        Process mask according to selection options
        
        Args:
            mask: Input mask
            options: Processing options
            
        Returns:
            Processed mask or None if failed
        """
        try:
            self.logger.debug(f"Processing mask - input shape: {mask.shape}, dtype: {mask.dtype}")
            
            # Validate input mask
            if mask.size == 0:
                self.logger.error("Input mask is empty")
                return None
            
            if len(mask.shape) < 2:
                self.logger.error(f"Input mask has invalid shape: {mask.shape}")
                return None
            
            # Ensure mask is binary
            if mask.dtype != np.bool_:
                self.logger.debug(f"Converting mask from {mask.dtype} to binary using threshold {options.threshold}")
                processed_mask = mask > options.threshold
                self.logger.debug(f"Binary conversion - selected pixels: {np.sum(processed_mask)}/{mask.size}")
            else:
                processed_mask = mask.copy()
                self.logger.debug(f"Mask already binary - selected pixels: {np.sum(processed_mask)}/{mask.size}")
            
            # Apply morphological operations
            if options.expand_pixels > 0:
                self.logger.debug(f"Expanding mask by {options.expand_pixels} pixels")
                original_count = np.sum(processed_mask)
                processed_mask = self._dilate_mask(processed_mask, options.expand_pixels)
                new_count = np.sum(processed_mask)
                self.logger.debug(f"Expansion result - pixels changed from {original_count} to {new_count}")
            
            if options.contract_pixels > 0:
                self.logger.debug(f"Contracting mask by {options.contract_pixels} pixels")
                original_count = np.sum(processed_mask)
                processed_mask = self._erode_mask(processed_mask, options.contract_pixels)
                new_count = np.sum(processed_mask)
                self.logger.debug(f"Contraction result - pixels changed from {original_count} to {new_count}")
            
            # Apply smoothing
            if options.smooth_iterations > 0:
                self.logger.debug(f"Smoothing mask with {options.smooth_iterations} iterations")
                original_count = np.sum(processed_mask)
                processed_mask = self._smooth_mask(processed_mask, options.smooth_iterations)
                new_count = np.sum(processed_mask)
                self.logger.debug(f"Smoothing result - pixels changed from {original_count} to {new_count}")
            
            # Invert if requested
            if options.invert:
                self.logger.debug("Inverting mask")
                original_count = np.sum(processed_mask)
                processed_mask = ~processed_mask
                new_count = np.sum(processed_mask)
                self.logger.debug(f"Inversion result - pixels changed from {original_count} to {new_count}")
            
            # Final validation
            if np.sum(processed_mask) == 0:
                self.logger.warning("Processed mask has no selected pixels - selection will be empty")
            
            self.logger.debug(f"Mask processing completed - final shape: {processed_mask.shape}, selected pixels: {np.sum(processed_mask)}")
            return processed_mask
            
        except Exception as e:
            self.logger.error(f"Failed to process mask: {e}", exc_info=True)
            return None
    
    def _create_selection_from_mask(self, 
                                  mask: np.ndarray, 
                                  document: Document) -> Optional[Selection]:
        """
        Create a Krita Selection from a binary mask using direct Selection object creation
        
        Args:
            mask: Binary mask
            document: Target document
            
        Returns:
            Selection object or None if failed
        """
        try:
            self.logger.debug(f"Creating selection from mask - shape: {mask.shape}, selected pixels: {np.sum(mask)}")
            
            # Validate inputs
            if mask is None or mask.size == 0:
                self.logger.error("Invalid mask provided for selection creation")
                return None
            
            if document is None:
                self.logger.error("Invalid document provided for selection creation")
                return None
            
            # Create a new Selection object from the Krita API
            try:
                selection = Selection()
                self.logger.debug("Selection object created successfully")
            except Exception as e:
                self.logger.error(f"Failed to create Selection object: {e}")
                return None
            
            # Convert the numpy mask to the appropriate format for Krita selection data
            # Krita expects grayscale uint8 data where 255 = selected, 0 = not selected
            self.logger.debug("Converting mask to Krita format")
            selection_data = self._convert_mask_to_krita_format(mask, document)
            if selection_data is None:
                self.logger.error("Failed to convert mask to Krita format")
                return None
            
            self.logger.debug(f"Mask converted to Krita format - data size: {len(selection_data)} bytes")
            
            # Get document bounds for proper positioning
            try:
                bounds = document.bounds()
                self.logger.debug(f"Document bounds: x={bounds.x()}, y={bounds.y()}, w={bounds.width()}, h={bounds.height()}")
            except Exception as e:
                self.logger.error(f"Failed to get document bounds: {e}")
                return None
            
            # Use setPixelData() to populate the selection with the mask data
            try:
                self.logger.debug("Attempting to use setPixelData method")
                selection.setPixelData(
                    selection_data,
                    bounds.x(), bounds.y(),
                    bounds.width(), bounds.height()
                )
                self.logger.debug("Selection data set successfully using setPixelData")
            except AttributeError as e:
                # Fallback: some Krita versions may not have setPixelData on Selection
                # Try using fromByteArray as alternative
                self.logger.warning(f"setPixelData not available on Selection (AttributeError: {e}), using fromByteArray fallback")
                try:
                    # Convert to proper byte array format for fromByteArray
                    byte_data = self._mask_to_selection_byte_array(mask)
                    if byte_data is None:
                        self.logger.error("Failed to create byte array for fromByteArray fallback")
                        return None
                    
                    self.logger.debug(f"Using fromByteArray fallback with {len(byte_data)} bytes")
                    selection.fromByteArray(byte_data)
                    self.logger.debug("Selection data set successfully using fromByteArray fallback")
                except Exception as fallback_e:
                    self.logger.error(f"fromByteArray fallback also failed: {fallback_e}")
                    return None
            except Exception as e:
                self.logger.error(f"Failed to set selection pixel data: {e}")
                return None
            
            # Validate the created selection
            try:
                # Check if selection has any content
                if hasattr(selection, 'isEmpty') and selection.isEmpty():
                    self.logger.warning("Created selection is empty")
                else:
                    self.logger.debug("Selection validation passed")
            except Exception as e:
                self.logger.warning(f"Could not validate selection: {e}")
            
            self.logger.debug("Selection created successfully from mask")
            return selection
            
        except Exception as e:
            self.logger.error(f"Unexpected error creating selection from mask: {e}", exc_info=True)
            return None
    
    def _apply_selection_operation(self,
                                 document: Document,
                                 selection: Selection,
                                 operation_type: SelectionOperationType) -> bool:
        """
        Apply selection operation to document
        
        Args:
            document: Target document
            selection: Selection to apply
            operation_type: Type of operation
            
        Returns:
            True if successful
        """
        try:
            current_selection = document.selection()
            
            if operation_type == SelectionOperationType.REPLACE or current_selection is None:
                document.setSelection(selection)
            elif operation_type == SelectionOperationType.ADD:
                current_selection.add(selection)
                document.setSelection(current_selection)
            elif operation_type == SelectionOperationType.SUBTRACT:
                current_selection.subtract(selection)
                document.setSelection(current_selection)
            elif operation_type == SelectionOperationType.INTERSECT:
                current_selection.intersect(selection)
                document.setSelection(current_selection)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to apply selection operation: {e}")
            return False
    
    def _extract_to_new_layer(self,
                            source_layer: Node,
                            document: Document,
                            selection: Selection,
                            options: LayerExtractionOptions) -> Optional[Node]:
        """
        Extract selection to a new layer
        
        Args:
            source_layer: Source layer
            document: Target document
            selection: Selection to extract
            options: Extraction options
            
        Returns:
            New layer or None if failed
        """
        try:
            # Set the selection
            document.setSelection(selection)
            
            # Copy selection content
            source_layer.copy()
            
            # Create new layer
            layer_name = options.layer_name or f"Extracted {source_layer.name()}"
            new_layer = document.createNode(layer_name, "paintlayer")
            
            if new_layer is None:
                return None
            
            # Add to document
            parent = source_layer.parentNode()
            if options.create_group and options.group_name:
                group = document.createNode(options.group_name, "grouplayer")
                parent.addChildNode(group, source_layer)
                group.addChildNode(new_layer, None)
            else:
                parent.addChildNode(new_layer, source_layer)
            
            # Paste content
            new_layer.paste()
            
            # Set layer properties
            new_layer.setBlendingMode(options.blend_mode)
            new_layer.setOpacity(options.opacity)
            
            # Clear selection
            document.setSelection(None)
            
            return new_layer
            
        except Exception as e:
            self.logger.error(f"Failed to extract to new layer: {e}")
            return None
    
    def _extract_as_duplicate(self,
                            source_layer: Node,
                            document: Document,
                            selection: Selection,
                            options: LayerExtractionOptions) -> Optional[Node]:
        """
        Extract as a duplicate layer with mask applied
        
        Args:
            source_layer: Source layer
            document: Target document
            selection: Selection to use as mask
            options: Extraction options
            
        Returns:
            Duplicated layer or None if failed
        """
        try:
            # Duplicate the source layer
            duplicated = source_layer.duplicate()
            if duplicated is None:
                return None
            
            # Set name
            layer_name = options.layer_name or f"{source_layer.name()} Copy"
            duplicated.setName(layer_name)
            
            # Add to document
            parent = source_layer.parentNode()
            parent.addChildNode(duplicated, source_layer)
            
            # Apply selection as transparency mask
            document.setSelection(selection)
            
            # Invert selection to mask out non-selected areas
            inverted_selection = Selection()
            inverted_selection.copy(selection)
            inverted_selection.invert()
            
            # Apply mask
            document.setSelection(inverted_selection)
            duplicated.cutToLayer()
            
            # Clear selection
            document.setSelection(None)
            
            # Set layer properties
            duplicated.setBlendingMode(options.blend_mode)
            duplicated.setOpacity(options.opacity)
            
            return duplicated
            
        except Exception as e:
            self.logger.error(f"Failed to extract as duplicate: {e}")
            return None
    
    def _apply_transparency_mask(self,
                               source_layer: Node,
                               document: Document,
                               selection: Selection,
                               options: LayerExtractionOptions) -> Optional[Node]:
        """
        Apply transparency mask to source layer
        
        Args:
            source_layer: Source layer to modify
            document: Target document
            selection: Selection to use as mask
            options: Options (mostly ignored for this mode)
            
        Returns:
            Modified source layer or None if failed
        """
        try:
            # Set selection
            document.setSelection(selection)
            
            # Invert selection to mask out non-selected areas
            inverted_selection = Selection()
            inverted_selection.copy(selection)
            inverted_selection.invert()
            
            # Apply transparency
            document.setSelection(inverted_selection)
            source_layer.cutToLayer()
            
            # Clear selection
            document.setSelection(None)
            
            return source_layer
            
        except Exception as e:
            self.logger.error(f"Failed to apply transparency mask: {e}")
            return None
    
    def _convert_mask_to_krita_format(self, mask: np.ndarray, document: Document) -> Optional[bytes]:
        """
        Convert numpy mask to Krita's expected grayscale uint8 format for selection
        
        Args:
            mask: Binary mask
            document: Target document
            
        Returns:
            Grayscale image data bytes or None if failed
        """
        try:
            height, width = mask.shape[:2]
            doc_height = document.height()
            doc_width = document.width()
            
            # Ensure mask dimensions match document dimensions
            if height != doc_height or width != doc_width:
                self.logger.warning(
                    f"Mask dimensions ({width}x{height}) differ from document "
                    f"({doc_width}x{doc_height}). Resizing mask."
                )
                # Resize mask to fit document - requires a library like OpenCV or Pillow
                # For now, we will pad or crop the mask
                resized_mask = np.zeros((doc_height, doc_width), dtype=np.uint8)
                min_h, min_w = min(height, doc_height), min(width, doc_width)
                resized_mask[:min_h, :min_w] = (mask[:min_h, :min_w] * 255).astype(np.uint8)
            else:
                resized_mask = (mask * 255).astype(np.uint8)
                
            return resized_mask.tobytes()
            
        except Exception as e:
            self.logger.error(f"Failed to convert mask to Krita format: {e}")
            return None

    def _mask_to_selection_byte_array(self, mask: np.ndarray) -> Optional[bytes]:
        """
        Convert binary mask to a byte array for fallback Selection.fromByteArray
        This should be used when setPixelData is not available.
        
        Args:
            mask: Binary mask
            
        Returns:
            Byte array for selection or None if failed
        """
        try:
            # This is a simplified version of what Krita might expect for a
            # basic bitmap selection. The exact format could be more complex.
            height, width = mask.shape[:2]
            selection_bitmap = (mask * 255).astype(np.uint8)
            return selection_bitmap.tobytes()
            
        except Exception as e:
            self.logger.error(f"Failed to create selection byte array: {e}")
            return None
    
    def _mask_to_image_data(self, mask: np.ndarray, document: Document) -> Optional[bytes]:
        """
        Convert mask to image data for layer creation
        
        Args:
            mask: Binary mask
            document: Target document
            
        Returns:
            Image data bytes or None if failed
        """
        try:
            height, width = mask.shape[:2]
            
            # Create RGBA image data (white mask with alpha)
            image_data = np.zeros((height, width, 4), dtype=np.uint8)
            
            # Set RGB to white where mask is True
            image_data[mask, :3] = 255
            # Set alpha to 255 where mask is True
            image_data[mask, 3] = 255
            
            return image_data.tobytes()
            
        except Exception as e:
            self.logger.error(f"Failed to convert mask to image data: {e}")
            return None
    
    def _dilate_mask(self, mask: np.ndarray, pixels: int) -> np.ndarray:
        """
        Dilate (expand) mask by specified pixels
        
        Args:
            mask: Binary mask
            pixels: Number of pixels to expand
            
        Returns:
            Dilated mask
        """
        try:
            # Simple implementation using numpy
            # For production, would use cv2.dilate or scipy.ndimage
            from scipy import ndimage
            
            structure = ndimage.generate_binary_structure(2, 2)
            return ndimage.binary_dilation(mask, structure, iterations=pixels)
            
        except ImportError:
            # Fallback without scipy
            self.logger.warning("Scipy not available, using simple dilation")
            return mask
        except Exception as e:
            self.logger.error(f"Failed to dilate mask: {e}")
            return mask
    
    def _erode_mask(self, mask: np.ndarray, pixels: int) -> np.ndarray:
        """
        Erode (contract) mask by specified pixels
        
        Args:
            mask: Binary mask
            pixels: Number of pixels to contract
            
        Returns:
            Eroded mask
        """
        try:
            # Simple implementation using numpy
            # For production, would use cv2.erode or scipy.ndimage
            from scipy import ndimage
            
            structure = ndimage.generate_binary_structure(2, 2)
            return ndimage.binary_erosion(mask, structure, iterations=pixels)
            
        except ImportError:
            # Fallback without scipy
            self.logger.warning("Scipy not available, using simple erosion")
            return mask
        except Exception as e:
            self.logger.error(f"Failed to erode mask: {e}")
            return mask
    
    def _smooth_mask(self, mask: np.ndarray, iterations: int) -> np.ndarray:
        """
        Smooth mask using morphological operations
        
        Args:
            mask: Binary mask
            iterations: Number of smoothing iterations
            
        Returns:
            Smoothed mask
        """
        try:
            from scipy import ndimage
            
            # Apply opening and closing operations for smoothing
            structure = ndimage.generate_binary_structure(2, 1)
            
            smoothed = mask
            for _ in range(iterations):
                smoothed = ndimage.binary_opening(smoothed, structure)
                smoothed = ndimage.binary_closing(smoothed, structure)
            
            return smoothed
            
        except ImportError:
            self.logger.warning("Scipy not available, skipping smoothing")
            return mask
        except Exception as e:
            self.logger.error(f"Failed to smooth mask: {e}")
            return mask
    
    def _cleanup_failed_operation(self, operation: SelectionOperation):
        """
        Clean up resources after a failed operation
        
        Args:
            operation: Failed operation to clean up
        """
        try:
            self.logger.debug(f"[{operation.operation_id}] Starting cleanup for failed operation")
            
            # Clean up any partially created nodes
            for node in operation.created_nodes:
                try:
                    if node and node.parentNode():
                        self.logger.debug(f"[{operation.operation_id}] Removing created node: {node.name()}")
                        node.remove()
                except Exception as e:
                    self.logger.warning(f"[{operation.operation_id}] Failed to remove node during cleanup: {e}")
            
            # Reset document selection to original state if possible
            try:
                if operation.original_selection is not None:
                    selection = Selection()
                    selection.fromByteArray(operation.original_selection)
                    operation.document.setSelection(selection)
                    self.logger.debug(f"[{operation.operation_id}] Restored original selection")
                else:
                    operation.document.setSelection(None)
                    self.logger.debug(f"[{operation.operation_id}] Cleared selection")
            except Exception as e:
                self.logger.warning(f"[{operation.operation_id}] Failed to restore selection during cleanup: {e}")
            
            # Clear any temporary data
            operation.created_nodes.clear()
            operation.modified_nodes.clear()
            
            self.logger.debug(f"[{operation.operation_id}] Cleanup completed")
            
        except Exception as e:
            self.logger.error(f"[{operation.operation_id}] Error during cleanup: {e}")
    
    def _record_operation(self, operation: SelectionOperation):
        """
        Record operation for undo/redo
        
        Args:
            operation: Operation to record
        """
        try:
            # Remove any operations after current index (for redo invalidation)
            if self.current_operation_index < len(self.operation_history) - 1:
                self.operation_history = self.operation_history[:self.current_operation_index + 1]
            
            # Add new operation
            self.operation_history.append(operation)
            self.current_operation_index += 1
            
            # Maintain maximum history size
            if len(self.operation_history) > self.max_undo_levels:
                self.operation_history.pop(0)
                self.current_operation_index -= 1
            
            self.logger.debug(f"Recorded operation: {operation.operation_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to record operation: {e}")
    
    def undo_last_operation(self) -> bool:
        """
        Undo the last operation
        
        Returns:
            True if operation was undone
        """
        try:
            if self.current_operation_index < 0:
                self.logger.info("No operations to undo")
                return False
            
            operation = self.operation_history[self.current_operation_index]
            
            # Undo the operation
            success = self._undo_operation(operation)
            
            if success:
                self.current_operation_index -= 1
                self.logger.info(f"Undid operation: {operation.operation_id}")
                return True
            else:
                self.logger.error(f"Failed to undo operation: {operation.operation_id}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to undo operation: {e}")
            return False
    
    def redo_last_operation(self) -> bool:
        """
        Redo the last undone operation
        
        Returns:
            True if operation was redone
        """
        try:
            if self.current_operation_index >= len(self.operation_history) - 1:
                self.logger.info("No operations to redo")
                return False
            
            self.current_operation_index += 1
            operation = self.operation_history[self.current_operation_index]
            
            # Redo the operation
            success = self._redo_operation(operation)
            
            if success:
                self.logger.info(f"Redid operation: {operation.operation_id}")
                return True
            else:
                self.logger.error(f"Failed to redo operation: {operation.operation_id}")
                self.current_operation_index -= 1
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to redo operation: {e}")
            return False
    
    def _undo_operation(self, operation: SelectionOperation) -> bool:
        """
        Undo a specific operation
        
        Args:
            operation: Operation to undo
            
        Returns:
            True if successful
        """
        try:
            document = operation.document
            
            # Remove created nodes
            for node in operation.created_nodes:
                try:
                    if node.parentNode():
                        node.remove()
                except:
                    pass
            
            # Restore original selection
            if operation.original_selection is not None:
                selection = Selection()
                selection.fromByteArray(operation.original_selection)
                document.setSelection(selection)
            else:
                document.setSelection(None)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to undo operation: {e}")
            return False
    
    def _redo_operation(self, operation: SelectionOperation) -> bool:
        """
        Redo a specific operation
        
        Args:
            operation: Operation to redo
            
        Returns:
            True if successful
        """
        try:
            # Re-execute the operation
            if operation.operation_type == "mask_to_selection":
                return self._redo_mask_to_selection(operation)
            elif operation.operation_type == "extract_layer":
                return self._redo_extract_layer(operation)
            elif operation.operation_type == "create_mask_layer":
                return self._redo_create_mask_layer(operation)
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to redo operation: {e}")
            return False
    
    def _redo_mask_to_selection(self, operation: SelectionOperation) -> bool:
        """Redo mask to selection operation"""
        try:
            selection = self.mask_to_selection(
                operation.original_mask,
                operation.document,
                SelectionOperationType.REPLACE,
                operation.selection_options
            )
            return selection is not None
        except:
            return False
    
    def _redo_extract_layer(self, operation: SelectionOperation) -> bool:
        """Redo layer extraction operation"""
        # This would be more complex in practice
        return False
    
    def _redo_create_mask_layer(self, operation: SelectionOperation) -> bool:
        """Redo mask layer creation operation"""
        try:
            layer = self.create_mask_layer(
                operation.original_mask,
                operation.document,
                None,
                operation.selection_options
            )
            return layer is not None
        except:
            return False
    
    def can_undo(self) -> bool:
        """Check if undo is possible"""
        return self.current_operation_index >= 0
    
    def can_redo(self) -> bool:
        """Check if redo is possible"""
        return self.current_operation_index < len(self.operation_history) - 1
    
    def clear_history(self):
        """Clear operation history"""
        self.operation_history.clear()
        self.current_operation_index = -1
        self.logger.info("Operation history cleared")
    
    def get_operation_history(self) -> List[Dict[str, Any]]:
        """
        Get operation history summary
        
        Returns:
            List of operation summaries
        """
        return [
            {
                'operation_id': op.operation_id,
                'operation_type': op.operation_type,
                'timestamp': op.timestamp,
                'created_nodes_count': len(op.created_nodes),
                'modified_nodes_count': len(op.modified_nodes)
            }
            for op in self.operation_history
        ]
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get selection manager statistics
        
        Returns:
            Statistics dictionary
        """
        return {
            'total_operations': self.total_operations,
            'successful_operations': self.successful_operations,
            'success_rate': (self.successful_operations / max(1, self.total_operations)) * 100,
            'operation_history_size': len(self.operation_history),
            'current_operation_index': self.current_operation_index,
            'can_undo': self.can_undo(),
            'can_redo': self.can_redo()
        }
