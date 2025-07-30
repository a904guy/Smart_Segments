"""
Interactive Preview Widget
--------------------------

A clickable preview widget that displays the original image with overlaid segments
and allows users to click to select/deselect segments with modifier key support.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Set, Callable

try:
    from PyQt5.QtWidgets import QLabel, QWidget
    from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QPen, QBrush
    from PyQt5.QtCore import Qt, QPoint, QRect, pyqtSignal
except ImportError:
    # Handle cases where PyQt5 is not available
    QLabel = object
    QWidget = object
    QImage = object
    QPixmap = object
    QPainter = object
    QColor = object
    QPen = object
    QBrush = object
    Qt = None
    QPoint = object
    QRect = object
    pyqtSignal = None


class InteractivePreviewWidget(QLabel):
    """
    An interactive preview widget that displays the original image with clickable segments.
    """
    
    # Signal emitted when selection changes (selected_mask_indices: Set[int])
    selection_changed = pyqtSignal(set)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        
        # Data
        self._original_pixmap: Optional[QPixmap] = None
        self._masks: List[Dict[str, Any]] = []
        self._selected_indices: Set[int] = set()
        
        # Display properties
        self._scale_factor = 1.0
        self._display_rect = QRect()
        
        # Visual settings
        self._selection_color = QColor(255, 100, 100, 120)  # Semi-transparent red
        self._hover_color = QColor(100, 255, 100, 80)  # Semi-transparent green
        self._border_color = QColor(255, 255, 255, 200)  # Semi-transparent white
        self._border_width = 2
        
        # Interaction state
        self._hover_index = -1
        
        # Setup widget
        self.setMinimumSize(400, 300)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("border: 1px solid gray;")
        self.setMouseTracking(True)  # Enable hover tracking
        
        self.logger.info("InteractivePreviewWidget initialized")
    
    def set_image_and_masks(self, pixmap: QPixmap, masks: List[Dict[str, Any]]):
        """
        Set the original image and segmentation masks.
        
        Args:
            pixmap: The original image pixmap
            masks: List of mask dictionaries with 'segmentation' numpy arrays
        """
        try:
            self._original_pixmap = pixmap
            self._masks = masks
            self._selected_indices.clear()
            self._hover_index = -1
            
            if pixmap:
                self.logger.info(f"Set image: {pixmap.width()}x{pixmap.height()}, {len(masks)} masks")
                self._update_display()
            else:
                self.setText("No image available")
                self.logger.warning("No pixmap provided")
                
        except Exception as e:
            self.logger.error(f"Failed to set image and masks: {e}")
            self.setText("Failed to load image")
    
    def get_selected_indices(self) -> Set[int]:
        """Get the currently selected mask indices."""
        return self._selected_indices.copy()
    
    def set_selected_indices(self, indices: Set[int]):
        """Set the selected mask indices."""
        try:
            old_selection = self._selected_indices.copy()
            self._selected_indices = set(indices)
            
            if old_selection != self._selected_indices:
                self._update_display()
                self.selection_changed.emit(self._selected_indices)
                self.logger.info(f"Selection changed: {self._selected_indices}")
                
        except Exception as e:
            self.logger.error(f"Failed to set selected indices: {e}")
    
    def clear_selection(self):
        """Clear all selections."""
        if self._selected_indices:
            self._selected_indices.clear()
            self._update_display()
            self.selection_changed.emit(self._selected_indices)
            self.logger.info("Selection cleared")
    
    def _update_display(self):
        """Update the displayed image with overlays."""
        try:
            if not self._original_pixmap or not self._masks:
                self.setText("No image or masks available")
                return
            
            # Scale the pixmap to fit the widget while maintaining aspect ratio
            widget_size = self.size()
            scaled_pixmap = self._original_pixmap.scaled(
                widget_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            
            # Calculate scale factor and display rect
            self._scale_factor = min(
                widget_size.width() / self._original_pixmap.width(),
                widget_size.height() / self._original_pixmap.height()
            )
            
            # Calculate centered display rectangle
            display_width = int(self._original_pixmap.width() * self._scale_factor)
            display_height = int(self._original_pixmap.height() * self._scale_factor)
            x_offset = (widget_size.width() - display_width) // 2
            y_offset = (widget_size.height() - display_height) // 2
            self._display_rect = QRect(x_offset, y_offset, display_width, display_height)
            
            # Create a new pixmap for drawing overlays
            display_pixmap = QPixmap(widget_size)
            display_pixmap.fill(Qt.transparent)
            
            painter = QPainter(display_pixmap)
            
            # Draw the scaled original image
            painter.drawPixmap(self._display_rect, scaled_pixmap)
            
            # Draw mask overlays
            self._draw_mask_overlays(painter)
            
            painter.end()
            
            # Set the final pixmap
            self.setPixmap(display_pixmap)
            
        except Exception as e:
            self.logger.error(f"Failed to update display: {e}")
            self.setText("Display update failed")
    
    def _draw_mask_overlays(self, painter: QPainter):
        """Draw mask overlays on the painter."""
        try:
            if not self._masks:
                return
            
            for i, mask_data in enumerate(self._masks):
                mask = mask_data.get('segmentation')
                if mask is None:
                    continue
                
                # Determine overlay color
                if i == self._hover_index and i not in self._selected_indices:
                    color = self._hover_color
                elif i in self._selected_indices:
                    color = self._selection_color
                else:
                    continue  # Don't draw unselected, non-hovered masks
                
                # Draw the mask overlay
                self._draw_single_mask_overlay(painter, mask, color, i in self._selected_indices)
                
        except Exception as e:
            self.logger.error(f"Failed to draw mask overlays: {e}")
    
    def _draw_single_mask_overlay(self, painter: QPainter, mask: np.ndarray, color: QColor, draw_border: bool = False):
        """Draw a single mask overlay using efficient numpy operations."""
        try:
            mask_height, mask_width = mask.shape
            
            # Create RGBA array directly using numpy (much faster)
            rgba_array = np.zeros((mask_height, mask_width, 4), dtype=np.uint8)
            
            # Set color where mask is True using numpy broadcasting
            mask_bool = mask.astype(bool)
            rgba_array[mask_bool] = [color.red(), color.green(), color.blue(), color.alpha()]
            
            # Create QImage from numpy array
            bytes_per_line = mask_width * 4
            mask_image = QImage(
                rgba_array.data, mask_width, mask_height, bytes_per_line, QImage.Format_RGBA8888
            )
            
            # Scale the mask image to match the display
            scaled_mask = QPixmap.fromImage(mask_image).scaled(
                self._display_rect.size(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation
            )
            
            # Draw the scaled mask overlay
            painter.drawPixmap(self._display_rect, scaled_mask)
            
            # Draw border if selected
            if draw_border:
                painter.setPen(QPen(self._border_color, self._border_width))
                painter.setBrush(QBrush(Qt.transparent))
                
                # Find bounding box of the mask (cached if possible)
                rows, cols = np.where(mask)
                if len(rows) > 0 and len(cols) > 0:
                    min_row, max_row = rows.min(), rows.max()
                    min_col, max_col = cols.min(), cols.max()
                    
                    # Scale bounding box to display coordinates
                    x1 = int(self._display_rect.x() + min_col * self._scale_factor)
                    y1 = int(self._display_rect.y() + min_row * self._scale_factor)
                    x2 = int(self._display_rect.x() + max_col * self._scale_factor)
                    y2 = int(self._display_rect.y() + max_row * self._scale_factor)
                    
                    painter.drawRect(x1, y1, x2 - x1, y2 - y1)
                
        except Exception as e:
            self.logger.error(f"Failed to draw single mask overlay: {e}")
    
    def _widget_to_image_coordinates(self, widget_point: QPoint) -> Optional[QPoint]:
        """Convert widget coordinates to original image coordinates."""
        try:
            if not self._display_rect.contains(widget_point):
                return None
            
            # Convert to display-relative coordinates
            rel_x = widget_point.x() - self._display_rect.x()
            rel_y = widget_point.y() - self._display_rect.y()
            
            # Scale to original image coordinates
            img_x = int(rel_x / self._scale_factor)
            img_y = int(rel_y / self._scale_factor)
            
            # Clamp to image bounds
            if self._original_pixmap:
                img_x = max(0, min(img_x, self._original_pixmap.width() - 1))
                img_y = max(0, min(img_y, self._original_pixmap.height() - 1))
            
            return QPoint(img_x, img_y)
            
        except Exception as e:
            self.logger.error(f"Failed to convert widget to image coordinates: {e}")
            return None
    
    def _find_mask_at_point(self, image_point: QPoint) -> int:
        """Find the mask index at the given image coordinates."""
        try:
            x, y = image_point.x(), image_point.y()
            
            # Check masks in reverse order (top to bottom priority)
            for i in reversed(range(len(self._masks))):
                mask_data = self._masks[i]
                mask = mask_data.get('segmentation')
                if mask is None:
                    continue
                
                # Check if point is within mask bounds
                mask_height, mask_width = mask.shape
                if 0 <= x < mask_width and 0 <= y < mask_height:
                    if mask[y, x]:  # Point is inside this mask
                        return i
            
            return -1  # No mask found
            
        except Exception as e:
            self.logger.error(f"Failed to find mask at point: {e}")
            return -1
    
    def mousePressEvent(self, event):
        """Handle mouse press events for mask selection."""
        try:
            if event.button() != Qt.LeftButton:
                super().mousePressEvent(event)
                return
            
            # Convert to image coordinates
            image_point = self._widget_to_image_coordinates(event.pos())
            if image_point is None:
                super().mousePressEvent(event)
                return
            
            # Find mask at click point
            mask_index = self._find_mask_at_point(image_point)
            if mask_index == -1:
                super().mousePressEvent(event)
                return
            
            # Handle selection based on modifier keys
            modifiers = event.modifiers()
            old_selection = self._selected_indices.copy()
            
            if modifiers & Qt.ControlModifier:
                # Ctrl+click: toggle selection
                if mask_index in self._selected_indices:
                    self._selected_indices.remove(mask_index)
                    self.logger.info(f"Ctrl+click: Removed mask {mask_index} from selection")
                else:
                    self._selected_indices.add(mask_index)
                    self.logger.info(f"Ctrl+click: Added mask {mask_index} to selection")
            elif modifiers & Qt.ShiftModifier:
                # Shift+click: add to selection
                self._selected_indices.add(mask_index)
                self.logger.info(f"Shift+click: Added mask {mask_index} to selection")
            else:
                # Normal click: replace selection
                self._selected_indices = {mask_index}
                self.logger.info(f"Click: Selected mask {mask_index}")
            
            # Update display and emit signal if selection changed
            if old_selection != self._selected_indices:
                self._update_display()
                self.selection_changed.emit(self._selected_indices)
            
        except Exception as e:
            self.logger.error(f"Failed to handle mouse press: {e}")
            super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event):
        """Handle mouse move events for hover effects."""
        try:
            # Convert to image coordinates
            image_point = self._widget_to_image_coordinates(event.pos())
            if image_point is None:
                if self._hover_index != -1:
                    old_hover = self._hover_index
                    self._hover_index = -1
                    # Only update if we were actually hovering over something
                    if old_hover != -1:
                        self._update_display()
                super().mouseMoveEvent(event)
                return
            
            # Find mask at hover point
            mask_index = self._find_mask_at_point(image_point)
            
            # Only update if hover index actually changed
            if mask_index != self._hover_index:
                self._hover_index = mask_index
                self._update_display()
            
        except Exception as e:
            self.logger.error(f"Failed to handle mouse move: {e}")
        
        super().mouseMoveEvent(event)
    
    def leaveEvent(self, event):
        """Handle mouse leave events."""
        try:
            if self._hover_index != -1:
                self._hover_index = -1
                self._update_display()
        except Exception as e:
            self.logger.error(f"Failed to handle leave event: {e}")
        
        super().leaveEvent(event)
    
    def resizeEvent(self, event):
        """Handle resize events."""
        try:
            super().resizeEvent(event)
            if self._original_pixmap:
                self._update_display()
        except Exception as e:
            self.logger.error(f"Failed to handle resize event: {e}")
