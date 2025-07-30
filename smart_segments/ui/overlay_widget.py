"""
Interactive Overlay Widget for Smart Segments
Provides canvas-based segmentation interaction with visual feedback
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum
from PyQt5.QtWidgets import (
    QWidget, QApplication, QVBoxLayout, QHBoxLayout, QPushButton, 
    QLabel, QFrame, QButtonGroup
)
from PyQt5.QtGui import (
    QPainter, QColor, QMouseEvent, QKeyEvent, QPen, QBrush, 
    QPixmap, QImage, QFont, QPainterPath
)
from PyQt5.QtCore import Qt, QRectF, pyqtSignal, QTimer, QPointF


class InteractionMode(Enum):
    """Modes for overlay interaction"""
    SINGLE_SEGMENT = "single"
    MULTI_SEGMENT = "multi"
    PREVIEW_ONLY = "preview"
    CONFIRMATION = "confirm"


class ClickIndicator:
    """Visual indicator for click points"""
    def __init__(self, x: int, y: int, is_positive: bool, timestamp: float):
        self.x = x
        self.y = y
        self.is_positive = is_positive  # True for positive, False for negative
        self.timestamp = timestamp
        self.fade_duration = 3.0  # seconds
        
    def get_opacity(self, current_time: float) -> float:
        """Calculate opacity based on age"""
        age = current_time - self.timestamp
        if age >= self.fade_duration:
            return 0.0
        return max(0.0, 1.0 - (age / self.fade_duration))
        
    def is_expired(self, current_time: float) -> bool:
        """Check if indicator should be removed"""
        return (current_time - self.timestamp) > self.fade_duration


class SegmentationOverlay(QWidget):
    """
    Custom QWidget overlay that renders on top of the canvas for interactive segmentation
    """
    
    # Signals
    click_detected = pyqtSignal(int, int, bool, bool)  # x, y, is_positive, is_multi_select
    mode_changed = pyqtSignal(str)  # mode name
    confirmation_requested = pyqtSignal()
    cancel_requested = pyqtSignal()
    undo_requested = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Widget setup for overlay
        self.setWindowFlags(
            Qt.Window | Qt.FramelessWindowHint | 
            Qt.WindowStaysOnTopHint | Qt.X11BypassWindowManagerHint
        )
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self.setFocusPolicy(Qt.StrongFocus)
        
        # Interaction state
        self.interaction_mode = InteractionMode.SINGLE_SEGMENT
        self.is_shift_pressed = False
        self.is_ctrl_pressed = False
        
        # Visual elements
        self.segment_preview: Optional[np.ndarray] = None
        self.confidence_level: float = 0.0
        self.click_indicators: List[ClickIndicator] = []
        self.selected_segments: List[np.ndarray] = []
        
        # Canvas mapping
        self.canvas_rect: Optional[QRectF] = None
        self.scale_factor: float = 1.0
        self.offset_x: int = 0
        self.offset_y: int = 0
        
        # Animation and updates
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_indicators)
        self.update_timer.start(100)  # Update every 100ms
        
        # Styling
        self.positive_click_color = QColor(0, 255, 0, 180)  # Green
        self.negative_click_color = QColor(255, 0, 0, 180)  # Red
        self.preview_color = QColor(0, 150, 255, 100)  # Blue
        self.selected_color = QColor(255, 165, 0, 120)  # Orange
        
    def set_canvas_mapping(self, canvas_rect: QRectF, scale_factor: float, 
                          offset_x: int, offset_y: int):
        """Set coordinate mapping between overlay and Krita canvas"""
        self.canvas_rect = canvas_rect
        self.scale_factor = scale_factor
        self.offset_x = offset_x
        self.offset_y = offset_y
        
    def map_to_canvas_coordinates(self, x: int, y: int) -> Tuple[int, int]:
        """Map overlay coordinates to canvas coordinates"""
        if self.canvas_rect is None:
            return x, y
            
        # Apply inverse transformation
        canvas_x = int((x - self.offset_x) / self.scale_factor)
        canvas_y = int((y - self.offset_y) / self.scale_factor)
        
        return canvas_x, canvas_y
        
    def map_from_canvas_coordinates(self, canvas_x: int, canvas_y: int) -> Tuple[int, int]:
        """Map canvas coordinates to overlay coordinates"""
        if self.canvas_rect is None:
            return canvas_x, canvas_y
            
        # Apply transformation
        x = int(canvas_x * self.scale_factor + self.offset_x)
        y = int(canvas_y * self.scale_factor + self.offset_y)
        
        return x, y
        
    def set_interaction_mode(self, mode: InteractionMode):
        """Set the current interaction mode"""
        self.interaction_mode = mode
        self.mode_changed.emit(mode.value)
        self.update()
        
    def set_segment_preview(self, mask: np.ndarray, confidence: float = 0.0):
        """Set the current segment preview with confidence level"""
        self.segment_preview = mask
        self.confidence_level = confidence
        self.update()
        
    def add_selected_segment(self, mask: np.ndarray):
        """Add a segment to the multi-selection"""
        self.selected_segments.append(mask.copy())
        self.update()
        
    def clear_selected_segments(self):
        """Clear all selected segments"""
        self.selected_segments.clear()
        self.update()
        
    def add_click_indicator(self, x: int, y: int, is_positive: bool):
        """Add visual click indicator"""
        import time
        indicator = ClickIndicator(x, y, is_positive, time.time())
        self.click_indicators.append(indicator)
        self.update()
        
    def _update_indicators(self):
        """Update and remove expired click indicators"""
        import time
        current_time = time.time()
        
        # Remove expired indicators
        self.click_indicators = [
            indicator for indicator in self.click_indicators 
            if not indicator.is_expired(current_time)
        ]
        
        # Trigger repaint if there are active indicators
        if self.click_indicators:
            self.update()
            
    def paintEvent(self, event):
        """Render the overlay with segment previews and visual feedback"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        
        # Clear background
        painter.fillRect(self.rect(), QColor(0, 0, 0, 0))
        
        # Draw selected segments (multi-selection)
        for segment_mask in self.selected_segments:
            self._draw_segment_mask(painter, segment_mask, self.selected_color)
            
        # Draw current preview segment
        if self.segment_preview is not None:
            # Adjust opacity based on confidence
            preview_color = QColor(self.preview_color)
            preview_color.setAlpha(int(100 + (self.confidence_level * 100)))
            self._draw_segment_mask(painter, self.segment_preview, preview_color)
            
        # Draw click indicators
        self._draw_click_indicators(painter)
        
        # Draw mode-specific overlays
        self._draw_mode_overlay(painter)
        
    def _draw_segment_mask(self, painter: QPainter, mask: np.ndarray, color: QColor):
        """Draw a segment mask with the specified color"""
        if mask is None or mask.size == 0:
            return
            
        # Create brush for filling
        brush = QBrush(color)
        painter.setBrush(brush)
        painter.setPen(Qt.NoPen)
        
        # Convert mask to path for smoother rendering
        path = self._mask_to_path(mask)
        if path:
            painter.drawPath(path)
            
    def _mask_to_path(self, mask: np.ndarray) -> QPainterPath:
        """Convert binary mask to QPainterPath for smooth rendering"""
        path = QPainterPath()
        
        if mask is None or mask.size == 0:
            return path
            
        # Simple rectangle approximation for now
        # In a full implementation, you'd want contour detection
        h, w = mask.shape[:2]
        
        # Find bounding box of mask
        coords = np.where(mask > 0)
        if len(coords[0]) > 0:
            y_min, y_max = coords[0].min(), coords[0].max()
            x_min, x_max = coords[1].min(), coords[1].max()
            
            # Map to overlay coordinates
            x1, y1 = self.map_from_canvas_coordinates(x_min, y_min)
            x2, y2 = self.map_from_canvas_coordinates(x_max, y_max)
            
            path.addRect(x1, y1, x2 - x1, y2 - y1)
            
        return path
        
    def _draw_click_indicators(self, painter: QPainter):
        """Draw click indicators with fade animation"""
        import time
        current_time = time.time()
        
        for indicator in self.click_indicators:
            opacity = indicator.get_opacity(current_time)
            if opacity <= 0:
                continue
                
            # Choose color based on click type
            color = self.positive_click_color if indicator.is_positive else self.negative_click_color
            color.setAlpha(int(opacity * 180))
            
            # Draw circle at click point
            painter.setBrush(QBrush(color))
            painter.setPen(QPen(Qt.white, 2))
            
            radius = 8 + (1.0 - opacity) * 4  # Expand as it fades
            painter.drawEllipse(
                QPointF(indicator.x, indicator.y), 
                radius, radius
            )
            
    def _draw_mode_overlay(self, painter: QPainter):
        """Draw mode-specific visual overlays"""
        if self.interaction_mode == InteractionMode.MULTI_SEGMENT:
            # Multi-selection indicator
            painter.setPen(QPen(QColor(255, 255, 0), 3, Qt.DashLine))
            painter.drawRect(self.rect().adjusted(5, 5, -5, -5))
            
        elif self.interaction_mode == InteractionMode.CONFIRMATION:
            # Confirmation overlay
            painter.fillRect(self.rect(), QColor(0, 0, 0, 50))
            
            # Draw confirmation text
            painter.setPen(QPen(Qt.white))
            font = QFont()
            font.setPointSize(16)
            font.setBold(True)
            painter.setFont(font)
            
            text = "Press Enter to confirm, Esc to cancel"
            text_rect = painter.fontMetrics().boundingRect(text)
            center = self.rect().center()
            text_pos = center - QPointF(text_rect.width() / 2, text_rect.height() / 2)
            painter.drawText(text_pos, text)
            
    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press events for click detection"""
        if event.button() != Qt.LeftButton:
            return
            
        # Map to canvas coordinates
        canvas_x, canvas_y = self.map_to_canvas_coordinates(event.x(), event.y())
        
        # Determine click type
        is_positive = not (event.modifiers() & Qt.ControlModifier)  # Ctrl = negative click
        is_multi_select = bool(event.modifiers() & Qt.ShiftModifier)  # Shift = multi-select
        
        # Add visual indicator
        self.add_click_indicator(event.x(), event.y(), is_positive)
        
        # Emit click signal
        self.click_detected.emit(canvas_x, canvas_y, is_positive, is_multi_select)
        
    def keyPressEvent(self, event: QKeyEvent):
        """Handle key press events for mode switching and actions"""
        key = event.key()
        
        if key == Qt.Key_Shift:
            self.is_shift_pressed = True
            if self.interaction_mode == InteractionMode.SINGLE_SEGMENT:
                self.set_interaction_mode(InteractionMode.MULTI_SEGMENT)
                
        elif key == Qt.Key_Control:
            self.is_ctrl_pressed = True
            
        elif key == Qt.Key_Return or key == Qt.Key_Enter:
            if self.interaction_mode == InteractionMode.CONFIRMATION:
                self.confirmation_requested.emit()
            else:
                self.set_interaction_mode(InteractionMode.CONFIRMATION)
                
        elif key == Qt.Key_Escape:
            if self.interaction_mode == InteractionMode.CONFIRMATION:
                self.set_interaction_mode(InteractionMode.SINGLE_SEGMENT)
            self.cancel_requested.emit()
            
        elif key == Qt.Key_Z and (event.modifiers() & Qt.ControlModifier):
            self.undo_requested.emit()
            
        self.update()
        
    def keyReleaseEvent(self, event: QKeyEvent):
        """Handle key release events"""
        key = event.key()
        
        if key == Qt.Key_Shift:
            self.is_shift_pressed = False
            if self.interaction_mode == InteractionMode.MULTI_SEGMENT:
                self.set_interaction_mode(InteractionMode.SINGLE_SEGMENT)
                
        elif key == Qt.Key_Control:
            self.is_ctrl_pressed = False
            
        self.update()
        
    def showEvent(self, event):
        """Handle show event"""
        super().showEvent(event)
        self.setFocus()  # Ensure we receive key events
        
    def clear_all_visuals(self):
        """Clear all visual elements"""
        self.segment_preview = None
        self.confidence_level = 0.0
        self.click_indicators.clear()
        self.selected_segments.clear()
        self.update()

