"""
UI Controls for Interactive Overlay System
Provides mode switching and confirmation controls
"""

from typing import Optional, Callable
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QFrame, QButtonGroup, QToolButton, QProgressBar, QSlider
)
from PyQt5.QtGui import QIcon, QFont, QPalette, QColor
from PyQt5.QtCore import Qt, pyqtSignal, QTimer

from .overlay_widget import InteractionMode


class ModeButton(QPushButton):
    """Custom button for mode selection"""
    
    def __init__(self, mode: InteractionMode, text: str, icon_text: str = None):
        super().__init__(text)
        self.mode = mode
        self.setCheckable(True)
        self.setFixedSize(120, 40)
        
        # Styling
        self.setStyleSheet("""
            QPushButton {
                border: 2px solid #555;
                border-radius: 8px;
                background-color: #333;
                color: white;
                font-weight: bold;
                padding: 8px;
            }
            QPushButton:checked {
                background-color: #0078d4;
                border-color: #106ebe;
            }
            QPushButton:hover {
                background-color: #444;
            }
            QPushButton:checked:hover {
                background-color: #106ebe;
            }
        """)


class ConfidenceIndicator(QWidget):
    """Visual indicator for segmentation confidence"""
    
    def __init__(self):
        super().__init__()
        self.confidence = 0.0
        self.setFixedSize(200, 20)
        
    def set_confidence(self, confidence: float):
        """Set confidence level (0.0 to 1.0)"""
        self.confidence = max(0.0, min(1.0, confidence))
        self.update()
        
    def paintEvent(self, event):
        """Draw confidence bar"""
        from PyQt5.QtGui import QPainter, QBrush, QPen
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Background
        painter.setBrush(QBrush(QColor(60, 60, 60)))
        painter.setPen(QPen(QColor(100, 100, 100)))
        painter.drawRoundedRect(self.rect(), 4, 4)
        
        # Confidence bar
        if self.confidence > 0:
            bar_width = int(self.rect().width() * self.confidence)
            
            # Color based on confidence level
            if self.confidence >= 0.8:
                color = QColor(0, 200, 0)  # Green
            elif self.confidence >= 0.5:
                color = QColor(255, 165, 0)  # Orange
            else:
                color = QColor(255, 100, 100)  # Red
                
            painter.setBrush(QBrush(color))
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(0, 0, bar_width, self.rect().height(), 4, 4)
        
        # Text
        painter.setPen(QPen(Qt.white))
        painter.drawText(self.rect(), Qt.AlignCenter, f"Confidence: {self.confidence:.1%}")


class OverlayControlPanel(QWidget):
    """
    Control panel for interactive overlay system
    Provides mode switching, confirmation, and status display
    """
    
    # Signals
    mode_change_requested = pyqtSignal(str)  # mode name
    confirm_segmentation = pyqtSignal()
    cancel_segmentation = pyqtSignal()
    undo_last_action = pyqtSignal()
    clear_all_selections = pyqtSignal()
    apply_to_layer = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_mode = InteractionMode.SINGLE_SEGMENT
        self.is_segmentation_active = False
        
        self._setup_ui()
        self._setup_styling()
        
    def _setup_ui(self):
        """Setup the UI layout and components"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Title
        title = QLabel("Smart Segments Interactive Mode")
        title.setFont(QFont("Arial", 12, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Mode selection buttons
        mode_frame = QFrame()
        mode_frame.setFrameStyle(QFrame.Box)
        mode_layout = QVBoxLayout(mode_frame)
        
        mode_label = QLabel("Selection Mode:")
        mode_label.setFont(QFont("Arial", 10, QFont.Bold))
        mode_layout.addWidget(mode_label)
        
        # Mode buttons
        button_layout = QHBoxLayout()
        
        self.single_mode_btn = ModeButton(
            InteractionMode.SINGLE_SEGMENT, 
            "Single\nSegment", 
            "●"
        )
        self.multi_mode_btn = ModeButton(
            InteractionMode.MULTI_SEGMENT, 
            "Multi\nSegment", 
            "●●●"
        )
        
        # Button group for exclusive selection
        self.mode_button_group = QButtonGroup()
        self.mode_button_group.addButton(self.single_mode_btn)
        self.mode_button_group.addButton(self.multi_mode_btn)
        self.single_mode_btn.setChecked(True)
        
        button_layout.addWidget(self.single_mode_btn)
        button_layout.addWidget(self.multi_mode_btn)
        button_layout.addStretch()
        
        mode_layout.addLayout(button_layout)
        layout.addWidget(mode_frame)
        
        # Confidence indicator
        confidence_frame = QFrame()
        confidence_frame.setFrameStyle(QFrame.Box)
        confidence_layout = QVBoxLayout(confidence_frame)
        
        confidence_label = QLabel("Segmentation Confidence:")
        confidence_label.setFont(QFont("Arial", 10, QFont.Bold))
        confidence_layout.addWidget(confidence_label)
        
        self.confidence_indicator = ConfidenceIndicator()
        confidence_layout.addWidget(self.confidence_indicator)
        
        layout.addWidget(confidence_frame)
        
        # Action buttons
        action_frame = QFrame()
        action_frame.setFrameStyle(QFrame.Box)
        action_layout = QVBoxLayout(action_frame)
        
        action_label = QLabel("Actions:")
        action_label.setFont(QFont("Arial", 10, QFont.Bold))
        action_layout.addWidget(action_label)
        
        # Primary actions
        primary_layout = QHBoxLayout()
        
        self.confirm_btn = QPushButton("✓ Confirm")
        self.confirm_btn.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #218838;
            }
            QPushButton:disabled {
                background-color: #6c757d;
            }
        """)
        self.confirm_btn.setEnabled(False)
        
        self.cancel_btn = QPushButton("✗ Cancel")
        self.cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: #dc3545;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #c82333;
            }
        """)
        
        primary_layout.addWidget(self.confirm_btn)
        primary_layout.addWidget(self.cancel_btn)
        action_layout.addLayout(primary_layout)
        
        # Secondary actions
        secondary_layout = QHBoxLayout()
        
        self.undo_btn = QPushButton("↶ Undo")
        self.undo_btn.setToolTip("Undo last click (Ctrl+Z)")
        
        self.clear_btn = QPushButton("🗑 Clear")
        self.clear_btn.setToolTip("Clear all selections")
        
        self.apply_btn = QPushButton("📄 Apply to Layer")
        self.apply_btn.setStyleSheet("""
            QPushButton {
                background-color: #007bff;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
            QPushButton:disabled {
                background-color: #6c757d;
            }
        """)
        self.apply_btn.setEnabled(False)
        
        secondary_layout.addWidget(self.undo_btn)
        secondary_layout.addWidget(self.clear_btn)
        action_layout.addLayout(secondary_layout)
        action_layout.addWidget(self.apply_btn)
        
        layout.addWidget(action_frame)
        
        # Status display
        status_frame = QFrame()
        status_frame.setFrameStyle(QFrame.Box)
        status_layout = QVBoxLayout(status_frame)
        
        status_label = QLabel("Status:")
        status_label.setFont(QFont("Arial", 10, QFont.Bold))
        status_layout.addWidget(status_label)
        
        self.status_label = QLabel("Ready - Click on image to segment")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #28a745; font-weight: bold;")
        status_layout.addWidget(self.status_label)
        
        # Click instructions
        instructions = QLabel(
            "Instructions:\\n"
            "• Left click: Add positive point\\n"
            "• Ctrl+click: Add negative point\\n"
            "• Shift+click: Multi-segment mode\\n"
            "• Enter: Confirm selection\\n"
            "• Esc: Cancel\\n"
            "• Ctrl+Z: Undo last click"
        )
        instructions.setFont(QFont("Arial", 8))
        instructions.setStyleSheet("color: #666; background-color: #f8f9fa; padding: 5px; border-radius: 3px;")
        status_layout.addWidget(instructions)
        
        layout.addWidget(status_frame)
        
        layout.addStretch()
        
        # Connect signals
        self._connect_signals()
        
    def _setup_styling(self):
        """Setup widget styling"""
        self.setStyleSheet("""
            QFrame {
                background-color: #2b2b2b;
                border: 1px solid #555;
                border-radius: 8px;
                margin: 2px;
                padding: 5px;
            }
            QLabel {
                color: white;
                background: transparent;
                border: none;
            }
            QPushButton {
                background-color: #444;
                color: white;
                border: 1px solid #666;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #555;
            }
            QPushButton:pressed {
                background-color: #333;
            }
            QPushButton:disabled {
                background-color: #2a2a2a;
                color: #666;
            }
        """)
        
    def _connect_signals(self):
        """Connect internal signals"""
        # Mode buttons
        self.single_mode_btn.clicked.connect(
            lambda: self._change_mode(InteractionMode.SINGLE_SEGMENT)
        )
        self.multi_mode_btn.clicked.connect(
            lambda: self._change_mode(InteractionMode.MULTI_SEGMENT)
        )
        
        # Action buttons
        self.confirm_btn.clicked.connect(self.confirm_segmentation.emit)
        self.cancel_btn.clicked.connect(self.cancel_segmentation.emit)
        self.undo_btn.clicked.connect(self.undo_last_action.emit)
        self.clear_btn.clicked.connect(self.clear_all_selections.emit)
        self.apply_btn.clicked.connect(self.apply_to_layer.emit)
        
    def _change_mode(self, mode: InteractionMode):
        """Handle mode change"""
        self.current_mode = mode
        self.mode_change_requested.emit(mode.value)
        self._update_status_for_mode(mode)
        
    def _update_status_for_mode(self, mode: InteractionMode):
        """Update status text based on mode"""
        if mode == InteractionMode.SINGLE_SEGMENT:
            self.set_status("Single segment mode - Click to select one segment", "ready")
        elif mode == InteractionMode.MULTI_SEGMENT:
            self.set_status("Multi-segment mode - Shift+click to add segments", "ready")
        elif mode == InteractionMode.CONFIRMATION:
            self.set_status("Confirm your selection or cancel", "confirm")
            
    def set_confidence(self, confidence: float):
        """Update confidence indicator"""
        self.confidence_indicator.set_confidence(confidence)
        
    def set_status(self, message: str, status_type: str = "info"):
        """Set status message with color coding"""
        self.status_label.setText(message)
        
        color_map = {
            "ready": "#28a745",      # Green
            "working": "#ffc107",    # Yellow
            "error": "#dc3545",      # Red
            "confirm": "#17a2b8",    # Cyan
            "info": "#6c757d"        # Gray
        }
        
        color = color_map.get(status_type, "#6c757d")
        self.status_label.setStyleSheet(f"color: {color}; font-weight: bold;")
        
    def set_segmentation_active(self, active: bool):
        """Update UI state based on segmentation activity"""
        self.is_segmentation_active = active
        
        # Enable/disable buttons based on state
        self.confirm_btn.setEnabled(active)
        self.apply_btn.setEnabled(active)
        self.undo_btn.setEnabled(active)
        
        # Update mode buttons
        self.single_mode_btn.setEnabled(not active or self.current_mode != InteractionMode.CONFIRMATION)
        self.multi_mode_btn.setEnabled(not active or self.current_mode != InteractionMode.CONFIRMATION)
        
        if active:
            self.set_status("Segmentation active - refine your selection", "working")
        else:
            self.set_status("Ready - Click on image to segment", "ready")
            
    def set_multi_segment_count(self, count: int):
        """Update display for multi-segment selection"""
        if count > 0:
            self.set_status(f"Multi-segment: {count} segments selected", "working")
            
    def highlight_confirmation_mode(self, highlight: bool):
        """Highlight UI when in confirmation mode"""
        if highlight:
            self.confirm_btn.setStyleSheet("""
                QPushButton {
                    background-color: #28a745;
                    color: white;
                    border: 3px solid #ffc107;
                    padding: 10px;
                    border-radius: 5px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #218838;
                }
            """)
        else:
            # Reset to normal styling
            self.confirm_btn.setStyleSheet("""
                QPushButton {
                    background-color: #28a745;
                    color: white;
                    border: none;
                    padding: 10px;
                    border-radius: 5px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #218838;
                }
                QPushButton:disabled {
                    background-color: #6c757d;
                }
            """)
            
    def reset_ui_state(self):
        """Reset UI to initial state"""
        self.single_mode_btn.setChecked(True)
        self.multi_mode_btn.setChecked(False)
        self.current_mode = InteractionMode.SINGLE_SEGMENT
        self.set_segmentation_active(False)
        self.set_confidence(0.0)
        self.highlight_confirmation_mode(False)
        self.set_status("Ready - Click on image to segment", "ready")
