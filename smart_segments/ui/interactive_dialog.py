"""
Interactive Segmentation Dialog
Main dialog that combines overlay widget and control panel for interactive segmentation
"""

import numpy as np
from typing import Optional, List, Dict, Any
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QSplitter, QWidget, 
    QMessageBox, QApplication
)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer, QRect
from PyQt5.QtGui import QKeyEvent

from .overlay_widget import SegmentationOverlay, InteractionMode
from .overlay_controls import OverlayControlPanel
from ..core.segmentation_api import SegmentationAPI, ClickType, SelectionMode
from krita import Document, Node


class InteractiveSegmentationDialog(QDialog):
    """
    Main dialog for interactive segmentation combining overlay and controls
    """
    
    # Signals
    segmentation_applied = pyqtSignal(np.ndarray, str)  # mask, target_layer_name
    dialog_closed = pyqtSignal()
    
    def __init__(self, extension, document: Document, layer: Node, parent=None):
        super().__init__(parent)
        
        # Core components
        self.extension = extension
        self.document = document
        self.layer = layer
        self.segmentation_api: Optional[SegmentationAPI] = None
        self.current_session_id: Optional[str] = None
        
        # UI state
        self.overlay_widget: Optional[SegmentationOverlay] = None
        self.control_panel: Optional[OverlayControlPanel] = None
        self.canvas_geometry: Optional[QRect] = None
        
        # Segmentation state
        self.current_preview_mask: Optional[np.ndarray] = None
        self.selected_masks: List[np.ndarray] = []
        self.current_confidence: float = 0.0
        self.interaction_active = False
        
        # Setup
        self.setWindowTitle("Smart Segments - Interactive Mode")
        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        self.resize(350, 600)
        
        self._setup_ui()
        self._initialize_segmentation()
        
    def _setup_ui(self):
        """Setup the dialog UI"""
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        
        # Create control panel
        self.control_panel = OverlayControlPanel()
        main_layout.addWidget(self.control_panel)
        
        # Connect control panel signals
        self._connect_control_signals()
        
        # Dialog styling
        self.setStyleSheet("""
            QDialog {
                background-color: #1e1e1e;
                color: white;
            }
        """)
        
    def _connect_control_signals(self):
        """Connect control panel signals to handlers"""
        if not self.control_panel:
            return
            
        self.control_panel.mode_change_requested.connect(self._handle_mode_change)
        self.control_panel.confirm_segmentation.connect(self._confirm_segmentation)
        self.control_panel.cancel_segmentation.connect(self._cancel_segmentation)
        self.control_panel.undo_last_action.connect(self._undo_last_action)
        self.control_panel.clear_all_selections.connect(self._clear_all_selections)
        self.control_panel.apply_to_layer.connect(self._apply_to_layer)
        
    def _initialize_segmentation(self):
        """Initialize segmentation session"""
        try:
            # Get segmentation API from extension
            self.segmentation_api = self.extension.get_segmentation_api()
            if not self.segmentation_api:
                self._show_error("Segmentation API not available. Please ensure the plugin is initialized.")
                return
                
            # Create session for current layer
            self.current_session_id = self.extension.create_session_for_document(
                self.document, self.layer
            )
            
            if not self.current_session_id:
                self._show_error("Failed to create segmentation session.")
                return
                
            # Setup overlay after small delay to ensure Krita canvas is ready
            QTimer.singleShot(500, self._setup_overlay)
            
        except Exception as e:
            self._show_error(f"Failed to initialize segmentation: {e}")
            
    def _setup_overlay(self):
        """Setup overlay widget on Krita canvas"""
        try:
            # Create overlay widget
            self.overlay_widget = SegmentationOverlay()
            
            # Connect overlay signals
            self.overlay_widget.click_detected.connect(self._handle_canvas_click)
            self.overlay_widget.mode_changed.connect(self._handle_overlay_mode_change)
            self.overlay_widget.confirmation_requested.connect(self._confirm_segmentation)
            self.overlay_widget.cancel_requested.connect(self._cancel_segmentation)
            self.overlay_widget.undo_requested.connect(self._undo_last_action)
            
            # Position overlay (this would need Krita canvas geometry in real implementation)
            self._position_overlay()
            
            # Show overlay
            self.overlay_widget.show()
            self.overlay_widget.setFocus()
            
            # Update status
            if self.control_panel:
                self.control_panel.set_status("Overlay active - Click on canvas to segment", "ready")
                
        except Exception as e:
            self._show_error(f"Failed to setup overlay: {e}")
            
    def _position_overlay(self):
        """Position overlay on Krita canvas"""
        # This is a simplified positioning - in real implementation,
        # you'd need to get actual Krita canvas geometry
        if self.overlay_widget:
            # For now, position relative to main window
            main_window = QApplication.activeWindow()
            if main_window:
                geometry = main_window.geometry()
                # Position on right side of main window
                overlay_rect = QRect(
                    geometry.right() - 800,
                    geometry.top() + 100,
                    600, 400
                )
                self.overlay_widget.setGeometry(overlay_rect)
                
                # Set canvas mapping (simplified)
                self.overlay_widget.set_canvas_mapping(
                    overlay_rect, 1.0, 0, 0
                )
                
    def _handle_canvas_click(self, x: int, y: int, is_positive: bool, is_multi_select: bool):
        """Handle click on canvas overlay"""
        try:
            if not self.segmentation_api or not self.current_session_id:
                return
                
            # Update UI state
            self.interaction_active = True
            if self.control_panel:
                self.control_panel.set_segmentation_active(True)
                self.control_panel.set_status("Processing segmentation...", "working")
                
            # Process click based on selection mode
            if is_multi_select:
                result = self.segmentation_api.segment_multi_click(
                    x, y, is_positive, 
                    replace_selection=False,
                    session_id=self.current_session_id
                )
            else:
                result = self.segmentation_api.segment_single_click(
                    x, y, is_positive,
                    session_id=self.current_session_id
                )
                
            if result:
                self._update_preview(result)
            else:
                if self.control_panel:
                    self.control_panel.set_status("Segmentation failed", "error")
                    
        except Exception as e:
            self._show_error(f"Segmentation error: {e}")
            
    def _update_preview(self, result):
        """Update preview with segmentation result"""
        try:
            # Store current result
            self.current_preview_mask = result.mask
            self.current_confidence = result.confidence
            
            # Update overlay
            if self.overlay_widget:
                self.overlay_widget.set_segment_preview(result.mask, result.confidence)
                
            # Update control panel
            if self.control_panel:
                self.control_panel.set_confidence(result.confidence)
                click_count = len(result.click_events)
                self.control_panel.set_status(
                    f"Segmentation complete ({click_count} clicks, {result.confidence:.1%} confidence)", 
                    "ready"
                )
                
        except Exception as e:
            self._show_error(f"Failed to update preview: {e}")
            
    def _handle_mode_change(self, mode_name: str):
        """Handle mode change from control panel"""
        try:
            mode = InteractionMode(mode_name)
            if self.overlay_widget:
                self.overlay_widget.set_interaction_mode(mode)
                
        except ValueError:
            pass  # Invalid mode name
            
    def _handle_overlay_mode_change(self, mode_name: str):
        """Handle mode change from overlay widget"""
        # Update control panel to match overlay
        pass
        
    def _confirm_segmentation(self):
        """Confirm current segmentation"""
        try:
            if self.current_preview_mask is None:
                self._show_error("No segmentation to confirm")
                return
                
            # Add to selected segments if in multi-segment mode
            if (self.overlay_widget and 
                self.overlay_widget.interaction_mode == InteractionMode.MULTI_SEGMENT):
                
                self.selected_masks.append(self.current_preview_mask.copy())
                if self.overlay_widget:
                    self.overlay_widget.add_selected_segment(self.current_preview_mask)
                    
                if self.control_panel:
                    self.control_panel.set_multi_segment_count(len(self.selected_masks))
                    self.control_panel.set_status("Segment added to selection", "ready")
                    
            else:
                # Single segment mode - apply immediately
                self._apply_segmentation([self.current_preview_mask])
                
        except Exception as e:
            self._show_error(f"Failed to confirm segmentation: {e}")
            
    def _cancel_segmentation(self):
        """Cancel current segmentation"""
        try:
            # Clear current preview
            self.current_preview_mask = None
            self.current_confidence = 0.0
            
            # Clear overlay
            if self.overlay_widget:
                self.overlay_widget.clear_all_visuals()
                
            # Reset control panel
            if self.control_panel:
                self.control_panel.reset_ui_state()
                
            # Clear session if exists
            if self.segmentation_api and self.current_session_id:
                self.segmentation_api.clear_session(self.current_session_id)
                
        except Exception as e:
            self._show_error(f"Failed to cancel segmentation: {e}")
            
    def _undo_last_action(self):
        """Undo last segmentation action"""
        try:
            if self.segmentation_api and self.current_session_id:
                if self.segmentation_api.undo_last_click(self.current_session_id):
                    # Get updated mask
                    updated_mask = self.segmentation_api.get_current_mask(self.current_session_id)
                    if updated_mask is not None:
                        self.current_preview_mask = updated_mask
                        if self.overlay_widget:
                            self.overlay_widget.set_segment_preview(updated_mask)
                    else:
                        # No mask after undo
                        self.current_preview_mask = None
                        if self.overlay_widget:
                            self.overlay_widget.set_segment_preview(None)
                            
                    if self.control_panel:
                        self.control_panel.set_status("Last action undone", "info")
                else:
                    if self.control_panel:
                        self.control_panel.set_status("Nothing to undo", "info")
                        
        except Exception as e:
            self._show_error(f"Failed to undo: {e}")
            
    def _clear_all_selections(self):
        """Clear all selections and start fresh"""
        try:
            # Clear all state
            self.selected_masks.clear()
            self.current_preview_mask = None
            self.current_confidence = 0.0
            
            # Clear overlay
            if self.overlay_widget:
                self.overlay_widget.clear_all_visuals()
                
            # Clear session
            if self.segmentation_api and self.current_session_id:
                self.segmentation_api.clear_session(self.current_session_id)
                
            # Reset control panel
            if self.control_panel:
                self.control_panel.reset_ui_state()
                self.control_panel.set_status("All selections cleared", "info")
                
        except Exception as e:
            self._show_error(f"Failed to clear selections: {e}")
            
    def _apply_to_layer(self):
        """Apply current segmentation to layer"""
        try:
            masks_to_apply = []
            
            # Collect masks to apply
            if self.selected_masks:
                masks_to_apply = self.selected_masks
            elif self.current_preview_mask is not None:
                masks_to_apply = [self.current_preview_mask]
            else:
                self._show_error("No segmentation to apply")
                return
                
            self._apply_segmentation(masks_to_apply)
            
        except Exception as e:
            self._show_error(f"Failed to apply to layer: {e}")
            
    def _apply_segmentation(self, masks: List[np.ndarray]):
        """Apply segmentation masks to Krita layer"""
        try:
            if not masks:
                return
                
            # Combine masks if multiple
            if len(masks) == 1:
                final_mask = masks[0]
            else:
                # Combine multiple masks (union)
                final_mask = masks[0].copy()
                for mask in masks[1:]:
                    final_mask = np.logical_or(final_mask, mask)
                    
            # Apply to layer using extension
            success = self.extension.apply_mask_to_layer(
                final_mask, self.layer, create_new_layer=True
            )
            
            if success:
                # Emit signal for external handling
                self.segmentation_applied.emit(final_mask, self.layer.name())
                
                # Show success message
                if self.control_panel:
                    self.control_panel.set_status("Segmentation applied successfully!", "ready")
                    
                # Close dialog after short delay
                QTimer.singleShot(2000, self.accept)
                
            else:
                self._show_error("Failed to apply segmentation to layer")
                
        except Exception as e:
            self._show_error(f"Failed to apply segmentation: {e}")
            
    def _show_error(self, message: str):
        """Show error message"""
        QMessageBox.critical(self, "Smart Segments Error", message)
        if self.control_panel:
            self.control_panel.set_status(f"Error: {message}", "error")
            
    def keyPressEvent(self, event: QKeyEvent):
        """Handle key press events"""
        # Forward to overlay if it exists
        if self.overlay_widget and self.overlay_widget.isVisible():
            self.overlay_widget.keyPressEvent(event)
        else:
            super().keyPressEvent(event)
            
    def keyReleaseEvent(self, event: QKeyEvent):
        """Handle key release events"""
        # Forward to overlay if it exists
        if self.overlay_widget and self.overlay_widget.isVisible():
            self.overlay_widget.keyReleaseEvent(event)
        else:
            super().keyReleaseEvent(event)
            
    def closeEvent(self, event):
        """Handle dialog close"""
        try:
            # Clean up overlay
            if self.overlay_widget:
                self.overlay_widget.hide()
                self.overlay_widget.deleteLater()
                self.overlay_widget = None
                
            # Close segmentation session
            if self.segmentation_api and self.current_session_id:
                self.segmentation_api.close_session(self.current_session_id)
                
            # Emit signal
            self.dialog_closed.emit()
            
        except Exception as e:
            print(f"Error during dialog cleanup: {e}")
            
        super().closeEvent(event)
        
    def showEvent(self, event):
        """Handle dialog show"""
        super().showEvent(event)
        
        # Ensure overlay is visible and focused
        if self.overlay_widget:
            self.overlay_widget.show()
            self.overlay_widget.raise_()
            self.overlay_widget.setFocus()
