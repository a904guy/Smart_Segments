"""
Smart Segments Dialog
--------------------

Main dialog for the Smart Segments plugin. Provides an interactive interface for
selecting and applying AI-generated image segments with clickable preview and
multiple selection modes.

Features:
- Interactive clickable preview with hover effects
- Multiple selection modes (click, Ctrl+click, Shift+click)
- Area-based sorting (largest segments first)
- Real-time synchronization between list and preview
- Multiple output options (selections, layers)
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Set

try:
    from PyQt5.QtCore import Qt, QSize
    from PyQt5.QtGui import QImage, QPixmap, QIcon, QPainter, QColor, QPen
    from PyQt5.QtWidgets import (
        QAbstractItemView,
        QCheckBox,
        QDialog,
        QDialogButtonBox,
        QHBoxLayout,
        QLabel,
        QListWidget,
        QListWidgetItem,
        QPushButton,
        QScrollArea,
        QSplitter,
        QVBoxLayout,
        QWidget,
    )
    from .interactive_preview import InteractivePreviewWidget
    
    PYQT5_AVAILABLE = True
except ImportError:
    # Graceful fallback for testing environments
    PYQT5_AVAILABLE = False
    
    # Mock objects for testing
    class MockQtClass:
        pass
    
    QDialog = MockQtClass
    QVBoxLayout = MockQtClass
    QPushButton = MockQtClass
    QListWidget = MockQtClass
    QListWidgetItem = MockQtClass
    QLabel = MockQtClass
    QDialogButtonBox = MockQtClass
    QAbstractItemView = MockQtClass
    QCheckBox = MockQtClass
    QImage = MockQtClass
    QPixmap = MockQtClass
    QIcon = MockQtClass
    InteractivePreviewWidget = MockQtClass
    Qt = None
    QSize = None


class SmartSegmentsDialog(QDialog):
    """
    Main dialog for Smart Segments plugin.
    
    Provides an interactive interface for selecting and applying AI-generated 
    image segments. Features include:
    - Interactive clickable preview with visual feedback
    - Multiple selection modes (normal, Ctrl+click, Shift+click)
    - Area-based sorting for better segment prioritization
    - Real-time synchronization between list and preview views
    - Flexible output options (selections, layers, or direct application)
    """
    
    # UI Constants
    MIN_WIDTH = 800
    MIN_HEIGHT = 600
    PREVIEW_SIZE = 100
    INFO_LABEL_HEIGHT = 40
    
    def __init__(
        self,
        extension,
        document,
        layer,
        masks: List[Dict[str, Any]],
        session_id: str,
        preview_size: int = PREVIEW_SIZE,
        enable_multi_selection: bool = True,
    ):
        """
        Initialize the Smart Segments dialog.

        Args:
            extension: SmartSegmentsExtension instance providing plugin functionality
            document: Krita Document instance to apply changes to
            layer: Krita Node (layer) instance containing the source image
            masks: List of segmentation mask dictionaries from AI processing
            session_id: Unique session identifier for cleanup purposes
            preview_size: Size in pixels for mask preview thumbnails
            enable_multi_selection: Whether to allow selecting multiple segments
        """
        if not PYQT5_AVAILABLE:
            raise RuntimeError("PyQt5 is required for the Smart Segments dialog")
            
        super().__init__(None)
        self.logger = logging.getLogger(__name__)
        
        # Core components
        self.extension = extension
        self.document = document
        self.layer = layer
        self.masks = masks
        self.session_id = session_id
        self.preview_size = preview_size
        
        # UI components (initialized in _setup_ui)
        self.list_widget = None
        self.interactive_preview = None
        self.create_selection_checkbox = None
        self.create_layer_checkbox = None
        self.button_box = None
        
        # Internal state
        self._original_pixmap = None
        
        # Initialize the dialog
        self._setup_ui()
        self._initialize_data()
        
    def _setup_ui(self):
        """Setup the user interface components."""
        self.setWindowTitle("Smart Segments")
        self.setMinimumSize(self.MIN_WIDTH, self.MIN_HEIGHT)
        
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(8)
        
        # Add components
        self._create_info_label(main_layout)
        self._create_main_content(main_layout)
        self._create_options_panel(main_layout)
        self._create_button_box(main_layout)
        
    def _create_info_label(self, parent_layout: QVBoxLayout):
        """Create and add the informational label."""
        info_text = (
            f"Found {len(self.masks)} segments. "
            "Click to select • Ctrl+click to toggle • Shift+click to add"
        )
        
        info_label = QLabel(info_text)
        info_label.setWordWrap(True)
        info_label.setMaximumHeight(self.INFO_LABEL_HEIGHT)
        info_label.setStyleSheet("QLabel { color: #666666; font-size: 11px; }")
        
        parent_layout.addWidget(info_label)
        
    def _create_main_content(self, parent_layout: QVBoxLayout):
        """Create the main content area with list and preview."""
        splitter = QSplitter(Qt.Horizontal)
        
        # Create list widget
        self._create_list_widget(splitter)
        
        # Create interactive preview
        self._create_interactive_preview(splitter)
        
        # Configure splitter
        splitter.setStretchFactor(0, 1)  # List widget
        splitter.setStretchFactor(1, 2)  # Preview widget (larger)
        
        parent_layout.addWidget(splitter)
        
    def _create_list_widget(self, parent_splitter: QSplitter):
        """Create and configure the segment list widget."""
        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.list_widget.itemSelectionChanged.connect(self._sync_list_to_preview)
        
        parent_splitter.addWidget(self.list_widget)
        
    def _create_interactive_preview(self, parent_splitter: QSplitter):
        """Create and configure the interactive preview widget."""
        self.interactive_preview = InteractivePreviewWidget()
        self.interactive_preview.setMinimumSize(400, 300)
        self.interactive_preview.selection_changed.connect(self._sync_preview_to_list)
        
        # Wrap in scroll area for better handling of large images
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.interactive_preview)
        scroll_area.setWidgetResizable(True)
        
        parent_splitter.addWidget(scroll_area)
        
    def _create_options_panel(self, parent_layout: QVBoxLayout):
        """Create the options panel with checkboxes."""
        options_layout = QHBoxLayout()
        
        self.create_selection_checkbox = QCheckBox("Create as selection")
        self.create_selection_checkbox.setChecked(True)
        self.create_selection_checkbox.setToolTip(
            "Create a selection from the chosen segments"
        )
        
        self.create_layer_checkbox = QCheckBox("Create as new layer")
        self.create_layer_checkbox.setToolTip(
            "Extract each segment into a separate layer"
        )
        
        options_layout.addWidget(self.create_selection_checkbox)
        options_layout.addWidget(self.create_layer_checkbox)
        options_layout.addStretch()  # Push options to the left
        
        parent_layout.addLayout(options_layout)
        
    def _create_button_box(self, parent_layout: QVBoxLayout):
        """Create the dialog button box."""
        self.button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        
        parent_layout.addWidget(self.button_box)
        
    def _initialize_data(self):
        """Initialize dialog data and populate UI components."""
        try:
            # Get layer pixmap for preview
            self._original_pixmap = self._get_layer_pixmap()
            
            # Sort masks by area (largest first)
            self._sort_masks_by_area()
            
            # Populate the list widget
            self._populate_list()
            
            # Initialize interactive preview
            if self._original_pixmap and self.interactive_preview:
                self.interactive_preview.set_image_and_masks(
                    self._original_pixmap, self.masks
                )
                
        except Exception as e:
            self.logger.error(f"Failed to initialize dialog data: {e}")
            self._show_error("Initialization Error", 
                           "Failed to initialize the dialog. Please try again.")
    
    def _show_error(self, title: str, message: str):
        """Show an error dialog to the user."""
        from PyQt5.QtWidgets import QMessageBox
        
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setWindowTitle(title)
        msg.setText(message)
        msg.exec_()

    def _get_layer_pixmap(self):
        """Get the pixmap of the current layer for preview."""
        try:
            # Initialize Krita bridge if not available
            if not self.extension._krita_bridge:
                from ..utils.krita_bridge import KritaImageBridge
                self.extension._krita_bridge = KritaImageBridge()
            
            img_array = self.extension._krita_bridge.layer_to_numpy(self.layer, convert_to_rgb=True)
            if img_array is None:
                self.logger.warning("Could not convert layer to numpy array")
                return None
            
            # Ensure the array is contiguous and in the right format
            img_array = img_array.copy()
            h, w = img_array.shape[:2]
            
            if len(img_array.shape) == 3:
                c = img_array.shape[2]
                if c == 3:  # RGB
                    # Convert to RGBA for consistent handling
                    rgba_array = np.zeros((h, w, 4), dtype=np.uint8)
                    rgba_array[:, :, :3] = img_array
                    rgba_array[:, :, 3] = 255  # Full alpha
                    img_array = rgba_array
                elif c == 4:  # Already RGBA
                    pass
                else:
                    self.logger.error(f"Unsupported channel count: {c}")
                    return None
            else:
                self.logger.error(f"Invalid array shape: {img_array.shape}")
                return None
            
            # Create QImage with proper memory layout
            bytes_per_line = w * 4
            qimage = QImage(img_array.data, w, h, bytes_per_line, QImage.Format_RGBA8888)
            
            # Convert to pixmap
            pixmap = QPixmap.fromImage(qimage)
            self.logger.info(f"Created pixmap: {pixmap.width()}x{pixmap.height()}")
            return pixmap
            
        except Exception as e:
            self.logger.error(f"Failed to get layer pixmap: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def _sort_masks_by_area(self):
        """Sort masks by area size (largest to smallest)."""
        try:
            if not self.masks:
                return
            
            self.logger.info(f"Sorting {len(self.masks)} masks by area size")
            
            # Sort masks by area in descending order (largest first)
            self.masks.sort(key=lambda mask_data: mask_data.get('area', 0), reverse=True)
            
            # Log the sorting results
            areas = [mask_data.get('area', 0) for mask_data in self.masks[:5]]  # Show first 5
            self.logger.info(f"Top 5 mask areas after sorting: {areas}")
            
        except Exception as e:
            self.logger.error(f"Failed to sort masks by area: {e}")

    def _update_preview(self):
        """Update the preview with selected masks overlaid."""
        try:
            if self._original_pixmap is None:
                self.preview_label.setText("Preview not available")
                return

            # Create a copy of the original pixmap to draw on
            preview_pixmap = self._original_pixmap.copy()
            painter = QPainter(preview_pixmap)

            # Overlay selected masks
            selected_items = self.list_widget.selectedItems()
            if not selected_items:
                self.preview_label.setPixmap(self._original_pixmap.scaled(
                    self.preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                ))
                painter.end()
                return

            for item in selected_items:
                mask_index = item.data(Qt.UserRole)
                if 0 <= mask_index < len(self.masks):
                    mask_data = self.masks[mask_index]
                    segmentation = mask_data.get("segmentation")
                    if segmentation is not None:
                        self._draw_mask_overlay(painter, segmentation)
            
            painter.end()
            self.preview_label.setPixmap(preview_pixmap.scaled(
                self.preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))

        except Exception as e:
            self.logger.error(f"Failed to update preview: {e}")

    def _draw_mask_overlay(self, painter: QPainter, mask: any):
        """Draw a single mask overlay with a semi-transparent color."""
        try:
            h, w = mask.shape
            color = QColor(255, 0, 0, 100)  # Semi-transparent red
            
            mask_image = QImage(w, h, QImage.Format_ARGB32)
            mask_image.fill(Qt.transparent)
            
            for y in range(h):
                for x in range(w):
                    if mask[y, x]:
                        mask_image.setPixel(x, y, color.rgba())
            
            painter.drawImage(0, 0, mask_image)

        except Exception as e:
            self.logger.error(f"Failed to draw mask overlay: {e}")

    def _populate_list(self):
        """
        Populate the list widget with mask previews.
        """
        if not self.masks:
            self.logger.warning("No masks to display.")
            return

        for i, mask_data in enumerate(self.masks):
            try:
                segmentation = mask_data.get("segmentation")
                if segmentation is None:
                    continue

                # Create a preview image from the mask
                h, w = segmentation.shape
                image = QImage(w, h, QImage.Format_ARGB32)
                image.fill(Qt.transparent)

                for y in range(h):
                    for x in range(w):
                        if segmentation[y, x]:
                            image.setPixel(x, y, Qt.white)

                pixmap = QPixmap.fromImage(image).scaled(
                    self.preview_size,
                    self.preview_size,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )

                item = QListWidgetItem(f"Mask {i+1} (Area: {mask_data.get('area', 'N/A')})")
                item.setIcon(QIcon(pixmap))
                item.setData(Qt.UserRole, i)  # Store mask index
                self.list_widget.addItem(item)

            except Exception as e:
                self.logger.error(f"Failed to create preview for mask {i}: {e}")
    
    def _sync_list_to_preview(self):
        """Synchronize list widget selection to interactive preview."""
        try:
            # Block signals to prevent infinite recursion
            self.interactive_preview.blockSignals(True)
            
            # Get selected indices from list widget
            selected_items = self.list_widget.selectedItems()
            selected_indices = set()
            for item in selected_items:
                mask_index = item.data(Qt.UserRole)
                if 0 <= mask_index < len(self.masks):
                    selected_indices.add(mask_index)
            
            # Update interactive preview selection
            self.interactive_preview.set_selected_indices(selected_indices)
            
            self.interactive_preview.blockSignals(False)
            self.logger.debug(f"Synced list to preview: {selected_indices}")
            
        except Exception as e:
            self.logger.error(f"Failed to sync list to preview: {e}")
            self.interactive_preview.blockSignals(False)
    
    def _sync_preview_to_list(self, selected_indices: set):
        """Synchronize interactive preview selection to list widget."""
        try:
            # Block signals to prevent infinite recursion
            self.list_widget.blockSignals(True)
            
            # Clear current selection
            self.list_widget.clearSelection()
            
            # Select corresponding items in list widget
            for i in range(self.list_widget.count()):
                item = self.list_widget.item(i)
                if item:
                    mask_index = item.data(Qt.UserRole)
                    if mask_index in selected_indices:
                        item.setSelected(True)
            
            self.list_widget.blockSignals(False)
            self.logger.debug(f"Synced preview to list: {selected_indices}")
            
        except Exception as e:
            self.logger.error(f"Failed to sync preview to list: {e}")
            self.list_widget.blockSignals(False)

    def get_selected_masks(self) -> Optional[List[Dict[str, Any]]]:
        """
        Get the selected masks from the list widget.

        Returns:
            List of selected mask dictionaries, or None if dialog was cancelled.
        """
        if self.result() == QDialog.Accepted:
            selected_items = self.list_widget.selectedItems()
            if not selected_items:
                return []

            selected_masks = []
            for item in selected_items:
                mask_index = item.data(Qt.UserRole)
                if 0 <= mask_index < len(self.masks):
                    selected_masks.append(self.masks[mask_index])

            return selected_masks

        return None

    def get_dialog_options(self) -> Dict[str, bool]:
        """
        Get the selected dialog options.

        Returns:
            Dictionary with the state of the dialog's checkboxes.
        """
        return {
            "create_selection": self.create_selection_checkbox.isChecked(),
            "create_layer": self.create_layer_checkbox.isChecked(),
        }

    def accept(self):
        """
        Override accept to apply selected masks before closing.
        """
        try:
            selected_items = self.list_widget.selectedItems()
            if not selected_items:
                # No selection, just close
                super().accept()
                return

            # Get selected masks
            selected_masks = []
            for item in selected_items:
                mask_index = item.data(Qt.UserRole)
                if 0 <= mask_index < len(self.masks):
                    selected_masks.append(self.masks[mask_index])

            if not selected_masks:
                super().accept()
                return

            # Get options and log them
            options = self.get_dialog_options()
            self.logger.info(f"Dialog accepted with options: {options}")
            
            # Apply masks
            self.logger.info("Calling _apply_selected_masks...")
            self._apply_selected_masks(selected_masks, options)
            self.logger.info("_apply_selected_masks call finished.")
            
            # Close session
            if self.session_id and self.extension._segmentation_api:
                self.extension._segmentation_api.close_session(self.session_id)
            
            super().accept()
            
        except Exception as e:
            self.logger.error(f"Error applying masks: {e}", exc_info=True)
            super().accept()

    def _apply_selected_masks(self, selected_masks: List[Dict[str, Any]], options: Dict[str, bool]):
        """
        Apply the selected masks to the document.
        
        Args:
            selected_masks: List of selected mask dictionaries
            options: Dialog options (create_selection, create_layer)
        """
        try:
            self.logger.info(f"_apply_selected_masks called with {len(selected_masks)} masks")
            self.logger.info(f"Options: {options}")
            
            if not self.extension._krita_bridge:
                self.logger.error("Krita bridge not available")
                return
            
            krita_bridge = self.extension._krita_bridge
            self.logger.info(f"Krita bridge available: {krita_bridge}")
            
            all_masks = []
            for i, mask_data in enumerate(selected_masks):
                self.logger.info(f"Processing mask {i+1}/{len(selected_masks)}")
                mask = mask_data.get('segmentation')
                if mask is None:
                    self.logger.warning(f"Mask {i+1} has no segmentation data")
                    continue
                all_masks.append(mask)

            if options.get('create_selection', False) and all_masks:
                # Combine all masks into one
                combined_mask = np.logical_or.reduce(all_masks)
                self.logger.info(f"Combined {len(all_masks)} masks into one")
                result = krita_bridge.create_selection_from_mask(combined_mask, self.document)
                if result:
                    self.logger.info("Successfully created selection from combined masks")
                else:
                    self.logger.error("Failed to create selection from combined masks")
            
            if options.get('create_layer', False):
                # Create new layers for each selected mask
                for i, mask_data in enumerate(selected_masks):
                    mask = mask_data.get('segmentation')
                    if mask is None:
                        continue
                    
                    try:
                        layer_name = f"Segment_{i+1}_area_{mask_data.get('area', 'unknown')}"
                        self.logger.info(f"Extracting layer content '{layer_name}' from mask {i+1}...")
                        result_layer = krita_bridge.extract_layer_content(
                            mask, self.layer, self.document, layer_name
                        )
                        if result_layer:
                            self.logger.info(f"Successfully extracted layer '{layer_name}' from mask {i+1}")
                        else:
                            self.logger.error(f"Failed to extract layer '{layer_name}' from mask {i+1}")
                    except Exception as e:
                        self.logger.error(f"Failed to create layer for mask {i+1}: {e}", exc_info=True)
            
            if not options.get('create_selection', False) and not options.get('create_layer', False):
                # Default: apply to current layer for each mask
                for i, mask_data in enumerate(selected_masks):
                    mask = mask_data.get('segmentation')
                    if mask is None:
                        continue
                    
                    try:
                        self.logger.info(f"Applying mask {i+1} to current layer (default behavior)...")
                        result_layer = krita_bridge.apply_mask_to_layer(mask, self.layer, self.document, create_new_layer=False)
                        if result_layer:
                            self.logger.info(f"Successfully applied mask {i+1} to current layer")
                        else:
                            self.logger.error(f"Failed to apply mask {i+1} to current layer")
                    except Exception as e:
                        self.logger.error(f"Failed to apply mask {i+1}: {e}", exc_info=True)
            
            # Refresh the document
            self.logger.info("Refreshing document projection...")
            self.document.refreshProjection()
            self.logger.info("Document refresh complete")
            
        except Exception as e:
            self.logger.error(f"Error in _apply_selected_masks: {e}", exc_info=True)
