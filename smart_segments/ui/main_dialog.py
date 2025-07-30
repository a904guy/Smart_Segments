"""
Main Smart Segments Dialog
Primary interface for the Smart Segments plugin with interactive overlay support
"""

from typing import Optional
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QTabWidget, QWidget, QMessageBox, QTextEdit
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont

from .interactive_dialog import InteractiveSegmentationDialog
from krita import Krita


class SmartSegmentsDialog(QDialog):
    """
    Main dialog for Smart Segments plugin
    Provides access to different segmentation modes including interactive overlay
    """
    
    def __init__(self, extension, parent=None):
        super().__init__(parent)
        
        self.extension = extension
        self.interactive_dialog: Optional[InteractiveSegmentationDialog] = None
        
        self.setWindowTitle("Smart Segments")
        self.setModal(False)
        self.resize(400, 500)
        
        self._setup_ui()
        
    def _setup_ui(self):
        """Setup the main dialog UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Title
        title = QLabel("Smart Segments - AI-Powered Segmentation")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #0078d4; margin-bottom: 10px;")
        layout.addWidget(title)
        
        # Tab widget for different modes
        tab_widget = QTabWidget()
        
        # Setup Wizard Tab
        setup_tab = self._create_setup_wizard_tab()
        tab_widget.addTab(setup_tab, "Setup Wizard")
        
        # Interactive Mode Tab
        interactive_tab = self._create_interactive_tab()
        tab_widget.addTab(interactive_tab, "Interactive Mode")
        
        # Quick Segment Tab (placeholder)
        quick_tab = self._create_quick_tab()
        tab_widget.addTab(quick_tab, "Quick Segment")
        
        # Batch Processing Tab (placeholder)
        batch_tab = self._create_batch_tab()
        tab_widget.addTab(batch_tab, "Batch Processing")
        
        layout.addWidget(tab_widget)
        
        # Status and info
        status_label = QLabel("Ready - Select a mode above to begin segmentation")
        status_label.setStyleSheet("color: #666; padding: 10px; background-color: #f0f0f0; border-radius: 5px;")
        layout.addWidget(status_label)
        
        # Dialog styling
        self.setStyleSheet("""
            QDialog {
                background-color: white;
            }
            QTabWidget::pane {
                border: 1px solid #c0c0c0;
                background-color: white;
            }
            QTabBar::tab {
                background-color: #e0e0e0;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: white;
                border-bottom: 2px solid #0078d4;
            }
        """)
        
    def _create_interactive_tab(self) -> QWidget:
        """Create the interactive mode tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(15)
        
        # Description
        desc = QLabel(
            "Interactive mode provides real-time segmentation with visual feedback. "
            "Click directly on the canvas to segment objects with immediate preview."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #333; padding: 10px; background-color: #f8f9fa; border-radius: 5px;")
        layout.addWidget(desc)
        
        # Features list
        features = QLabel(
            "Features:\\n"
            "• Real-time segment preview\\n"
            "• Multi-segment selection with Shift+click\\n"
            "• Visual click indicators and confidence display\\n"
            "• Undo/redo support\\n"
            "• Direct canvas interaction"
        )
        features.setStyleSheet("color: #555; padding: 10px;")
        layout.addWidget(features)
        
        # Launch button
        launch_btn = QPushButton("🎯 Launch Interactive Mode")
        launch_btn.setStyleSheet("""
            QPushButton {
                background-color: #0078d4;
                color: white;
                border: none;
                padding: 15px;
                border-radius: 8px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #106ebe;
            }
            QPushButton:pressed {
                background-color: #005a9e;
            }
        """)
        launch_btn.clicked.connect(self._launch_interactive_mode)
        layout.addWidget(launch_btn)
        
        # Requirements
        requirements = QLabel(
            "Requirements:\\n"
            "• Active document with at least one layer\\n"
            "• Plugin must be initialized (AI models loaded)"
        )
        requirements.setStyleSheet("color: #666; font-size: 11px; padding: 5px;")
        layout.addWidget(requirements)
        
        layout.addStretch()
        return widget
    def _create_setup_wizard_tab(self) -> QWidget:
        """Create the setup wizard tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Description
        desc = QLabel(
            "Follow the steps to complete the setup of Smart Segments plugin. "
            "This includes environment configuration and model loading."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #333; padding: 10px; background-color: #f8f9fa; border-radius: 5px;")
        layout.addWidget(desc)

        # Setup step list
        setup_steps = QLabel(
            "Setup Steps:\\n"
            "1. Environment Check\\n"
            "2. GPU Setup\\n"
            "3. Model Download\\n"
            "4. Dependency Installation"
        )
        setup_steps.setStyleSheet("color: #555; padding: 10px;")
        layout.addWidget(setup_steps)

        # Start Setup button
        start_btn = QPushButton("🚀 Start Setup Wizard")
        start_btn.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                border: none;
                padding: 15px;
                border-radius: 8px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #218838;
            }
            QPushButton:pressed {
                background-color: #1e7e34;
            }
        """)
        start_btn.clicked.connect(self._start_setup_wizard)
        layout.addWidget(start_btn)

        # Troubleshooting tips
        tips = QLabel(
            "Troubleshooting Tips:\\n"
            "• Ensure all dependencies are installed.\\n"
            "• Verify internet connection for model downloads."
        )
        tips.setStyleSheet("color: #666; font-size: 11px; padding: 5px;")
        layout.addWidget(tips)

        layout.addStretch()
        return widget

    def _start_setup_wizard(self):
        """Start the setup wizard process"""
        try:
            from .setup_wizard_dialog import SetupWizardDialog
            
            wizard = SetupWizardDialog(self.extension, self)
            wizard.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start setup wizard: {e}")
        
    def _create_quick_tab(self) -> QWidget:
        """Create the quick segment tab (placeholder)"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Placeholder content
        placeholder = QLabel(
            "Quick Segment Mode\\n\\n"
            "This mode will provide one-click segmentation "
            "for common objects and backgrounds.\\n\\n"
            "Coming in future update..."
        )
        placeholder.setAlignment(Qt.AlignCenter)
        placeholder.setStyleSheet("color: #666; font-size: 12px;")
        layout.addWidget(placeholder)
        
        return widget
        
    def _create_batch_tab(self) -> QWidget:
        """Create the batch processing tab (placeholder)"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Placeholder content
        placeholder = QLabel(
            "Batch Processing Mode\\n\\n"
            "This mode will allow processing multiple layers "
            "or documents with saved segmentation settings.\\n\\n"
            "Coming in future update..."
        )
        placeholder.setAlignment(Qt.AlignCenter)
        placeholder.setStyleSheet("color: #666; font-size: 12px;")
        layout.addWidget(placeholder)
        
        return widget
        
    def _launch_interactive_mode(self):
        """Launch interactive segmentation mode"""
        try:
            # Check if plugin is initialized
            if not self.extension.is_initialized():
                QMessageBox.warning(
                    self, 
                    "Plugin Not Ready",
                    "Smart Segments is still initializing. Please wait a moment and try again."
                )
                return
                
            # Get active document and layer
            app = Krita.instance()
            doc = app.activeDocument()
            
            if not doc:
                QMessageBox.warning(
                    self,
                    "No Document",
                    "Please open a document before using interactive mode."
                )
                return
                
            active_layer = doc.activeNode()
            if not active_layer:
                QMessageBox.warning(
                    self,
                    "No Layer Selected",
                    "Please select a layer before using interactive mode."
                )
                return
                
            # Close existing interactive dialog if open
            if self.interactive_dialog:
                self.interactive_dialog.close()
                
            # Create and show interactive dialog
            self.interactive_dialog = InteractiveSegmentationDialog(
                self.extension, doc, active_layer, self
            )
            
            # Connect signals
            self.interactive_dialog.segmentation_applied.connect(
                self._on_segmentation_applied
            )
            self.interactive_dialog.dialog_closed.connect(
                self._on_interactive_dialog_closed
            )
            
            # Show dialog
            self.interactive_dialog.show()
            
            # Minimize main dialog to get out of the way
            self.showMinimized()
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to launch interactive mode: {e}"
            )
            
    def _on_segmentation_applied(self, mask, layer_name: str):
        """Handle successful segmentation application"""
        # Show notification (could be expanded)
        print(f"Segmentation applied to layer: {layer_name}")
        
    def _on_interactive_dialog_closed(self):
        """Handle interactive dialog closing"""
        # Restore main dialog
        self.showNormal()
        self.raise_()
        self.activateWindow()
        
        # Clean up reference
        self.interactive_dialog = None
        
    def closeEvent(self, event):
        """Handle main dialog close"""
        # Close interactive dialog if open
        if self.interactive_dialog:
            self.interactive_dialog.close()
            
        super().closeEvent(event)
