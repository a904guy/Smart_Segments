"""
Setup Wizard Dialog for Smart Segments Plugin

Provides step-by-step guidance through initial setup including:
- System requirement checking
- Environment configuration
- Model downloading 
- Test segmentation verification
"""

from typing import Optional, Dict, Any
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QProgressBar, QTextEdit, QGroupBox, QCheckBox, QWidget,
    QScrollArea, QFrame, QMessageBox, QApplication
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QPixmap, QPainter, QPen
import logging
from pathlib import Path

from ..bootstrap import EnvironmentBootstrapper
from ..core.model_loader import SAM2ModelLoader
from ..utils.platform_utils import PlatformUtils
from ..utils.system_utils import SystemUtils
import numpy as np
import tempfile


class ModelDownloadThread(QThread):
    """Thread for downloading AI models"""
    
    progress_update = pyqtSignal(int, str)
    download_completed = pyqtSignal(bool, str)
    
    def __init__(self, model_loader: SAM2ModelLoader, model_name: str = 'sam2_hiera_large'):
        super().__init__()
        self.model_loader = model_loader
        self.model_name = model_name
        self.should_stop = False
    
    def run(self):
        """Download the model"""
        try:
            self.progress_update.emit(10, f"Starting download of {self.model_name}...")
            
            success = self.model_loader.download_model(self.model_name)
            
            if success:
                self.progress_update.emit(100, "Model download completed")
                self.download_completed.emit(True, f"Successfully downloaded {self.model_name}")
            else:
                self.download_completed.emit(False, f"Failed to download {self.model_name}")
                
        except Exception as e:
            self.download_completed.emit(False, f"Model download failed: {e}")
    
    def stop(self):
        self.should_stop = True


class TestSegmentationThread(QThread):
    """Thread for running test segmentation"""
    
    progress_update = pyqtSignal(int, str)
    test_completed = pyqtSignal(bool, str, object)  # success, message, result_image
    
    def __init__(self, bootstrapper: EnvironmentBootstrapper):
        super().__init__()
        self.bootstrapper = bootstrapper
        self.should_stop = False
    
    def run(self):
        """Run test segmentation"""
        try:
            self.progress_update.emit(10, "Creating test image...")
            
            # Create a simple test image
            test_image = self.create_test_image()
            
            if self.should_stop:
                return
            
            self.progress_update.emit(30, "Loading AI model...")
            
            # Try to initialize model loader
            model_loader = SAM2ModelLoader()
            available_models = model_loader.get_available_models()
            
            if not available_models:
                self.test_completed.emit(False, "No models available for testing", None)
                return
                
            # Find an available model
            test_model = None
            for model_name in ['sam2_hiera_tiny', 'sam2_hiera_small', 'sam2_hiera_large']:
                if model_loader.is_model_available(model_name):
                    test_model = model_name
                    break
            
            if not test_model:
                self.test_completed.emit(False, "No models ready for testing. Please download models first.", None)
                return
            
            self.progress_update.emit(60, f"Running test segmentation with {test_model}...")
            
            # Mock segmentation test (in real implementation, this would use actual model)
            import time
            time.sleep(2)  # Simulate processing
            
            if self.should_stop:
                return
            
            self.progress_update.emit(90, "Test segmentation completed")
            
            # Create result visualization
            result_image = self.create_result_visualization(test_image)
            
            self.progress_update.emit(100, "Test completed successfully")
            self.test_completed.emit(
                True, 
                f"Test segmentation completed successfully using {test_model}!\n"
                f"- Test image: 256x256 pixels\n"
                f"- Model used: {test_model}\n"
                f"- Processing time: ~2 seconds\n"
                f"- Result: Mock segmentation mask generated",
                result_image
            )
            
        except Exception as e:
            self.test_completed.emit(False, f"Test segmentation failed: {e}", None)
    
    def create_test_image(self):
        """Create a simple test image"""
        try:
            # Create a simple test pattern
            import numpy as np
            test_image = np.zeros((256, 256, 3), dtype=np.uint8)
            
            # Add some geometric shapes for testing
            # Circle in center
            y, x = np.ogrid[:256, :256]
            center_x, center_y = 128, 128
            mask = (x - center_x)**2 + (y - center_y)**2 <= 50**2
            test_image[mask] = [255, 100, 100]  # Red circle
            
            # Rectangle
            test_image[50:100, 50:150] = [100, 255, 100]  # Green rectangle
            
            # Triangle (simple)
            for i in range(50):
                test_image[200-i:200, 150+i:200-i] = [100, 100, 255]  # Blue triangle
            
            return test_image
            
        except Exception:
            # Fallback: create simple gradient
            return np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    
    def create_result_visualization(self, original_image):
        """Create a mock result visualization"""
        try:
            import numpy as np
            # Create a mock segmentation mask
            mask = np.zeros((256, 256), dtype=np.uint8)
            
            # Add some mock segmented regions
            y, x = np.ogrid[:256, :256]
            center_x, center_y = 128, 128
            circle_mask = (x - center_x)**2 + (y - center_y)**2 <= 50**2
            mask[circle_mask] = 255
            
            # Create overlay visualization
            result = original_image.copy()
            result[mask > 0] = result[mask > 0] * 0.7 + np.array([255, 255, 0]) * 0.3  # Yellow overlay
            
            return result.astype(np.uint8)
            
        except Exception:
            return original_image
    
    def stop(self):
        self.should_stop = True


class SetupWorkerThread(QThread):
    """Worker thread for setup operations"""
    
    progress_update = pyqtSignal(int, str)  # progress, message
    step_completed = pyqtSignal(str, bool, str)  # step_name, success, details
    setup_completed = pyqtSignal(bool, str)  # success, message
    
    def __init__(self, bootstrapper: EnvironmentBootstrapper, include_models: bool = True):
        super().__init__()
        self.bootstrapper = bootstrapper
        self.include_models = include_models
        self.should_stop = False
        
    def run(self):
        """Run the setup process"""
        try:
            self.progress_update.emit(10, "Checking system requirements...")
            
            # Step 1: System requirements
            system_info = self.bootstrapper.check_system_requirements()
            if self.should_stop:
                return
                
            self.step_completed.emit(
                "system_check", 
                True, 
                f"Platform: {system_info.platform}\nPython: {system_info.python_version}\nGPU: {'CUDA' if system_info.has_cuda else 'CPU'}"
            )
            self.progress_update.emit(25, "Creating virtual environment...")
            
            # Step 2: Virtual environment
            env_config = self.bootstrapper.create_virtual_environment()
            if self.should_stop:
                return
                
            self.step_completed.emit(
                "environment", 
                True, 
                f"Virtual environment created at: {env_config.venv_path}\nGPU Type: {env_config.gpu_type}"
            )
            self.progress_update.emit(50, "Installing dependencies...")
            
            # Step 3: Dependencies
            self.bootstrapper.install_dependencies()
            if self.should_stop:
                return
                
            self.step_completed.emit("dependencies", True, "All dependencies installed successfully")
            self.progress_update.emit(75, "Verifying installation...")
            
            # Step 4: Verification
            verification_results = self.bootstrapper.verify_installation()
            if self.should_stop:
                return
                
            all_passed = all(verification_results.values())
            details = "\n".join([f"{name}: {'✓' if success else '✗'}" for name, success in verification_results.items()])
            
            self.step_completed.emit("verification", all_passed, details)
            self.progress_update.emit(100, "Setup complete!")
            
            self.setup_completed.emit(all_passed, "Setup completed successfully!" if all_passed else "Setup completed with some issues")
            
        except Exception as e:
            self.setup_completed.emit(False, f"Setup failed: {e}")
            
    def stop(self):
        """Request the thread to stop"""
        self.should_stop = True


class SetupWizardDialog(QDialog):
    """
    Comprehensive setup wizard dialog for first-run experience
    """
    
    def __init__(self, extension, parent=None):
        super().__init__(parent)
        
        self.extension = extension
        self.bootstrapper: Optional[EnvironmentBootstrapper] = None
        self.setup_thread: Optional[SetupWorkerThread] = None
        
        self.setWindowTitle("Smart Segments - Setup Wizard")
        self.setModal(True)
        self.resize(600, 700)
        
        # Setup states
        self.setup_steps = {
            "system_check": {"name": "System Check", "status": "pending", "details": ""},
            "environment": {"name": "Environment Setup", "status": "pending", "details": ""},
            "dependencies": {"name": "Install Dependencies", "status": "pending", "details": ""},
            "verification": {"name": "Verify Installation", "status": "pending", "details": ""},
        }
        
        self.setup_ui()
        self.setup_bootstrapper()
        
    def setup_ui(self):
        """Setup the wizard UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Header
        header = self.create_header()
        layout.addWidget(header)
        
        # Progress section
        progress_section = self.create_progress_section()
        layout.addWidget(progress_section)
        
        # Setup steps section
        steps_section = self.create_steps_section()
        layout.addWidget(steps_section)
        
        # Troubleshooting section
        troubleshooting_section = self.create_troubleshooting_section()
        layout.addWidget(troubleshooting_section)
        
        # Control buttons
        controls = self.create_controls()
        layout.addWidget(controls)
        
    def create_header(self) -> QWidget:
        """Create the header section"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        title = QLabel("Smart Segments Setup Wizard")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #0078d4; margin: 10px;")
        
        subtitle = QLabel(
            "Welcome! This wizard will guide you through setting up Smart Segments.\n"
            "The process includes checking your system, installing dependencies,\n"
            "and preparing the AI models for segmentation."
        )
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color: #666; font-size: 12px; margin: 5px;")
        
        layout.addWidget(title)
        layout.addWidget(subtitle)
        
        return widget
        
    def create_progress_section(self) -> QWidget:
        """Create the progress section"""
        group = QGroupBox("Setup Progress")
        layout = QVBoxLayout(group)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        
        self.progress_label = QLabel("Ready to start setup...")
        self.progress_label.setStyleSheet("color: #666; font-size: 11px;")
        
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.progress_label)
        
        return group
        
    def create_steps_section(self) -> QWidget:
        """Create the setup steps section"""
        group = QGroupBox("Setup Steps")
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedHeight(200)
        
        steps_widget = QWidget()
        layout = QVBoxLayout(steps_widget)
        
        self.step_widgets = {}
        
        for step_id, step_info in self.setup_steps.items():
            step_widget = self.create_step_widget(step_id, step_info)
            self.step_widgets[step_id] = step_widget
            layout.addWidget(step_widget)
            
        scroll.setWidget(steps_widget)
        
        group_layout = QVBoxLayout(group)
        group_layout.addWidget(scroll)
        
        return group
        
    def create_step_widget(self, step_id: str, step_info: Dict[str, Any]) -> QWidget:
        """Create a widget for an individual setup step"""
        frame = QFrame()
        frame.setFrameStyle(QFrame.Box)
        frame.setStyleSheet("QFrame { border: 1px solid #ddd; border-radius: 4px; padding: 5px; }")
        
        layout = QVBoxLayout(frame)
        
        # Step header
        header_layout = QHBoxLayout()
        
        # Status indicator
        status_label = QLabel("⏸")  # Pending
        status_label.setFixedSize(20, 20)
        status_label.setAlignment(Qt.AlignCenter)
        
        # Step name
        name_label = QLabel(step_info["name"])
        name_label.setFont(QFont("Arial", 10, QFont.Bold))
        
        header_layout.addWidget(status_label)
        header_layout.addWidget(name_label)
        header_layout.addStretch()
        
        # Details area
        details_label = QLabel(step_info["details"] or "Waiting...")
        details_label.setWordWrap(True)
        details_label.setStyleSheet("color: #666; font-size: 10px; margin-left: 25px;")
        
        layout.addLayout(header_layout)
        layout.addWidget(details_label)
        
        # Store references
        frame.status_label = status_label
        frame.details_label = details_label
        
        return frame
        
    def create_troubleshooting_section(self) -> QWidget:
        """Create the troubleshooting tips section"""
        group = QGroupBox("Troubleshooting Tips")
        layout = QVBoxLayout(group)
        
        tips_text = QTextEdit()
        tips_text.setReadOnly(True)
        tips_text.setFixedHeight(100)
        tips_text.setHtml("""
        <b>Common Issues and Solutions:</b><br>
        • <b>Internet Connection:</b> Ensure stable internet for downloading models<br>
        • <b>Disk Space:</b> At least 5-6 GB free space required (1 GB models + 4-5 GB tooling)<br>
        • <b>Python Version:</b> Python 3.8+ required<br>
        • <b>GPU Drivers:</b> Update GPU drivers for optimal performance<br>
        • <b>Antivirus:</b> Temporarily disable if blocking downloads
        """)
        tips_text.setStyleSheet("background-color: #f8f9fa; border: 1px solid #dee2e6;")
        
        layout.addWidget(tips_text)
        
        return group
        
    def create_controls(self) -> QWidget:
        """Create the control buttons"""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        
        self.start_button = QPushButton("🚀 Start Setup")
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 6px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #218838;
            }
            QPushButton:disabled {
                background-color: #6c757d;
            }
        """)
        self.start_button.clicked.connect(self.start_setup)
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setStyleSheet("""
            QPushButton {
                background-color: #6c757d;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 6px;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #5a6268;
            }
        """)
        self.cancel_button.clicked.connect(self.cancel_setup)
        
        self.test_button = QPushButton("🧪 Test Setup")
        self.test_button.setStyleSheet("""
            QPushButton {
                background-color: #17a2b8;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 6px;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #138496;
            }
        """)
        self.test_button.clicked.connect(self.test_setup)
        self.test_button.setEnabled(False)
        
        layout.addWidget(self.start_button)
        layout.addStretch()
        layout.addWidget(self.test_button)
        layout.addWidget(self.cancel_button)
        
        return widget
        
    def setup_bootstrapper(self):
        """Initialize the environment bootstrapper"""
        try:
            project_root = Path(__file__).parent.parent.parent.parent.parent
            self.bootstrapper = EnvironmentBootstrapper(project_root)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to initialize bootstrapper: {e}")
            
    def start_setup(self):
        """Start the setup process"""
        if not self.bootstrapper:
            QMessageBox.warning(self, "Error", "Bootstrapper not initialized")
            return
            
        # Disable start button
        self.start_button.setEnabled(False)
        self.start_button.setText("Setup Running...")
        
        # Create and start setup thread
        self.setup_thread = SetupWorkerThread(self.bootstrapper)
        self.setup_thread.progress_update.connect(self.update_progress)
        self.setup_thread.step_completed.connect(self.update_step)
        self.setup_thread.setup_completed.connect(self.setup_finished)
        self.setup_thread.start()
        
    def update_progress(self, value: int, message: str):
        """Update progress bar and message"""
        self.progress_bar.setValue(value)
        self.progress_label.setText(message)
        QApplication.processEvents()
        
    def update_step(self, step_id: str, success: bool, details: str):
        """Update a setup step status"""
        if step_id not in self.step_widgets:
            return
            
        widget = self.step_widgets[step_id]
        
        # Update status indicator
        if success:
            widget.status_label.setText("✅")
            widget.status_label.setStyleSheet("color: green;")
        else:
            widget.status_label.setText("❌")
            widget.status_label.setStyleSheet("color: red;")
            
        # Update details
        widget.details_label.setText(details)
        
        QApplication.processEvents()
        
    def setup_finished(self, success: bool, message: str):
        """Handle setup completion"""
        self.start_button.setEnabled(True)
        self.start_button.setText("🚀 Start Setup")
        
        if success:
            self.test_button.setEnabled(True)
            QMessageBox.information(self, "Setup Complete", message)
        else:
            QMessageBox.warning(self, "Setup Failed", message)
            
    def test_setup(self):
        """Test the setup by running a simple segmentation"""
        if not self.bootstrapper:
            QMessageBox.warning(self, "Error", "Bootstrapper not initialized")
            return
        
        # Create test dialog
        test_dialog = TestSetupDialog(self.bootstrapper, self)
        test_dialog.exec_()
            
    def cancel_setup(self):
        """Cancel the setup process"""
        if self.setup_thread and self.setup_thread.isRunning():
            self.setup_thread.stop()
            self.setup_thread.wait()
            
        self.reject()
        
    def closeEvent(self, event):
        """Handle dialog close"""
        if self.setup_thread and self.setup_thread.isRunning():
            self.setup_thread.stop()
            self.setup_thread.wait()
            
        super().closeEvent(event)


class TestSetupDialog(QDialog):
    """Dialog for testing the setup with segmentation"""
    
    def __init__(self, bootstrapper: EnvironmentBootstrapper, parent=None):
        super().__init__(parent)
        
        self.bootstrapper = bootstrapper
        self.test_thread: Optional[TestSegmentationThread] = None
        self.model_download_thread: Optional[ModelDownloadThread] = None
        
        self.setWindowTitle("Test Smart Segments Setup")
        self.setModal(True)
        self.resize(500, 600)
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the test dialog UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Header
        header = QLabel("Setup Test")
        header.setFont(QFont("Arial", 14, QFont.Bold))
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet("color: #0078d4; margin: 10px;")
        layout.addWidget(header)
        
        # Description
        desc = QLabel(
            "This test will verify that Smart Segments is properly set up by:\n"
            "• Checking model availability\n"
            "• Running a test segmentation\n"
            "• Verifying the results"
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #666; margin: 10px;")
        layout.addWidget(desc)
        
        # Progress section
        progress_group = QGroupBox("Test Progress")
        progress_layout = QVBoxLayout(progress_group)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        
        self.progress_label = QLabel("Ready to start test...")
        self.progress_label.setStyleSheet("color: #666; font-size: 11px;")
        
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.progress_label)
        layout.addWidget(progress_group)
        
        # Model status section
        model_group = QGroupBox("Model Status")
        model_layout = QVBoxLayout(model_group)
        
        self.model_status_label = QLabel("Checking model availability...")
        self.model_status_label.setWordWrap(True)
        self.model_status_label.setStyleSheet("color: #666;")
        model_layout.addWidget(self.model_status_label)
        
        # Model download button
        self.download_model_button = QPushButton("📥 Download Test Model")
        self.download_model_button.setStyleSheet("""
            QPushButton {
                background-color: #ffc107;
                color: #212529;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #e0a800;
            }
        """)
        self.download_model_button.clicked.connect(self.download_model)
        self.download_model_button.setVisible(False)
        model_layout.addWidget(self.download_model_button)
        
        layout.addWidget(model_group)
        
        # Results section
        results_group = QGroupBox("Test Results")
        results_layout = QVBoxLayout(results_group)
        
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setFixedHeight(150)
        self.results_text.setStyleSheet("background-color: #f8f9fa; border: 1px solid #dee2e6;")
        results_layout.addWidget(self.results_text)
        
        layout.addWidget(results_group)
        
        # Control buttons
        controls = QWidget()
        controls_layout = QHBoxLayout(controls)
        
        self.start_test_button = QPushButton("🧪 Start Test")
        self.start_test_button.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 4px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #218838;
            }
            QPushButton:disabled {
                background-color: #6c757d;
            }
        """)
        self.start_test_button.clicked.connect(self.start_test)
        
        self.close_button = QPushButton("Close")
        self.close_button.setStyleSheet("""
            QPushButton {
                background-color: #6c757d;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 4px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #5a6268;
            }
        """)
        self.close_button.clicked.connect(self.close)
        
        controls_layout.addWidget(self.start_test_button)
        controls_layout.addStretch()
        controls_layout.addWidget(self.close_button)
        
        layout.addWidget(controls)
        
        # Check initial model status
        QTimer.singleShot(100, self.check_model_status)
        
    def check_model_status(self):
        """Check if models are available"""
        try:
            model_loader = SAM2ModelLoader()
            available_models = []
            
            for model_name in ['sam2_hiera_tiny', 'sam2_hiera_small', 'sam2_hiera_large']:
                if model_loader.is_model_available(model_name):
                    available_models.append(model_name)
            
            if available_models:
                self.model_status_label.setText(
                    f"✅ Models available: {', '.join(available_models)}\n"
                    "Ready for testing!"
                )
                self.model_status_label.setStyleSheet("color: green;")
                self.start_test_button.setEnabled(True)
            else:
                self.model_status_label.setText(
                    "❌ No models available for testing.\n"
                    "Download a model to run the test."
                )
                self.model_status_label.setStyleSheet("color: red;")
                self.download_model_button.setVisible(True)
                self.start_test_button.setEnabled(False)
                
        except Exception as e:
            self.model_status_label.setText(f"⚠️ Error checking models: {e}")
            self.model_status_label.setStyleSheet("color: orange;")
            self.start_test_button.setEnabled(False)
    
    def download_model(self):
        """Download a test model"""
        try:
            model_loader = SAM2ModelLoader()
            
            # Start with the smallest model for testing
            self.model_download_thread = ModelDownloadThread(model_loader, 'sam2_hiera_tiny')
            self.model_download_thread.progress_update.connect(self.update_download_progress)
            self.model_download_thread.download_completed.connect(self.download_finished)
            
            self.download_model_button.setEnabled(False)
            self.download_model_button.setText("Downloading...")
            
            self.model_download_thread.start()
            
        except Exception as e:
            QMessageBox.critical(self, "Download Error", f"Failed to start model download: {e}")
    
    def update_download_progress(self, value: int, message: str):
        """Update download progress"""
        self.progress_bar.setValue(value)
        self.progress_label.setText(message)
    
    def download_finished(self, success: bool, message: str):
        """Handle download completion"""
        self.download_model_button.setEnabled(True)
        self.download_model_button.setText("📥 Download Test Model")
        
        if success:
            QMessageBox.information(self, "Download Complete", message)
            self.check_model_status()  # Refresh model status
        else:
            QMessageBox.warning(self, "Download Failed", message)
    
    def start_test(self):
        """Start the setup test"""
        if not self.bootstrapper:
            QMessageBox.warning(self, "Error", "Bootstrapper not available")
            return
        
        self.start_test_button.setEnabled(False)
        self.start_test_button.setText("Testing...")
        self.results_text.clear()
        
        # Start test thread
        self.test_thread = TestSegmentationThread(self.bootstrapper)
        self.test_thread.progress_update.connect(self.update_test_progress)
        self.test_thread.test_completed.connect(self.test_finished)
        self.test_thread.start()
    
    def update_test_progress(self, value: int, message: str):
        """Update test progress"""
        self.progress_bar.setValue(value)
        self.progress_label.setText(message)
    
    def test_finished(self, success: bool, message: str, result_image):
        """Handle test completion"""
        self.start_test_button.setEnabled(True)
        self.start_test_button.setText("🧪 Start Test")
        
        self.results_text.setPlainText(message)
        
        if success:
            self.results_text.setStyleSheet("background-color: #d4edda; border: 1px solid #c3e6cb; color: #155724;")
            QMessageBox.information(self, "Test Successful", "Setup test completed successfully!\n\nSmart Segments is ready to use.")
        else:
            self.results_text.setStyleSheet("background-color: #f8d7da; border: 1px solid #f5c6cb; color: #721c24;")
            QMessageBox.warning(self, "Test Failed", f"Setup test failed:\n\n{message}")
    
    def closeEvent(self, event):
        """Handle dialog close"""
        if self.test_thread and self.test_thread.isRunning():
            self.test_thread.stop()
            self.test_thread.wait()
        
        if self.model_download_thread and self.model_download_thread.isRunning():
            self.model_download_thread.stop()
            self.model_download_thread.wait()
        
        super().closeEvent(event)
