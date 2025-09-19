"""
Smart Segments Extension for Krita
AI-powered intelligent segmentation tool
"""

import os
import sys
import subprocess
import logging
import traceback
import json
from pathlib import Path
from typing import Optional, Dict, Any, List

from PyQt5.QtCore import QObject, QTimer, pyqtSignal
from PyQt5.QtWidgets import QMessageBox, QProgressDialog, QApplication
from krita import Extension, Krita, Document, Node

# Import core modules lazily to prevent startup failures
# These will be imported only when needed
SegmentationAPI = None
KritaImageBridge = None 
EnvironmentBootstrapper = None
SmartSegmentsDialog = None


class SmartSegmentsExtension(Extension):
    """Main extension class for Smart Segments plugin"""
    
    # Plugin signals
    initialization_complete = pyqtSignal(bool)  # Success/failure
    segmentation_complete = pyqtSignal(object)  # SegmentationResult
    error_occurred = pyqtSignal(str)  # Error message
    
    def __init__(self, parent):
        super().__init__(parent)
        
        # Plugin state
        self._initialized = False
        self._initialization_in_progress = False
        self._segmentation_api: Optional[SegmentationAPI] = None
        self._krita_bridge: Optional[KritaImageBridge] = None
        self._current_document: Optional[Document] = None
        self._active_session_id: Optional[str] = None
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Environment management
        self._bootstrapper: Optional[EnvironmentBootstrapper] = None
        self._environment_ready = False
        
        # Plugin configuration
        self.config = {
            'model_name': 'sam2_hiera_large',
            'device': None,  # Auto-detect
            'max_sessions': 5,
            'auto_initialize': True,
            'use_gpu_if_available': True
        }

        # Setup state file path
        plugin_file = Path(__file__).resolve()
        self.project_root = plugin_file.parent.parent.parent
        self.state_file = self.project_root / "plugin_state.json"
        
        # Load persistent state
        self._load_plugin_state()
        
        self.logger.info("Smart Segments Extension initialized")
        
    def _lazy_load_dependencies(self, minimal_only=False):
        """Lazy load complex dependencies to avoid blocking plugin registration"""
        global SegmentationAPI, KritaImageBridge, EnvironmentBootstrapper, SmartSegmentsDialog

        # Ensure venv is in path before trying to import from it
        try:
            plugin_file = Path(__file__).resolve()
            project_root = plugin_file.parent.parent.parent
            venv_path = project_root / "venv"
            if venv_path.exists():
                candidates = []
                # Windows venv site-packages
                if os.name == 'nt':
                    candidates.append(venv_path / "Lib" / "site-packages")
                # Unix-like venv site-packages
                candidates.extend(venv_path.glob("lib/python*/site-packages"))
                for sp in candidates:
                    if sp.exists():
                        sp_str = str(sp)
                        if sp_str not in sys.path:
                            self.logger.info(f"Adding venv site-packages to sys.path: {sp_str}")
                            sys.path.insert(0, sp_str)
                        break
        except Exception as e:
            self.logger.warning(f"Could not add venv to sys.path: {e}")

        if minimal_only:
            # Load only minimal bootstrap for initial setup
            try:
                from .minimal_bootstrap import MinimalBootstrap
                return MinimalBootstrap
            except ImportError as e:
                self.logger.error(f"Failed to load minimal bootstrap: {e}")
                raise
        
        if SegmentationAPI is None:
            try:
                self.logger.info("Attempting to import core dependencies...")
                
                self.logger.info("Importing SegmentationAPI...")
                from .core.segmentation_api import SegmentationAPI
                self.logger.info(f"...SegmentationAPI imported: {SegmentationAPI}")
                
                self.logger.info("Importing KritaImageBridge...")
                from .utils.krita_bridge import KritaImageBridge
                self.logger.info(f"...KritaImageBridge imported: {KritaImageBridge}")
                
                self.logger.info("Importing EnvironmentBootstrapper...")
                from .bootstrap import EnvironmentBootstrapper
                self.logger.info(f"...EnvironmentBootstrapper imported: {EnvironmentBootstrapper}")
                
                self.logger.info("Importing SmartSegmentsDialog...")
                from .ui.main_dialog import SmartSegmentsDialog
                self.logger.info(f"...SmartSegmentsDialog imported: {SmartSegmentsDialog}")
                
                self.logger.info("Dependencies loaded successfully")

            except Exception as e:
                self.logger.error(f"CRITICAL: Failed to load dependencies: {e}", exc_info=True)
                raise
        
    def _get_krita_config_dir(self) -> Path:
        """Get Krita's configuration directory in a cross-platform way"""
        from pathlib import Path
        import os
        import platform
        
        if platform.system() == "Windows":
            return Path(os.path.expanduser("~")) / "AppData" / "Roaming" / "krita"
        elif platform.system() == "Darwin":  # macOS
            return Path(os.path.expanduser("~")) / "Library" / "Application Support" / "krita"
        else:  # Linux and others
            # Try XDG_CONFIG_HOME first, fallback to ~/.config
            xdg_config = os.environ.get('XDG_CONFIG_HOME')
            if xdg_config:
                return Path(xdg_config) / "krita"
            else:
                return Path(os.path.expanduser("~")) / ".config" / "krita"
    
    def _setup_logging(self) -> logging.Logger:
        """Setup plugin logging"""
        logger = logging.getLogger("smart_segments_plugin")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            log_dir = Path(__file__).parent / "logs"
            log_dir.mkdir(exist_ok=True)
            
            # File handler
            log_file = log_dir / "smart_segments.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            # Formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
        
        return logger
        
    def setup(self):
        """Setup the extension - called by Krita"""
        try:
            self.logger.info("Setting up Smart Segments extension...")
            
            # Defer complex initialization to avoid blocking plugin loading
            # Just mark that we're ready to create menu items
            self.logger.info("Smart Segments extension setup complete")
            
        except Exception as e:
            self.logger.error(f"Failed to setup extension: {e}")
            self.logger.error(traceback.format_exc())
            
    def _initialize_async(self):
        """Initialize the plugin asynchronously"""
        if self._initialization_in_progress or self._initialized:
            return
            
        self._initialization_in_progress = True
        
        try:
            self.logger.info("Starting async initialization...")
            
            # Show progress dialog
            progress = QProgressDialog("Initializing Smart Segments...", "Cancel", 0, 100)
            progress.setWindowTitle("Smart Segments")
            progress.show()
            QApplication.processEvents()
            
            # Check/setup environment
            progress.setValue(20)
            progress.setLabelText("Checking environment...")
            QApplication.processEvents()
            
            if not self._check_environment():
                progress.close()
                # Environment not ready - trigger setup process
                self.logger.info("Environment not ready, triggering setup...")
                try:
                    MinimalBootstrap = self._lazy_load_dependencies(minimal_only=True)
                    self._run_minimal_setup(MinimalBootstrap)
                    return  # Setup will handle the initialization
                except Exception as setup_error:
                    self.logger.error(f"Failed to start setup: {setup_error}")
                    self._show_error_dialog(
                        "Setup Failed",
                        f"Failed to start environment setup: {setup_error}\n\n"
                        "Please check the logs and try restarting Krita."
                    )
                    return
            
            # Load dependencies first
            progress.setValue(40)
            progress.setLabelText("Loading dependencies...")
            QApplication.processEvents()
            
            self._lazy_load_dependencies()  # This loads SegmentationAPI and other classes
            
            # Initialize segmentation API
            progress.setValue(60)
            progress.setLabelText("Loading AI models...")
            QApplication.processEvents()
            
            self._segmentation_api = SegmentationAPI(
                model_name=self.config['model_name'],
                device=self.config['device'],
                max_sessions=self.config['max_sessions']
            )
            
            progress.setValue(100)
            progress.close()
            
            self._initialized = True
            self._initialization_in_progress = False
            
            self.logger.info("Smart Segments initialization complete")
            self.initialization_complete.emit(True)
            
        except Exception as e:
            self._initialization_in_progress = False
            error_msg = f"Failed to initialize Smart Segments: {e}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            
            if 'progress' in locals():
                progress.close()
            
            # Check if this is a CUDA-related error and show specialized dialog
            error_str = str(e).lower()
            try:
                if "no kernel image is available" in error_str or ("cuda" in error_str and ("error" in error_str or "failed" in error_str)):
                    self._show_cuda_error_dialog(str(e))
                else:
                    self._show_error_dialog("Initialization Failed", error_msg)
            except Exception:
                pass
            
            self.initialization_complete.emit(False)
            self.error_occurred.emit(error_msg)
            
    def _check_environment(self) -> bool:
        """Check if the required environment is ready"""
        try:
            # Use Krita's resource path for cross-platform compatibility
            project_root = self._get_krita_config_dir()
            venv_path = project_root / "venv"
            
            # Check if virtual environment exists
            if not venv_path.exists():
                self.logger.error(f"Virtual environment does not exist at {venv_path}")
                return False
            
            # Check if python executable exists (try multiple possibilities on Windows)
            if os.name == 'nt':  # Windows
                python_candidates = [
                    venv_path / "Scripts" / "python.exe",
                    venv_path / "Scripts" / "python3.exe"
                ]
                python_path = None
                for candidate in python_candidates:
                    if candidate.exists():
                        python_path = candidate
                        break
                
                if python_path is None:
                    self.logger.error(f"Python executable not found. Tried: {python_candidates}")
                    return False
            else:  # Unix/Linux/macOS
                python_path = venv_path / "bin" / "python"
                if not python_path.exists():
                    self.logger.error(f"Python executable not found at {python_path}")
                    return False
            
            # Test basic functionality
            try:
                result = subprocess.run([
                    str(python_path), '-c', 'import sys; print("Virtual env working")'
                ], capture_output=True, text=True, timeout=30)
                
                if result.returncode != 0:
                    self.logger.error(f"Virtual environment test failed: {result.stderr}")
                    return False
            except Exception as e:
                self.logger.error(f"Virtual environment test error: {e}")
                return False
            
            # Test required dependencies
            required_modules = ['numpy', 'torch', 'cv2', 'PIL', 'sam2']
            missing_deps = []
            
            for module in required_modules:
                try:
                    result = subprocess.run([
                        str(python_path), '-c', f'import {module}; print("{module} OK")'
                    ], capture_output=True, text=True, timeout=30)
                    
                    if result.returncode != 0:
                        missing_deps.append(module)
                        self.logger.warning(f"Missing dependency: {module}")
                except Exception as e:
                    missing_deps.append(module)
                    self.logger.warning(f"Error checking {module}: {e}")
            
            if missing_deps:
                self.logger.error(f"Missing required dependencies: {missing_deps}")
                return False
            
            self.logger.info("Environment verification successful - all dependencies available")
            self._environment_ready = True
            return True
            
        except Exception as e:
            self.logger.error(f"Environment check failed: {e}")
            return False
            
            
    def createActions(self, window):
        """Create actions for the plugin - called by Krita"""
        try:
            # Main segmentation action (merged functionality)
            action = window.createAction("smart_segments_dialog", "Smart Segments", "tools/scripts")
            action.triggered.connect(self.show_smart_segments)
            
            # Settings action
            settings_action = window.createAction("smart_segments_settings", "Smart Settings Technicals", "tools/scripts")
            settings_action.triggered.connect(self.show_settings_dialog)
            
            self.logger.info("Plugin actions created")
            
        except Exception as e:
            self.logger.error(f"Failed to create actions: {e}")
            
    def show_smart_segments(self):
        """Show Smart Segments - unified interface for both interactive and quick segment modes"""
        try:
            # Check if we have a document and layer first
            doc = Krita.instance().activeDocument()
            if not doc:
                self._show_error_dialog("No Document", "Please open a document first.")
                return
                
            active_node = doc.activeNode()
            if not active_node:
                self._show_error_dialog("No Layer", "Please select a layer first.")
                return
            
            # Check if plugin is initialized first
            if not self._initialized:
                # Check if environment is already set up
                if self._is_setup_completed():
                    # Environment is ready, just initialize
                    self.logger.info("Environment ready, initializing plugin...")
                    self._initialize_async()
                    
                    # Check if initialization completed synchronously
                    if self._initialized:
                        self._show_smart_segments_window(doc, active_node)
                    # Otherwise, rely on the async initializer to show progress/errors
                    return
                
                # Ask user if they want to start setup
                msg = QMessageBox()
                msg.setWindowTitle("Smart Segments - Setup Required")
                msg.setText("Welcome to Smart Segments!")
                msg.setInformativeText(
                    "This is your first time using Smart Segments.\n\n"
                    "The plugin needs to:\n"
                    "• Download AI models (~1 GB)\n"
                    "• Install PyTorch and dependencies\n"
                    "• Set up the segmentation environment\n\n"
                    "This process may take 10-15 minutes with internet connection.\n\n"
                    "Would you like to start the setup now?"
                )
                msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
                msg.setDefaultButton(QMessageBox.Yes)
                
                if msg.exec_() == QMessageBox.Yes:
                    # Use minimal bootstrap for initial setup
                    try:
                        MinimalBootstrap = self._lazy_load_dependencies(minimal_only=True)
                        self._run_minimal_setup(MinimalBootstrap)
                    except Exception as dep_error:
                        self.logger.error(f"Failed to load minimal bootstrap: {dep_error}")
                        self._show_error_dialog(
                            "Setup Dependencies Missing",
                            "Cannot start setup because minimal bootstrap is missing.\n\n"
                            "This usually means the plugin installation is incomplete.\n"
                            "Please reinstall the plugin or contact support."
                        )
                return
                
            # If initialized, show the Smart Segments window
            self._show_smart_segments_window(doc, active_node)
            
        except Exception as e:
            self.logger.error(f"Failed to show Smart Segments: {e}")
            self._show_error_dialog("Failed to open Smart Segments", str(e))
            
    def _show_smart_segments_window(self, document: Document, layer: Node):
        """Show the unified Smart Segments window"""
        try:
            # Perform quick segmentation first to get all masks
            self.logger.info(f"Starting segmentation for layer: {layer.name()}")
            
            # Additional layer validation
            layer_info = {
                'name': layer.name(),
                'type': layer.type(),
                'visible': layer.visible(),
                'bounds': layer.bounds()
            }
            self.logger.info(f"Layer validation - Info: {layer_info}")
            
            # Check if layer has valid bounds
            bounds = layer.bounds()
            if bounds.width() <= 0 or bounds.height() <= 0:
                self._show_error_dialog(
                    "Invalid Layer", 
                    f"The selected layer '{layer.name()}' has invalid dimensions ({bounds.width()}x{bounds.height()}). Please select a layer with content."
                )
                return
            
            # Load dependencies
            self.logger.info("Loading dependencies...")
            self._lazy_load_dependencies()
            self.logger.info("Dependencies loaded successfully")
            
            # Show progress dialog
            progress = QProgressDialog("Analyzing image...", "Cancel", 0, 100)
            progress.setWindowTitle("Smart Segments")
            progress.show()
            QApplication.processEvents()
            
            # Initialize Krita bridge if not done
            if not self._krita_bridge:
                self.logger.info("Initializing Krita bridge...")
                self._krita_bridge = KritaImageBridge()
                self.logger.info("Krita bridge initialized")
            
            # Convert layer to numpy array
            progress.setValue(20)
            progress.setLabelText("Converting layer data...")
            QApplication.processEvents()
            
            self.logger.info(f"Converting layer '{layer.name()}' to numpy array...")
            image_array = self._krita_bridge.layer_to_numpy(layer)
            if image_array is None:
                progress.close()
                self.logger.error("Layer conversion failed - image_array is None")
                self._show_error_dialog("Layer Conversion Failed", "Could not convert layer to image data.")
                return
            
            self.logger.info(f"Layer converted successfully: shape={image_array.shape}, dtype={image_array.dtype}")
            
            # Create segmentation session
            progress.setValue(40)
            progress.setLabelText("Setting up AI model...")
            QApplication.processEvents()
            
            self.logger.info(f"Creating segmentation session with image shape: {image_array.shape}")
            if not self._segmentation_api:
                progress.close()
                self.logger.error("Segmentation API is None - initialization may have failed")
                self._show_error_dialog("Initialization Error", "Segmentation API not available. Try restarting Krita.")
                return
                
            session_id = self._segmentation_api.create_session(image_array)
            if not session_id:
                progress.close()
                self.logger.error("Session creation failed - session_id is None")
                self._show_error_dialog("Session Creation Failed", "Could not create segmentation session.")
                return
            
            self.logger.info(f"Segmentation session created successfully: {session_id}")
            
            # Generate all masks
            progress.setValue(60)
            progress.setLabelText("Generating segmentation masks...")
            QApplication.processEvents()
            
            self.logger.info(f"Starting segment_everything for session: {session_id}")
            all_masks = self._segmentation_api.segment_everything(session_id)
            if not all_masks:
                progress.close()
                self.logger.error("Segmentation failed - no masks generated")
                self._show_error_dialog("Segmentation Failed", "Could not generate segmentation masks.")
                return
            
            self.logger.info(f"Segmentation completed successfully - generated {len(all_masks)} masks")
            
            progress.setValue(100)
            progress.close()
            
            # Show Smart Segments window (renamed Quick Segment dialog)
            self._show_smart_segments_selector(document, layer, all_masks, session_id)
            
        except Exception as e:
            if 'progress' in locals():
                progress.close()
            self.logger.error(f"Smart Segments failed: {e}")
            self._show_error_dialog("Smart Segments Failed", f"An error occurred: {e}")
            
    def _show_smart_segments_selector(self, document: Document, layer: Node, all_masks: List[Dict], session_id: str):
        """Show the Smart Segments selector window with all masks"""
        try:
            from .ui.smart_segments_dialog import SmartSegmentsDialog
            
            dialog = SmartSegmentsDialog(self, document, layer, all_masks, session_id)
            dialog.exec_()
            
        except ImportError as e:
            self.logger.error(f"Failed to import SmartSegmentsDialog: {e}")
            self._show_error_dialog(
                "Dialog Import Error", 
                f"Could not load the Smart Segments dialog.\n\nError: {e}"
            )
        except Exception as e:
            self.logger.error(f"Error showing Smart Segments selector: {e}")
            self._show_error_dialog(
                "Smart Segments Dialog Error", 
                f"An error occurred while showing the dialog:\n\n{e}"
            )
    
    def show_smart_segments_dialog(self):
        """Show the main Smart Segments dialog with interactive overlay"""
        try:
            # Check if plugin is initialized first (before trying to load dependencies)
            if not self._initialized:
                # Check if environment is already set up
                if self._is_setup_completed():
                    # Environment is ready, just initialize
                    self.logger.info("Environment ready, initializing plugin...")
                    self._initialize_async()
                    # Show initialization dialog while loading
                    self._show_initialization_dialog()
                    return
                
                # Ask user if they want to start setup
                msg = QMessageBox()
                msg.setWindowTitle("Smart Segments - Setup Required")
                msg.setText("Welcome to Smart Segments!")
                msg.setInformativeText(
                    "This is your first time using Smart Segments.\n\n"
                    "The plugin needs to:\n"
                    "• Download AI models (~1 GB)\n"
                    "• Install PyTorch and dependencies\n"
                    "• Set up the segmentation environment\n\n"
                    "This process may take 10-15 minutes with internet connection.\n\n"
                    "Would you like to start the setup now?"
                )
                msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
                msg.setDefaultButton(QMessageBox.Yes)
                
                if msg.exec_() == QMessageBox.Yes:
                    # Use minimal bootstrap for initial setup
                    try:
                        MinimalBootstrap = self._lazy_load_dependencies(minimal_only=True)
                        self._run_minimal_setup(MinimalBootstrap)
                    except Exception as dep_error:
                        self.logger.error(f"Failed to load minimal bootstrap: {dep_error}")
                        self._show_error_dialog(
                            "Setup Dependencies Missing",
                            "Cannot start setup because minimal bootstrap is missing.\n\n"
                            "This usually means the plugin installation is incomplete.\n"
                            "Please reinstall the plugin or contact support."
                        )
                return
                
            # If initialized, try to load dependencies and show main dialog
            try:
                self._lazy_load_dependencies()
                if SmartSegmentsDialog:
                    dialog = SmartSegmentsDialog(self)
                    dialog.exec_()
                else:
                    self._show_error_dialog("Dialog Error", "Main dialog not available")
            except Exception as dep_error:
                self.logger.error(f"Failed to load dependencies: {dep_error}")
                self._show_error_dialog(
                    "Dependencies Missing", 
                    f"Failed to load required dependencies:\n{dep_error}\n\n"
                    "The plugin may need to be reinstalled."
                )
            
        except Exception as e:
            self.logger.error(f"Failed to show main dialog: {e}")
            self._show_error_dialog("Failed to open Smart Segments dialog", str(e))
    def quick_segment_current_layer(self):
        """Quick segmentation of current layer"""
        try:
            # Check if we have a document and layer first
            doc = Krita.instance().activeDocument()
            if not doc:
                self._show_error_dialog("No Document", "Please open a document first.")
                return
                
            active_node = doc.activeNode()
            if not active_node:
                self._show_error_dialog("No Layer", "Please select a layer first.")
                return
            
            # Additional layer validation
            layer_info = {
                'name': active_node.name(),
                'type': active_node.type(),
                'visible': active_node.visible(),
                'bounds': active_node.bounds()
            }
            self.logger.info(f"Layer validation - Info: {layer_info}")
            
            # Check if layer has valid bounds
            bounds = active_node.bounds()
            if bounds.width() <= 0 or bounds.height() <= 0:
                self._show_error_dialog(
                    "Invalid Layer", 
                    f"The selected layer '{active_node.name()}' has invalid dimensions ({bounds.width()}x{bounds.height()}). Please select a layer with content."
                )
                return
            
            # If not initialized, check if setup is completed first
            if not self._initialized:
                # Check if environment is already set up
                if self._is_setup_completed():
                    # Environment is ready, just initialize
                    self.logger.info("Quick Segment: Environment ready, initializing plugin...")
                    self._initialize_async()
                    
                    # Check if initialization completed synchronously
                    if self._initialized:
                        self._perform_quick_segmentation(doc, active_node)
                    # Otherwise, rely on the async initializer to show progress/errors
                    return
                
                # Setup is not completed, ask user
                msg = QMessageBox()
                msg.setWindowTitle("Smart Segments - Quick Segment")
                msg.setText("Smart Segments needs to be set up first.")
                msg.setInformativeText(
                    "Quick Segment requires the AI models to be downloaded.\n\n"
                    "Would you like to run the setup now?"
                )
                msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
                msg.setDefaultButton(QMessageBox.Yes)
                
                if msg.exec_() == QMessageBox.Yes:
                    # Use minimal bootstrap for initial setup
                    try:
                        MinimalBootstrap = self._lazy_load_dependencies(minimal_only=True)
                        self._run_minimal_setup(MinimalBootstrap)
                    except Exception as dep_error:
                        self.logger.error(f"Failed to load minimal bootstrap: {dep_error}")
                        self._show_error_dialog(
                            "Setup Dependencies Missing",
                            "Cannot start setup because minimal bootstrap is missing.\n\n"
                            "This usually means the plugin installation is incomplete.\n"
                            "Please reinstall the plugin or contact support."
                        )
                return
                
            # If initialized, perform quick segmentation
            self._perform_quick_segmentation(doc, active_node)
            
        except Exception as e:
            self.logger.error(f"Quick segment failed: {e}")
            self._show_error_dialog("Quick Segment Failed", str(e))
            
    def show_settings_dialog(self):
        """Show plugin settings dialog"""
        try:
            # Show basic settings info for now
            info = self.get_plugin_info()
            
            msg = QMessageBox()
            msg.setWindowTitle("Smart Segments Settings")
            msg.setText("Smart Segments Configuration")
            
            settings_text = f"""Plugin Status:
• Version: {info['version']}
• Initialized: {info['initialized']}
• Environment Ready: {info['environment_ready']}
• Active Session: {info['active_session'] or 'None'}

Current Configuration:
• Model: {info['config']['model_name']}
• Device: {info['config']['device'] or 'Auto-detect'}
• Max Sessions: {info['config']['max_sessions']}
• Auto Initialize: {info['config']['auto_initialize']}
• Use GPU: {info['config']['use_gpu_if_available']}

Full settings dialog will be available in the next update."""
            
            msg.setInformativeText(settings_text)
            msg.exec_()
            
        except Exception as e:
            self.logger.error(f"Failed to show settings dialog: {e}")
            self._show_error_dialog("Settings Error", str(e))
            
    def _show_initialization_dialog(self):
        """Show dialog indicating plugin is initializing"""
        msg = QMessageBox()
        msg.setWindowTitle("Smart Segments")
        msg.setText("Smart Segments is initializing...")
        msg.setInformativeText(
            "The AI models are being loaded. This may take a moment.\n"
            "Please wait and try again in a few seconds."
        )
        msg.exec_()
        
    def _run_minimal_setup(self, MinimalBootstrap):
        """Run minimal setup using bootstrap that doesn't require numpy"""
        try:
            self.logger.info("Starting minimal setup process...")
            
            # Show progress dialog
            progress = QProgressDialog("Setting up Smart Segments...", "Cancel", 0, 100)
            progress.setWindowTitle("Smart Segments Setup")
            progress.show()
            QApplication.processEvents()
            
            # Initialize minimal bootstrap
            progress.setValue(10)
            progress.setLabelText("Initializing setup...")
            QApplication.processEvents()
            
            # Use Krita's config directory for cross-platform compatibility
            project_root = self._get_krita_config_dir()
            self.logger.info(f"Creating MinimalBootstrap with project_root: {project_root}")
            bootstrap = MinimalBootstrap(project_root)
            self.logger.info(f"MinimalBootstrap created: {bootstrap}")
            self.logger.info(f"Bootstrap type: {type(bootstrap)}")
            
            # Check system requirements
            progress.setValue(20)
            progress.setLabelText("Checking system requirements...")
            QApplication.processEvents()
            
            system_info = bootstrap.check_system_requirements()
            self.logger.info(f"System check results: {system_info}")
            
            # Create virtual environment
            progress.setValue(40)
            progress.setLabelText("Creating virtual environment...")
            QApplication.processEvents()
            
            success = bootstrap.create_virtual_environment()
            if not success:
                raise Exception(
                    "Failed to create virtual environment.\n\n"
                    "Please ensure Python 3.10 is installed on your system:\n"
                    "• On Windows: Install Python 3.10 from python.org\n"
                    "• Make sure 'py' launcher is available (installed by default)\n"
                    "• Or install Python 3.10 via Microsoft Store\n\n"
                    "After installing Python 3.10, restart Krita and try again."
                )
            else:
                # Install dependencies in venv
                progress.setValue(60)
                progress.setLabelText("Installing dependencies (this may take several minutes)...")
                QApplication.processEvents()
                
                # Create progress callback
                def update_progress(value, label):
                    progress.setValue(value)
                    progress.setLabelText(label)
                    QApplication.processEvents()
                
                success = bootstrap.install_dependencies(progress_callback=update_progress)
                if not success:
                    raise Exception("Failed to install dependencies")
                
                # Download AI models
                progress.setValue(88)
                progress.setLabelText("Downloading AI models (this may take a while)...")
                QApplication.processEvents()
                success = bootstrap.download_models(progress_callback=update_progress)
                if not success:
                    raise Exception("Failed to download AI models")
                
                # Verify installation
                progress.setValue(95)
                progress.setLabelText("Verifying installation...")
                QApplication.processEvents()
                verification = bootstrap.verify_installation()
                if not verification:
                    raise Exception("Installation verification failed")
            
            progress.setValue(100)
            progress.close()
            
            # Show success message
            msg = QMessageBox()
            msg.setWindowTitle("Setup Complete")
            msg.setText("Smart Segments setup completed successfully!")
            msg.setInformativeText(
                "The plugin is now ready to use.\n\n"
                "You can now:\n"
                "• Use Smart Segments dialog for interactive segmentation\n"
                "• Try Quick Segment for fast processing\n"
                "• Configure settings as needed\n\n"
                "Click Smart Segments again to start using the plugin."
            )
            msg.exec_()
            
            # Mark environment as ready and try to initialize
            self._environment_ready = True
            self._save_plugin_state()
            QTimer.singleShot(1000, self._initialize_async)
            
        except Exception as e:
            if 'progress' in locals():
                progress.close()
                
            error_msg = f"Setup failed: {e}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            
            # Show detailed error message
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setWindowTitle("Setup Failed")
            msg.setText("Smart Segments setup failed.")
            
            # Check if it's a Python installation issue
            if "Failed to create virtual environment" in str(e):
                msg.setInformativeText(
                    "Python 3.10 is required but not found.\n\n"
                    "To fix this issue:\n"
                    "1. Download Python 3.10 from https://python.org\n"
                    "2. During installation, check 'Add Python to PATH'\n"
                    "3. Complete the installation\n"
                    "4. Restart Krita\n"
                    "5. Try Smart Segments again\n\n"
                    "Alternative: Install Python 3.10 from Microsoft Store"
                )
            else:
                msg.setInformativeText(
                    "The automatic setup process encountered an error.\n\n"
                    "This could be due to:\n"
                    "• Network connectivity issues\n"
                    "• Insufficient disk space\n"
                    "• Permission problems\n"
                    "• System compatibility issues\n\n"
                    "Please check the logs for more details."
                )
            msg.setDetailedText(str(e))
            msg.exec_()
    
    def _show_error_dialog(self, title: str, message: str):
        """Show error dialog"""
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setWindowTitle(title)
        msg.setText(message)
        msg.exec_()
    
    def _show_cuda_error_dialog(self, error_message: str):
        """Show specialized dialog for CUDA-related errors with helpful guidance"""
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("Smart Segments - GPU Compatibility Issue")
        
        if "no kernel image is available" in error_message.lower():
            msg.setText("GPU Compatibility Issue Detected")
            msg.setInformativeText(
                "Smart Segments encountered a CUDA compatibility error.\n\n"
                "The plugin has automatically switched to CPU mode, which will work but may be slower.\n\n"
                "To resolve this GPU issue:\n"
                "• Update your GPU drivers\n"
                "• Ensure PyTorch was installed with the correct CUDA version\n"
                "• Check if your GPU is supported\n\n"
                "The plugin will continue working in CPU mode."
            )
        elif "cuda" in error_message.lower() and "out of memory" in error_message.lower():
            msg.setText("GPU Memory Issue")
            msg.setInformativeText(
                "Your GPU doesn't have enough memory to run Smart Segments.\n\n"
                "The plugin has switched to CPU mode.\n\n"
                "To use GPU mode:\n"
                "• Close other GPU-intensive applications\n"
                "• Consider using a smaller model variant\n"
                "• Use CPU mode for now (will be slower but functional)"
            )
        else:
            msg.setText("GPU Error Detected")
            msg.setInformativeText(
                "Smart Segments encountered a GPU-related error.\n\n"
                "The plugin has automatically switched to CPU mode.\n\n"
                "This may result in slower performance but full functionality is maintained."
            )
        
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()
        
    # Public API methods for UI components
    
    def is_initialized(self) -> bool:
        """Check if plugin is fully initialized"""
        return self._initialized
        
    def get_segmentation_api(self) -> Optional[SegmentationAPI]:
        """Get the segmentation API instance"""
        return self._segmentation_api
        
    def get_krita_bridge(self) -> Optional[KritaImageBridge]:
        """Get the Krita bridge instance"""
        return self._krita_bridge
        
    def create_session_for_document(self, document: Document, layer: Node) -> Optional[str]:
        """Create segmentation session for a document/layer"""
        try:
            if not self._initialized or not self._segmentation_api:
                self.logger.error("Plugin not initialized")
                return None
                
            # Convert Krita layer to numpy array
            image_array = self._krita_bridge.layer_to_numpy(layer)
            if image_array is None:
                self.logger.error("Failed to convert layer to numpy array")
                return None
                
            # Create segmentation session
            session_id = self._segmentation_api.create_session(image_array)
            
            # Store current context
            self._current_document = document
            self._active_session_id = session_id
            
            self.logger.info(f"Created session {session_id} for document {document.name()}")
            return session_id
            
        except Exception as e:
            self.logger.error(f"Failed to create session: {e}")
            return None
            
    def apply_mask_to_layer(self, mask, target_layer: Node, create_new_layer: bool = True) -> bool:
        """Apply segmentation mask to a layer"""
        try:
            if not self._krita_bridge:
                return False
                
            return self._krita_bridge.apply_mask_to_layer(mask, target_layer, create_new_layer)
            
        except Exception as e:
            self.logger.error(f"Failed to apply mask to layer: {e}")
            return False
            
    def get_plugin_info(self) -> Dict[str, Any]:
        """Get plugin information and status"""
        info = {
            'name': 'Smart Segments',
            'version': '1.0.2',
            'initialized': self._initialized,
            'environment_ready': self._environment_ready,
            'active_session': self._active_session_id,
            'config': self.config.copy()
        }
        
        if self._segmentation_api:
            info['api_stats'] = self._segmentation_api.get_api_stats()
            
        return info
        
    def shutdown(self):
        """Cleanup plugin resources"""
        try:
            self.logger.info("Shutting down Smart Segments extension...")
            
            if self._segmentation_api:
                self._segmentation_api.shutdown()
                
            self._initialized = False
            self._segmentation_api = None
            self._krita_bridge = None
            self._current_document = None
            self._active_session_id = None
            
            self.logger.info("Smart Segments extension shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
            
    def _perform_quick_segmentation(self, document: Document, layer: Node):
        """Perform quick segmentation on the given layer"""
        try:
            self.logger.info(f"Starting quick segmentation for layer: {layer.name()}")
            
            # Load dependencies
            self.logger.info("Loading dependencies...")
            self._lazy_load_dependencies()
            self.logger.info("Dependencies loaded successfully")
            
            # Show progress dialog
            progress = QProgressDialog("Analyzing image...", "Cancel", 0, 100)
            progress.setWindowTitle("Quick Segment")
            progress.show()
            QApplication.processEvents()
            
            # Initialize Krita bridge if not done
            if not self._krita_bridge:
                self.logger.info("Initializing Krita bridge...")
                self._krita_bridge = KritaImageBridge()
                self.logger.info("Krita bridge initialized")
            
            # Convert layer to numpy array
            progress.setValue(20)
            progress.setLabelText("Converting layer data...")
            QApplication.processEvents()
            
            self.logger.info(f"Converting layer '{layer.name()}' to numpy array...")
            image_array = self._krita_bridge.layer_to_numpy(layer)
            if image_array is None:
                progress.close()
                self.logger.error("Layer conversion failed - image_array is None")
                self._show_error_dialog("Layer Conversion Failed", "Could not convert layer to image data.")
                return
            
            self.logger.info(f"Layer converted successfully: shape={image_array.shape}, dtype={image_array.dtype}")
            
            # Create segmentation session
            progress.setValue(40)
            progress.setLabelText("Setting up AI model...")
            QApplication.processEvents()
            
            self.logger.info(f"Creating segmentation session with image shape: {image_array.shape}")
            if not self._segmentation_api:
                progress.close()
                self.logger.error("Segmentation API is None - initialization may have failed")
                self._show_error_dialog("Initialization Error", "Segmentation API not available. Try restarting Krita.")
                return
                
            session_id = self._segmentation_api.create_session(image_array)
            if not session_id:
                progress.close()
                self.logger.error("Session creation failed - session_id is None")
                self._show_error_dialog("Session Creation Failed", "Could not create segmentation session.")
                return
            
            self.logger.info(f"Segmentation session created successfully: {session_id}")
            
            # Generate all masks
            progress.setValue(60)
            progress.setLabelText("Generating segmentation masks...")
            QApplication.processEvents()
            
            self.logger.info(f"Starting segment_everything for session: {session_id}")
            all_masks = self._segmentation_api.segment_everything(session_id)
            if not all_masks:
                progress.close()
                self.logger.error("Segmentation failed - no masks generated")
                self._show_error_dialog("Segmentation Failed", "Could not generate segmentation masks.")
                return
            
            self.logger.info(f"Segmentation completed successfully - generated {len(all_masks)} masks")
            
            progress.setValue(100)
            progress.close()
            
            # Show quick segment selector dialog
            self._show_quick_segment_selector(document, layer, all_masks, session_id)
            
        except Exception as e:
            if 'progress' in locals():
                progress.close()
            self.logger.error(f"Quick segmentation failed: {e}")
            self._show_error_dialog("Quick Segmentation Failed", f"An error occurred: {e}")
            
    def _show_quick_segment_selector(self, document: Document, layer: Node, all_masks: List[Dict], session_id: str):
        """Show the quick segment selector dialog with all masks"""
        try:
            from .ui.smart_segments_dialog import SmartSegmentsDialog
            
            dialog = SmartSegmentsDialog(self, document, layer, all_masks, session_id)
            dialog.exec_()
            
        except ImportError as e:
            self.logger.error(f"Failed to import SmartSegmentsDialog: {e}")
            self._show_error_dialog(
                "Dialog Import Error", 
                f"Could not load the Quick Segment dialog.\n\nError: {e}"
            )
        except Exception as e:
            self.logger.error(f"Error showing quick segment selector: {e}")
            self._show_error_dialog(
                "Quick Segment Dialog Error", 
                f"An error occurred while showing the dialog:\n\n{e}"
            )

    # State management methods
    
    def _load_plugin_state(self):
        """Load persistent plugin state from file."""
        try:
            if self.state_file.exists():
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                
                # Check if setup was completed successfully
                if state.get('setup_completed', False):
                    self._environment_ready = True
                    self.logger.info("Found existing setup completion state")
                
                # Load other configuration if needed
                if 'config' in state:
                    for key, value in state['config'].items():
                        if key in self.config:
                            self.config[key] = value
            
        except Exception as e:
            self.logger.warning(f"Failed to load plugin state: {e}")
            # If state file is corrupted, start fresh
            self._environment_ready = False
    
    def _save_plugin_state(self):
        """Save persistent plugin state to file."""
        try:
            state = {
                'setup_completed': self._environment_ready,
                'last_updated': str(self.state_file.stat().st_mtime) if self.state_file.exists() else None,
                'config': self.config.copy()
            }
            
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
                
            self.logger.debug("Plugin state saved successfully")
            
        except Exception as e:
            self.logger.warning(f"Failed to save plugin state: {e}")
    
    def _is_setup_completed(self) -> bool:
        """Check if setup was previously completed successfully."""
        # Simple check - just use the in-memory state that was loaded from file
        return self._environment_ready
    
