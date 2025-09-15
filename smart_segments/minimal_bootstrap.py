"""
Minimal Bootstrap Module for Smart Segments
This module handles the initial setup without requiring numpy or other heavy dependencies
"""

import os
import sys
import subprocess
import logging
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List

# Use lightweight platform utilities (no heavy deps) to locate Python on Windows
try:
    from .utils.platform_utils import PythonDetector
except Exception:
    PythonDetector = None


class MinimalBootstrap:
    """Minimal bootstrap for initial plugin setup"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        
        # Setup logging with same configuration as main plugin
        self.logger = logging.getLogger("smart_segments_minimal_bootstrap")
        self.logger.setLevel(logging.DEBUG)
        
        # Add handlers if not already configured
        if not self.logger.handlers:
            # Use the same log file as the main plugin
            log_dir = Path(__file__).parent / "logs"
            log_dir.mkdir(exist_ok=True)
            
            log_file = log_dir / "smart_segments.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            
            self.logger.addHandler(file_handler)
        
        # Plugin directory contains requirements and other bundled resources
        self.plugin_root = Path(__file__).resolve().parent
        
        # Target Python tag to match Krita's embedded interpreter (major.minor)
        import sys as _sys
        self.target_py_tag = f"{_sys.version_info.major}.{_sys.version_info.minor}"
        
        # Setup paths
        self.venv_path = project_root / "venv"
        self.requirements_file = self.plugin_root / "requirements.txt"
        
    def check_system_requirements(self) -> Dict[str, Any]:
        """Basic system requirements check"""
        try:
            import platform
            
            system_info = {
                'platform': platform.system(),
                'python_version': platform.python_version(),
                'python_executable': sys.executable,
                'architecture': platform.machine(),
                'has_internet': self._check_internet_connection(),
                'disk_space_gb': self._get_available_disk_space(),
                'can_create_venv': self._can_create_virtualenv(),
                'has_cuda': self._detect_cuda()
            }
            
            self.logger.info(f"System check completed: {system_info}")
            return system_info
            
        except Exception as e:
            self.logger.error(f"System check failed: {e}")
            return {'error': str(e)}
    
    def _check_internet_connection(self) -> bool:
        """Check if internet connection is available"""
        try:
            import socket
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            return True
        except:
            return False
    
    def _get_available_disk_space(self) -> float:
        """Get available disk space in GB"""
        try:
            total, used, free = shutil.disk_usage(str(self.project_root))
            return free / (1024**3)  # Convert to GB
        except:
            return 0.0

    def _detect_cuda(self) -> bool:
        """Detect if CUDA is available in a cross-platform way"""
        try:
            import platform
            
            # Get nvidia-smi path (cross-platform)
            if platform.system() == "Windows":
                # Try multiple possible locations for nvidia-smi.exe
                possible_paths = [
                    Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "NVIDIA Corporation" / "NVSMI" / "nvidia-smi.exe",
                    Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)")) / "NVIDIA Corporation" / "NVSMI" / "nvidia-smi.exe",
                    Path("C:/Windows/System32/nvidia-smi.exe"),  # Sometimes in System32
                ]
                
                nvidia_smi_path = None
                for path in possible_paths:
                    if path.exists():
                        nvidia_smi_path = path
                        break
                
                if nvidia_smi_path is None:
                    # Try using PATH as fallback
                    command = ["nvidia-smi"]
                else:
                    command = [str(nvidia_smi_path)]
            else: # Linux/macOS
                command = ["nvidia-smi"]
            
            # Run nvidia-smi
            result = subprocess.run(command, capture_output=True, text=True, timeout=10)
            return result.returncode == 0
            
        except (subprocess.SubprocessError, FileNotFoundError):
            return False
    
    def _can_create_virtualenv(self) -> bool:
        """Check if we can create virtual environments (prefer real Python on Windows)."""
        try:
            cmds = []
            # Prefer Windows launcher when available
            if os.name == 'nt' and shutil.which('py'):
                # Try to match Krita's embedded Python version (major.minor)
                cmds.append(['py', f'-{self.target_py_tag}', '-m', 'venv', '--help'])
                cmds.append(['py', '-3', '-m', 'venv', '--help'])
            
            # Try a detected system Python (>=3.8)
            if PythonDetector is not None:
                try:
                    reco = PythonDetector().get_recommended_python(min_version=(3, 8))
                    if reco and reco.get('executable'):
                        cmds.append([reco['executable'], '-m', 'venv', '--help'])
                except Exception:
                    pass
            
            # Avoid using Krita.exe as Python
            exe_name = Path(sys.executable).name.lower()
            if 'krita' not in exe_name:
                cmds.append([sys.executable, '-m', 'venv', '--help'])
            
            # Fallbacks
            if os.name == 'nt' and shutil.which('python'):
                cmds.append(['python', '-m', 'venv', '--help'])
            if shutil.which('python3'):
                cmds.append(['python3', '-m', 'venv', '--help'])
            if shutil.which('virtualenv'):
                cmds.append(['virtualenv', '--help'])
            
            for cmd in cmds:
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                    if result.returncode == 0:
                        return True
                except Exception:
                    continue
            return False
        except Exception:
            return False
    
    def create_virtual_environment(self) -> bool:
        """Create a virtual environment if it doesn't exist or is incomplete"""
        self.logger.info(f"create_virtual_environment called with venv_path: {self.venv_path}")
        try:
            # Check if virtual environment already exists and is valid
            if self.venv_path.exists():
                self.logger.info(f"Virtual environment exists at {self.venv_path}, checking validity...")
                
                # Check if python executable exists
                if os.name == 'nt':  # Windows
                    python_path = self.venv_path / "Scripts" / "python.exe"
                    pip_path = self.venv_path / "Scripts" / "pip.exe"
                else:  # Unix/Linux/macOS
                    python_path = self.venv_path / "bin" / "python"
                    pip_path = self.venv_path / "bin" / "pip"
                
                if python_path.exists() and pip_path.exists():
                    # Test if the virtual environment works and matches Krita's Python version
                    try:
                        result = subprocess.run([
                            str(python_path), '-c', 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")'
                        ], capture_output=True, text=True, timeout=30)
                        
                        if result.returncode == 0:
                            venv_tag = result.stdout.strip()
                            if venv_tag != self.target_py_tag:
                                self.logger.warning(
                                    f"Existing venv Python {venv_tag} does not match Krita Python {self.target_py_tag}. "
                                    "Recreating venv to avoid binary incompatibilities (e.g., NumPy wheels)."
                                )
                            else:
                                self.logger.info("Existing virtual environment is valid and version-matched")
                                return True
                        else:
                            self.logger.warning(f"Virtual environment test failed: {result.stderr}")
                    except Exception as e:
                        self.logger.warning(f"Virtual environment test error: {e}")
                # If we reach here, remove and recreate
                shutil.rmtree(self.venv_path)
            
            self.logger.info(f"Creating new virtual environment at {self.venv_path}")
            
            # Build a list of creation commands (Windows first preferences)
            creation_cmds = []
            if os.name == 'nt' and shutil.which('py'):
                # Prefer exact major.minor to match Krita's Python (e.g., 3.10)
                creation_cmds.append(['py', f'-{self.target_py_tag}', '-m', 'venv', str(self.venv_path)])
                # Fallback to default major (may cause ABI mismatch)
                creation_cmds.append(['py', '-3', '-m', 'venv', str(self.venv_path)])
            
            if PythonDetector is not None:
                try:
                    reco = PythonDetector().get_recommended_python(min_version=(3, 8))
                    if reco and reco.get('executable'):
                        creation_cmds.append([reco['executable'], '-m', 'venv', str(self.venv_path)])
                except Exception:
                    pass
            
            exe_name = Path(sys.executable).name.lower()
            if 'krita' not in exe_name:
                creation_cmds.append([sys.executable, '-m', 'venv', str(self.venv_path)])
            
            if os.name == 'nt' and shutil.which('python'):
                creation_cmds.append(['python', '-m', 'venv', str(self.venv_path)])
            if shutil.which('python3'):
                creation_cmds.append(['python3', '-m', 'venv', str(self.venv_path)])
            if shutil.which('virtualenv'):
                creation_cmds.append(['virtualenv', str(self.venv_path)])
            
            last_err = None
            for cmd in creation_cmds:
                try:
                    self.logger.info(f"Trying venv creation with: {' '.join(map(str, cmd))}")
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                    if result.returncode == 0:
                        # Verify the venv Python version matches Krita's major.minor
                        py_exe = self.venv_path / ('Scripts' if os.name == 'nt' else 'bin') / ('python.exe' if os.name == 'nt' else 'python')
                        ver_check = subprocess.run([str(py_exe), '-c', 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")'], capture_output=True, text=True, timeout=20)
                        if ver_check.returncode == 0:
                            venv_tag = ver_check.stdout.strip()
                            if venv_tag != self.target_py_tag:
                                self.logger.warning(f"Venv Python {venv_tag} != Krita Python {self.target_py_tag}. This may break binary packages. Trying next candidate if available...")
                                # If we have more candidates, continue loop to try another
                                last_err = f"Version mismatch venv={venv_tag}, krita={self.target_py_tag}"
                                continue
                        self.logger.info("Virtual environment created successfully")
                        return True
                    else:
                        last_err = result.stderr or result.stdout
                        self.logger.warning(f"Venv creation failed with {cmd[0]}: {last_err}")
                except Exception as e:
                    last_err = str(e)
                    self.logger.warning(f"Venv creation error with {cmd[0]}: {e}")
            
            self.logger.error(f"Failed to create venv after trying multiple interpreters. Last error: {last_err}")
            self.logger.error("No suitable Python 3.10 interpreter found. Please install Python 3.10 from python.org")
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to create virtual environment: {e}")
            return False
    
    def _check_dependencies_installed(self) -> Dict[str, bool]:
        """Check which dependencies are already installed"""
        try:
            # Get python path in virtual environment
            if os.name == 'nt':  # Windows
                python_path = self.venv_path / "Scripts" / "python.exe"
            else:  # Unix/Linux/macOS
                python_path = self.venv_path / "bin" / "python"
            
            if not python_path.exists():
                return {}
            
            # Test imports for basic dependencies
            test_modules = {
                'numpy': 'numpy>=1.21.0',
                'cv2': 'opencv-python>=4.5.0', 
                'PIL': 'Pillow>=8.0.0',
                'requests': 'requests>=2.25.0',
                'tqdm': 'tqdm>=4.60.0',
                'torch': 'torch>=1.9.0'
            }
            
            installed = {}
            for module, package in test_modules.items():
                try:
                    result = subprocess.run([
                        str(python_path), '-c', f'import {module}; print("OK")'
                    ], capture_output=True, text=True, timeout=10)
                    installed[package] = result.returncode == 0
                except:
                    installed[package] = False
            
            return installed
            
        except Exception as e:
            self.logger.error(f"Dependency check failed: {e}")
            return {}
    
    def install_dependencies(self, progress_callback=None) -> bool:
        """Install basic dependencies if not already present
        
        Args:
            progress_callback: Optional callback function(value, label) to update progress
        """
        try:
            # Get pip path in virtual environment
            if os.name == 'nt':  # Windows
                pip_path = self.venv_path / "Scripts" / "pip.exe"
            else:  # Unix/Linux/macOS
                pip_path = self.venv_path / "bin" / "pip"
            
            if not pip_path.exists():
                self.logger.error(f"Pip not found at {pip_path}")
                return False
            
            # Check what's already installed
            installed_deps = self._check_dependencies_installed()
            self.logger.info(f"Currently installed dependencies: {installed_deps}")
            
            # Upgrade pip first
            self.logger.info("Upgrading pip...")
            result = subprocess.run([
                str(pip_path), 'install', '--upgrade', 'pip'
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                self.logger.warning(f"Pip upgrade failed: {result.stderr}")
            
            # Install basic dependencies
            basic_deps = [
                ('numpy<2', 'NumPy (numerical computing)'),
                ('opencv-python>=4.5.0', 'OpenCV (image processing)'),
                ('Pillow>=8.0.0', 'Pillow (image I/O)'),
                ('requests>=2.25.0', 'Requests (HTTP library)'),
                ('tqdm>=4.60.0', 'TQDM (progress bars)')
            ]
            
            self.logger.info("Installing core dependencies...")
            # Progress from 60% to 75% for basic deps (3% per package)
            base_progress = 60
            for i, (dep, desc) in enumerate(basic_deps):
                progress = base_progress + (i * 3)
                if progress_callback:
                    progress_callback(progress, f"Installing {desc}...")
                
                self.logger.info(f"Installing {dep}...")
                result = subprocess.run([
                    str(pip_path), 'install', dep
                ], capture_output=True, text=True, timeout=1200)
                
                if result.returncode != 0:
                    self.logger.error(f"Failed to install {dep}: {result.stderr or result.stdout}")
                    return False
            
            # Install PyTorch if not already installed
            if not installed_deps.get('torch>=1.9.0', False):
                # Detect CUDA availability
                has_cuda = self._detect_cuda()
                
                if progress_callback:
                    progress_callback(75, f"Installing PyTorch {'(with CUDA)' if has_cuda else '(CPU-only)'}...")
                
                if has_cuda:
                    self.logger.info("CUDA detected - Installing PyTorch with CUDA support...")
                    torch_command = [
                        str(pip_path), 'install', 
                        'torch', 
                        'torchvision',
                        '--index-url', 'https://download.pytorch.org/whl/cu121'  # CUDA 12.1
                    ]
                    
                    result = subprocess.run(torch_command, capture_output=True, text=True, timeout=1800)
                    
                    if result.returncode != 0:
                        self.logger.warning(f"PyTorch CUDA install failed, trying CPU version: {result.stderr}")
                        # Fallback to CPU version
                        torch_command = [
                            str(pip_path), 'install', 
                            'torch', 
                            'torchvision',
                            '--index-url', 'https://download.pytorch.org/whl/cpu'
                        ]
                        result = subprocess.run(torch_command, capture_output=True, text=True, timeout=1800)
                else:
                    self.logger.info("No CUDA detected - Installing PyTorch CPU version...")
                    torch_command = [
                        str(pip_path), 'install', 
                        'torch', 
                        'torchvision',
                        '--index-url', 'https://download.pytorch.org/whl/cpu'
                    ]
                    result = subprocess.run(torch_command, capture_output=True, text=True, timeout=1800)
                
                if result.returncode != 0:
                    self.logger.error(f"PyTorch install failed: {result.stderr}")
                    return False
                else:
                    if has_cuda:
                        self.logger.info("PyTorch installed with CUDA support")
                    else:
                        self.logger.info("PyTorch installed (CPU-only)")
            else:
                self.logger.info("PyTorch already installed, skipping")
            
            # Install SAM2 from GitHub
            if progress_callback:
                progress_callback(85, "Installing SAM2 (AI segmentation model)...")
            
            self.logger.info("Installing SAM2...")
            sam2_command = [
                str(pip_path), 'install',
                'git+https://github.com/facebookresearch/segment-anything-2.git'
            ]
            
            result = subprocess.run(sam2_command, capture_output=True, text=True, timeout=3600)
            
            if result.returncode != 0:
                self.logger.error(f"SAM2 install failed: {result.stderr or result.stdout}")
                return False
            else:
                self.logger.info("SAM2 installed successfully")
            
            self.logger.info("Dependencies installed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to install dependencies: {e}")
            return False

    
    def verify_installation(self) -> Dict[str, bool]:
        """Verify that basic dependencies are installed"""
        try:
            # Get python path in virtual environment
            if os.name == 'nt':  # Windows
                python_path = self.venv_path / "Scripts" / "python.exe"
            else:  # Unix/Linux/macOS
                python_path = self.venv_path / "bin" / "python"
            
            if not python_path.exists():
                return {'venv': False}
            
            # Test imports
            test_imports = ['numpy', 'torch', 'cv2', 'PIL', 'sam2']
            results = {'venv': True}
            
            for module in test_imports:
                try:
                    result = subprocess.run([
                        str(python_path), '-c', f'import {module}; print("OK")'
                    ], capture_output=True, text=True, timeout=30)
                    
                    results[module.replace('cv2', 'opencv')] = result.returncode == 0
                except:
                    results[module.replace('cv2', 'opencv')] = False
            
            self.logger.info(f"Verification results: {results}")
            return results
            
        except Exception as e:
            self.logger.error(f"Verification failed: {e}")
            return {'error': str(e)}
    
    def download_models(self, progress_callback=None) -> bool:
        """Download required AI models
        
        Args:
            progress_callback: Optional callback function(value, label) to update progress
        """
        try:
            import urllib.request
            import hashlib
            
            # Define model info
            models = [
                {
                    'name': 'sam2_hiera_large.pt',
                    'url': 'https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt',
                    'size_mb': 897,  # Approximate size in MB
                    'sha256': None  # Add hash if known for verification
                }
            ]
            
            # Create models directory
            models_dir = self.project_root / "models"
            models_dir.mkdir(exist_ok=True)
            
            self.logger.info(f"Downloading models to {models_dir}")
            
            for i, model_info in enumerate(models):
                model_path = models_dir / model_info['name']
                
                # Check if model already exists
                if model_path.exists():
                    self.logger.info(f"Model {model_info['name']} already exists, skipping download")
                    continue
                
                self.logger.info(f"Downloading {model_info['name']} (~{model_info['size_mb']} MB)...")
                
                if progress_callback:
                    progress_callback(88 + (i * 5), f"Downloading {model_info['name']} (~{model_info['size_mb']} MB)...")
                
                # Download with progress tracking
                try:
                    def download_hook(block_num, block_size, total_size):
                        downloaded = block_num * block_size
                        percent = min(int(downloaded * 100 / total_size), 100) if total_size > 0 else 0
                        if progress_callback and percent % 10 == 0:  # Update every 10%
                            progress_callback(
                                88 + (i * 5) + (percent // 20),
                                f"Downloading {model_info['name']}: {percent}%"
                            )
                    
                    # Download to temp file first
                    temp_path = model_path.with_suffix('.tmp')
                    urllib.request.urlretrieve(model_info['url'], temp_path, reporthook=download_hook)
                    
                    # Rename temp file to final name
                    temp_path.rename(model_path)
                    self.logger.info(f"Successfully downloaded {model_info['name']}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to download {model_info['name']}: {e}")
                    # Clean up temp file if it exists
                    temp_path = model_path.with_suffix('.tmp')
                    if temp_path.exists():
                        temp_path.unlink()
                    return False
            
            self.logger.info("All models downloaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Model download failed: {e}")
            return False
