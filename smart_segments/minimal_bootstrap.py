"""
Minimal Bootstrap Module for Smart Segments
This module handles the initial setup without requiring numpy or other heavy dependencies
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List


class MinimalBootstrap:
    """Minimal bootstrap for initial plugin setup"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.logger = logging.getLogger("smart_segments_minimal_bootstrap")
        
        # Plugin directory contains requirements and other bundled resources
        self.plugin_root = Path(__file__).resolve().parent
        
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
            import shutil
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
                nvidia_smi_path = Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "NVIDIA Corporation" / "NVSMI" / "nvidia-smi.exe"
                if not nvidia_smi_path.exists():
                    return False
                command = [str(nvidia_smi_path)]
            else: # Linux/macOS
                command = ["nvidia-smi"]
            
            # Run nvidia-smi
            result = subprocess.run(command, capture_output=True, text=True, timeout=10)
            return result.returncode == 0
            
        except (subprocess.SubprocessError, FileNotFoundError):
            return False
    
    def _can_create_virtualenv(self) -> bool:
        """Check if we can create virtual environments"""
        try:
            result = subprocess.run([sys.executable, '-m', 'venv', '--help'], 
                                  capture_output=True, timeout=10)
            return result.returncode == 0
        except:
            return False
    
    def create_virtual_environment(self) -> bool:
        """Create a virtual environment if it doesn't exist or is incomplete"""
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
                    # Test if the virtual environment works
                    try:
                        result = subprocess.run([
                            str(python_path), '-c', 'import sys; print(sys.version)'
                        ], capture_output=True, text=True, timeout=30)
                        
                        if result.returncode == 0:
                            self.logger.info("Existing virtual environment is valid")
                            return True
                        else:
                            self.logger.warning(f"Virtual environment test failed: {result.stderr}")
                    except Exception as e:
                        self.logger.warning(f"Virtual environment test error: {e}")
            
            self.logger.info(f"Creating new virtual environment at {self.venv_path}")
            
            # Remove existing venv if it exists but is invalid
            if self.venv_path.exists():
                import shutil
                shutil.rmtree(self.venv_path)
            
            # Create new virtual environment
            result = subprocess.run([
                sys.executable, '-m', 'venv', str(self.venv_path)
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                self.logger.error(f"Failed to create venv: {result.stderr}")
                return False
                
            self.logger.info("Virtual environment created successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create virtual environment: {e}")
            return False
    
    def _check_dependencies_installed(self) -> Dict[str, bool]:
        """Check which dependencies are already installed"""
        try:
            # Get python path in virtual environment
            if os.name == 'nt':  # Windows
                python_path = self.venv_path / "Scripts" / "python"
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
    
    def install_dependencies(self) -> bool:
        """Install basic dependencies if not already present"""
        try:
            # Get pip path in virtual environment
            if os.name == 'nt':  # Windows
                pip_path = self.venv_path / "Scripts" / "pip"
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
            
            # Install basic dependencies only if not already installed
            basic_deps = [
                'numpy>=1.21.0',
                'opencv-python>=4.5.0',
                'Pillow>=8.0.0',
                'requests>=2.25.0',
                'tqdm>=4.60.0'
            ]
            
            self.logger.info("Checking and installing basic dependencies...")
            for dep in basic_deps:
                if not installed_deps.get(dep, False):
                    self.logger.info(f"Installing {dep}...")
                    result = subprocess.run([
                        str(pip_path), 'install', dep
                    ], capture_output=True, text=True, timeout=600)
                    
                    if result.returncode != 0:
                        self.logger.error(f"Failed to install {dep}: {result.stderr}")
                        return False
                else:
                    self.logger.info(f"{dep} already installed, skipping")
            
            # Install PyTorch if not already installed
            if not installed_deps.get('torch>=1.9.0', False):
                # Detect CUDA availability
                has_cuda = self._detect_cuda()
                
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
            self.logger.info("Installing SAM2...")
            sam2_command = [
                str(pip_path), 'install',
                'git+https://github.com/facebookresearch/segment-anything-2.git'
            ]
            
            result = subprocess.run(sam2_command, capture_output=True, text=True, timeout=1800)
            
            if result.returncode != 0:
                self.logger.error(f"SAM2 install failed: {result.stderr}")
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
                python_path = self.venv_path / "Scripts" / "python"
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
    
    def download_models(self) -> bool:
        """Download required AI models"""
        try:
            # This is a placeholder - actual model downloading would be implemented here
            self.logger.info("Model downloading not implemented in minimal bootstrap")
            return True
            
        except Exception as e:
            self.logger.error(f"Model download failed: {e}")
            return False
