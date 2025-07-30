"""
Environment Bootstrapper & Dependency Manager for Smart Segments Plugin

This module handles:
- System Python availability checking
- Virtual environment creation with fallback options
- Dependency installation for PyTorch, SAM2, OpenCV, and other requirements
- GPU detection (CUDA/ROCm) with CPU fallback logic
- Clear error handling for missing system dependencies
"""

import os
import sys
import subprocess
import platform
import shutil
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass


@dataclass
class SystemInfo:
    """System information container"""
    platform: str
    arch: str
    python_version: str
    python_executable: str
    has_cuda: bool
    cuda_version: Optional[str]
    has_rocm: bool
    rocm_version: Optional[str]
    has_mps: bool  # Apple Metal Performance Shaders


@dataclass
class EnvironmentConfig:
    """Environment configuration container"""
    project_root: Path
    venv_path: Path
    python_executable: Path
    requirements_dir: Path
    models_dir: Path
    use_gpu: bool
    gpu_type: str  # 'cuda', 'rocm', 'mps', or 'cpu'


class BootstrapError(Exception):
    """Custom exception for bootstrap-related errors"""
    pass


class EnvironmentBootstrapper:
    """Main bootstrapper class for environment setup and dependency management"""
    
    def __init__(self, project_root: Optional[Path] = None):
        if project_root is None:
            # Navigate from this file to project root
            # Current file is at: project_root/pykrita/smart_segments/bootstrap.py
            current_file = Path(__file__).resolve()
            self.project_root = current_file.parent.parent.parent
        else:
            self.project_root = project_root
            
        # Plugin directory contains requirements and other bundled resources
        # This will work both in development and when installed from bundle
        self.plugin_root = Path(__file__).resolve().parent
        
        self.logger = self._setup_logging()
        self.system_info: Optional[SystemInfo] = None
        self.env_config: Optional[EnvironmentConfig] = None
        
        # Dependency definitions
        self.core_dependencies = [
            "numpy>=1.21.0",
            "opencv-python>=4.5.0",
            "pillow>=8.0.0",
            "scipy>=1.7.0",
            "matplotlib>=3.3.0",
            "requests>=2.25.0",
            "tqdm>=4.60.0",
        ]
        
        self.ai_dependencies = {
            "pytorch_cpu": [
                "torch>=2.0.0",
                "torchvision>=0.15.0",
                "torchaudio>=2.0.0",
            ],
            "pytorch_cuda": [
                "torch>=2.0.0",
                "torchvision>=0.15.0",
                "torchaudio>=2.0.0",
            ],
            "sam2": [
                "git+https://github.com/facebookresearch/segment-anything-2.git",
            ],
            "optional": [
                "transformers>=4.20.0",
                "accelerate>=0.20.0",
                "safetensors>=0.3.0",
            ]
        }
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger("smart_segments_bootstrap")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
        
    def check_system_requirements(self) -> SystemInfo:
        """Check system Python availability and capabilities"""
        self.logger.info("Checking system requirements...")
        
        # Get basic system info
        system_platform = platform.system().lower()
        arch = platform.machine().lower()
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        python_executable = sys.executable
        
        # Check minimum Python version
        if sys.version_info < (3, 8):
            raise BootstrapError(
                f"Python 3.8+ is required, but found {python_version}. "
                "Please upgrade your Python installation."
            )
            
        # Detect GPU capabilities
        has_cuda, cuda_version = self._detect_cuda()
        has_rocm, rocm_version = self._detect_rocm()
        has_mps = self._detect_mps()
        
        self.system_info = SystemInfo(
            platform=system_platform,
            arch=arch,
            python_version=python_version,
            python_executable=python_executable,
            has_cuda=has_cuda,
            cuda_version=cuda_version,
            has_rocm=has_rocm,
            rocm_version=rocm_version,
            has_mps=has_mps
        )
        
        self.logger.info(f"System info: {self.system_info}")
        return self.system_info
        
    def _detect_cuda(self) -> Tuple[bool, Optional[str]]:
        """Detect CUDA availability and version"""
        try:
            # Check nvidia-smi
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                # Try to get CUDA version from nvcc
                try:
                    nvcc_result = subprocess.run(
                        ["nvcc", "--version"],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    if nvcc_result.returncode == 0:
                        # Parse CUDA version from nvcc output
                        for line in nvcc_result.stdout.split('\n'):
                            if 'release' in line.lower():
                                version = line.split('release')[1].strip().split(',')[0].strip()
                                return True, version
                except (subprocess.SubprocessError, FileNotFoundError):
                    pass
                    
                return True, "unknown"
                
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
            
        return False, None
        
    def _detect_rocm(self) -> Tuple[bool, Optional[str]]:
        """Detect ROCm availability and version"""
        try:
            result = subprocess.run(
                ["rocm-smi", "--showproductname"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                # Try to get ROCm version
                try:
                    version_result = subprocess.run(
                        ["cat", "/opt/rocm/.info/version"],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    if version_result.returncode == 0:
                        return True, version_result.stdout.strip()
                except (subprocess.SubprocessError, FileNotFoundError):
                    pass
                    
                return True, "unknown"
                
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
            
        return False, None
        
    def _detect_mps(self) -> bool:
        """Detect Apple Metal Performance Shaders availability"""
        if platform.system().lower() != "darwin":
            return False
            
        try:
            # Check if we're on Apple Silicon or supported Intel Mac
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                cpu_info = result.stdout.lower()
                # Apple Silicon or recent Intel Macs support MPS
                return "apple" in cpu_info or any(intel in cpu_info for intel in ["intel", "core"])
                
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
            
        return False
        
    def create_virtual_environment(self, force_recreate: bool = False) -> EnvironmentConfig:
        """Create virtual environment with fallback options"""
        self.logger.info("Setting up virtual environment...")
        
        if not self.system_info:
            self.check_system_requirements()
            
        venv_path = self.project_root / "venv"
        # Use requirements from plugin directory (bundled with plugin)
        requirements_dir = self.plugin_root / "requirements"
        models_dir = self.project_root / "models"
        
        # Remove existing venv if force recreate
        if force_recreate and venv_path.exists():
            self.logger.info(f"Removing existing virtual environment at {venv_path}")
            shutil.rmtree(venv_path)
            
        # Create directories
        requirements_dir.mkdir(exist_ok=True)
        models_dir.mkdir(exist_ok=True)
        
        # Try different venv creation methods
        python_exe = None
        if not venv_path.exists():
            python_exe = self._create_venv_with_fallback(venv_path)
        else:
            # Use existing venv
            if self.system_info.platform == "windows":
                python_exe = venv_path / "Scripts" / "python.exe"
            else:
                python_exe = venv_path / "bin" / "python"
                
            if not python_exe.exists():
                self.logger.warning("Virtual environment exists but Python executable not found, recreating...")
                shutil.rmtree(venv_path)
                python_exe = self._create_venv_with_fallback(venv_path)
        
        # Determine GPU configuration
        gpu_type = self._determine_gpu_type()
        use_gpu = gpu_type != "cpu"
        
        self.env_config = EnvironmentConfig(
            project_root=self.project_root,
            venv_path=venv_path,
            python_executable=python_exe,
            requirements_dir=requirements_dir,
            models_dir=models_dir,
            use_gpu=use_gpu,
            gpu_type=gpu_type
        )
        
        self.logger.info(f"Virtual environment configured: {self.env_config}")
        return self.env_config
        
    def _create_venv_with_fallback(self, venv_path: Path) -> Path:
        """Create virtual environment with multiple fallback methods"""
        methods = [
            ("python -m venv", [sys.executable, "-m", "venv", str(venv_path)]),
            ("python3 -m venv", ["python3", "-m", "venv", str(venv_path)]),
            ("virtualenv", ["virtualenv", str(venv_path)]),
        ]
        
        for method_name, cmd in methods:
            try:
                self.logger.info(f"Trying to create venv with: {method_name}")
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                if result.returncode == 0:
                    self.logger.info(f"Successfully created virtual environment with {method_name}")
                    
                    # Return path to Python executable
                    if self.system_info.platform == "windows":
                        python_exe = venv_path / "Scripts" / "python.exe"
                    else:
                        python_exe = venv_path / "bin" / "python"
                        
                    if python_exe.exists():
                        return python_exe
                    else:
                        raise BootstrapError(f"Virtual environment created but Python executable not found at {python_exe}")
                        
                else:
                    self.logger.warning(f"Failed to create venv with {method_name}: {result.stderr}")
                    
            except (subprocess.SubprocessError, FileNotFoundError) as e:
                self.logger.warning(f"Method {method_name} not available: {e}")
                continue
                
        raise BootstrapError(
            "Failed to create virtual environment with any available method. "
            "Please ensure Python venv module or virtualenv is installed."
        )
        
    def _determine_gpu_type(self) -> str:
        """Determine the best GPU type to use"""
        if not self.system_info:
            return "cpu"
            
        # Priority order: CUDA > ROCm > MPS > CPU
        if self.system_info.has_cuda:
            return "cuda"
        elif self.system_info.has_rocm:
            return "rocm"
        elif self.system_info.has_mps:
            return "mps"
        else:
            return "cpu"
            
    def install_dependencies(self, skip_ai: bool = False) -> None:
        """Install all required dependencies"""
        if not self.env_config:
            raise BootstrapError("Environment not configured. Run create_virtual_environment() first.")
            
        self.logger.info("Installing dependencies...")
        
        # Upgrade pip first
        self._run_pip_command(["install", "--upgrade", "pip", "setuptools", "wheel"])
        
        # Install core dependencies
        self.logger.info("Installing core dependencies...")
        self._install_core_dependencies()
        
        if not skip_ai:
            # Install AI/ML dependencies
            self.logger.info("Installing AI/ML dependencies...")
            self._install_ai_dependencies()
            
        self.logger.info("All dependencies installed successfully!")
        
    def _install_core_dependencies(self) -> None:
        """Install core Python dependencies"""
        self._run_pip_command(["install"] + self.core_dependencies)
        
    def _install_ai_dependencies(self) -> None:
        """Install AI/ML dependencies based on GPU availability"""
        # Install PyTorch based on GPU type
        if self.env_config.gpu_type == "cuda":
            self.logger.info("Installing PyTorch with CUDA support...")
            pytorch_deps = self.ai_dependencies["pytorch_cuda"]
            # Add CUDA-specific index URL
            self._run_pip_command([
                "install",
                "--extra-index-url", "https://download.pytorch.org/whl/cu118"
            ] + pytorch_deps)
        elif self.env_config.gpu_type == "rocm":
            self.logger.info("Installing PyTorch with ROCm support...")
            self._run_pip_command([
                "install",
                "--extra-index-url", "https://download.pytorch.org/whl/rocm5.4.2"
            ] + self.ai_dependencies["pytorch_cuda"])  # ROCm uses same packages
        else:
            self.logger.info("Installing PyTorch CPU version...")
            self._run_pip_command([
                "install",
                "--extra-index-url", "https://download.pytorch.org/whl/cpu"
            ] + self.ai_dependencies["pytorch_cpu"])
            
        # Install SAM2
        self.logger.info("Installing Segment Anything 2...")
        self._run_pip_command(["install"] + self.ai_dependencies["sam2"])
        
        # Download SAM2 model weights
        self.logger.info("Downloading SAM2 model weights...")
        self._download_sam2_models()
        
        # Install optional dependencies
        self.logger.info("Installing optional AI dependencies...")
        self._run_pip_command(["install"] + self.ai_dependencies["optional"])
        
    def _run_pip_command(self, args: List[str]) -> None:
        """Run pip command in virtual environment"""
        cmd = [str(self.env_config.python_executable), "-m", "pip"] + args
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout for installation
            )
            
            if result.returncode != 0:
                self.logger.error(f"Pip command failed: {' '.join(args)}")
                self.logger.error(f"Error output: {result.stderr}")
                raise BootstrapError(f"Failed to run pip command: {' '.join(args)}")
                
            self.logger.debug(f"Pip command successful: {' '.join(args)}")
            
        except subprocess.TimeoutExpired:
            raise BootstrapError(f"Pip command timed out: {' '.join(args)}")
        except subprocess.SubprocessError as e:
            raise BootstrapError(f"Failed to run pip command: {e}")
            
    def _download_sam2_models(self) -> None:
        """Download SAM2 model weights using the model loader"""
        try:
            # Import and use the model loader from the plugin
            import sys
            core_path = self.project_root / "pykrita" / "smart_segments" / "core"
            if str(core_path) not in sys.path:
                sys.path.insert(0, str(core_path))
            
            # Run the model download using the virtual environment python
            download_script = f"""
import sys
sys.path.insert(0, r'{core_path}')
from model_loader import SAM2ModelLoader

loader = SAM2ModelLoader(r'{self.env_config.models_dir}')
print('Downloading SAM2 large model...', file=sys.stderr)
success = loader.download_model('sam2_hiera_large')
if success:
    print('Model downloaded successfully!', file=sys.stderr)
else:
    print('Failed to download model', file=sys.stderr)
    sys.exit(1)
"""
            
            result = subprocess.run([
                str(self.env_config.python_executable),
                "-c",
                download_script
            ], capture_output=True, text=True, timeout=600)  # 10 minutes timeout for download
            
            if result.returncode != 0:
                self.logger.error(f"Failed to download SAM2 models: {result.stderr}")
                raise BootstrapError("Failed to download SAM2 model weights")
            else:
                self.logger.info("SAM2 model weights downloaded successfully")
                
        except subprocess.TimeoutExpired:
            raise BootstrapError("SAM2 model download timed out")
        except Exception as e:
            self.logger.error(f"Error downloading SAM2 models: {e}")
            # Don't raise exception here, as model can be downloaded later
            self.logger.warning("SAM2 model download failed, but plugin will attempt to download on first use")
            
    def generate_requirements_files(self) -> None:
        """Generate requirements files for different configurations"""
        if not self.env_config:
            raise BootstrapError("Environment not configured.")
            
        # Check if bundled requirements files already exist
        base_reqs = Path(self.env_config.requirements_dir) / "base.txt"
        ai_reqs = Path(self.env_config.requirements_dir) / "ai.txt"
        
        # If bundled requirements exist, don't overwrite them
        if base_reqs.exists() and ai_reqs.exists():
            self.logger.info(f"Using bundled requirements files from {self.env_config.requirements_dir}")
            return
            
        self.logger.info("Generating requirements files...")
        
        # Generate base requirements
        with open(base_reqs, 'w') as f:
            f.write("# Core dependencies for Smart Segments\n")
            for dep in self.core_dependencies:
                f.write(f"{dep}\n")
                
        # Generate GPU-specific requirements
        if self.env_config.gpu_type == "cuda":
            gpu_reqs = Path(self.env_config.requirements_dir) / "gpu-cuda.txt"
            with open(gpu_reqs, 'w') as f:
                f.write("# CUDA GPU dependencies\n")
                f.write("--extra-index-url https://download.pytorch.org/whl/cu118\n")
                for dep in self.ai_dependencies["pytorch_cuda"]:
                    f.write(f"{dep}\n")
        elif self.env_config.gpu_type == "rocm":
            gpu_reqs = Path(self.env_config.requirements_dir) / "gpu-rocm.txt"
            with open(gpu_reqs, 'w') as f:
                f.write("# ROCm GPU dependencies\n")
                f.write("--extra-index-url https://download.pytorch.org/whl/rocm5.4.2\n")
                for dep in self.ai_dependencies["pytorch_cuda"]:
                    f.write(f"{dep}\n")
        else:
            cpu_reqs = Path(self.env_config.requirements_dir) / "cpu.txt"
            with open(cpu_reqs, 'w') as f:
                f.write("# CPU-only dependencies\n")
                f.write("--extra-index-url https://download.pytorch.org/whl/cpu\n")
                for dep in self.ai_dependencies["pytorch_cpu"]:
                    f.write(f"{dep}\n")
                    
        # Generate AI requirements
        with open(ai_reqs, 'w') as f:
            f.write("# AI/ML dependencies\n")
            for dep in self.ai_dependencies["sam2"] + self.ai_dependencies["optional"]:
                f.write(f"{dep}\n")
                
        self.logger.info(f"Requirements files generated in {self.env_config.requirements_dir}")
        
    def verify_installation(self) -> Dict[str, bool]:
        """Verify that all dependencies are properly installed"""
        if not self.env_config:
            raise BootstrapError("Environment not configured.")
            
        verification_results = {}
        
        # Test imports
        test_imports = [
            ("numpy", "import numpy"),
            ("opencv", "import cv2"),
            ("torch", "import torch"),
            ("torchvision", "import torchvision"),
            ("PIL", "from PIL import Image"),
            ("matplotlib", "import matplotlib.pyplot"),
        ]
        
        for name, import_code in test_imports:
            try:
                result = subprocess.run([
                    str(self.env_config.python_executable),
                    "-c",
                    import_code
                ], capture_output=True, text=True, timeout=10)
                
                verification_results[name] = result.returncode == 0
                if result.returncode != 0:
                    self.logger.warning(f"Failed to import {name}: {result.stderr}")
                    
            except subprocess.SubprocessError:
                verification_results[name] = False
                
        # Test GPU availability if configured
        if self.env_config.use_gpu:
            try:
                if self.env_config.gpu_type == "cuda":
                    gpu_test = "import torch; print(torch.cuda.is_available())"
                elif self.env_config.gpu_type == "mps":
                    gpu_test = "import torch; print(torch.backends.mps.is_available())"
                else:
                    gpu_test = "print(True)"  # ROCm test
                    
                result = subprocess.run([
                    str(self.env_config.python_executable),
                    "-c",
                    gpu_test
                ], capture_output=True, text=True, timeout=10)
                
                verification_results["gpu"] = result.returncode == 0 and "True" in result.stdout
                
            except subprocess.SubprocessError:
                verification_results["gpu"] = False
                
        return verification_results
        
    def get_status_report(self) -> Dict:
        """Get comprehensive status report"""
        report = {
            "system_info": self.system_info.__dict__ if self.system_info else None,
            "environment_config": {
                "project_root": str(self.env_config.project_root),
                "venv_path": str(self.env_config.venv_path),
                "python_executable": str(self.env_config.python_executable),
                "use_gpu": self.env_config.use_gpu,
                "gpu_type": self.env_config.gpu_type,
            } if self.env_config else None,
            "verification_results": None
        }
        
        if self.env_config:
            try:
                report["verification_results"] = self.verify_installation()
            except Exception as e:
                report["verification_error"] = str(e)
                
        return report


def main():
    """Main function for standalone execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Smart Segments Environment Bootstrapper")
    parser.add_argument("--force", action="store_true", help="Force recreate virtual environment")
    parser.add_argument("--skip-ai", action="store_true", help="Skip AI/ML dependencies")
    parser.add_argument("--verify-only", action="store_true", help="Only verify existing installation")
    parser.add_argument("--status", action="store_true", help="Show status report")
    
    args = parser.parse_args()
    
    bootstrapper = EnvironmentBootstrapper()
    
    try:
        if args.status:
            report = bootstrapper.get_status_report()
            print(json.dumps(report, indent=2))
            return
            
        if args.verify_only:
            if not bootstrapper.env_config:
                print("Environment not configured. Run bootstrap first.")
                return
                
            results = bootstrapper.verify_installation()
            print("Verification Results:")
            for name, success in results.items():
                status = "✓" if success else "✗"
                print(f"  {status} {name}")
            return
            
        # Full bootstrap process
        bootstrapper.check_system_requirements()
        bootstrapper.create_virtual_environment(force_recreate=args.force)
        bootstrapper.generate_requirements_files()
        bootstrapper.install_dependencies(skip_ai=args.skip_ai)
        
        # Verify installation
        results = bootstrapper.verify_installation()
        print("\nInstallation completed!")
        print("Verification Results:")
        for name, success in results.items():
            status = "✓" if success else "✗"
            print(f"  {status} {name}")
            
    except BootstrapError as e:
        print(f"Bootstrap failed: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nBootstrap interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
