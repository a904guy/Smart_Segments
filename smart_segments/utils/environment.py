"""
Environment Utilities for Smart Segments Plugin

This module provides utility functions for environment management,
dependency checking, and system detection.
"""

import os
import sys
import subprocess
import importlib.util
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Import cross-platform compatibility utilities
from .platform_utils import path_handler, python_detector, file_ops, PlatformType


class EnvironmentUtils:
    """Utility class for environment-related operations"""
    
    @staticmethod
    def is_dependency_installed(package_name: str, import_name: Optional[str] = None) -> bool:
        """
        Check if a Python package is installed and importable
        
        Args:
            package_name: Name of the package (e.g., 'opencv-python')
            import_name: Name used for import (e.g., 'cv2'). If None, uses package_name
            
        Returns:
            bool: True if package is installed and importable
        """
        import_name = import_name or package_name.replace('-', '_')
        
        try:
            importlib.import_module(import_name)
            return True
        except ImportError:
            return False
    
    @staticmethod
    def get_python_executable() -> str:
        """Get the current Python executable path"""
        return sys.executable
    
    @staticmethod
    def get_python_version() -> Tuple[int, int, int]:
        """Get the current Python version as a tuple"""
        return sys.version_info[:3]
    
    @staticmethod
    def is_virtual_environment() -> bool:
        """Check if running in a virtual environment"""
        return (
            hasattr(sys, 'real_prefix') or
            (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
        )
    
    @staticmethod
    def get_virtual_env_path() -> Optional[Path]:
        """Get the virtual environment path if running in one"""
        if not EnvironmentUtils.is_virtual_environment():
            return None
            
        # Try different methods to get venv path
        venv_path = os.environ.get('VIRTUAL_ENV')
        if venv_path:
            return Path(venv_path)
            
        # Fallback: derive from sys.prefix
        return Path(sys.prefix)
    
    @staticmethod
    def run_command(cmd: List[str], timeout: int = 30, capture_output: bool = True) -> subprocess.CompletedProcess:
        """
        Run a system command with error handling
        
        Args:
            cmd: Command to run as list of strings
            timeout: Timeout in seconds
            capture_output: Whether to capture stdout/stderr
            
        Returns:
            subprocess.CompletedProcess: Result of the command
            
        Raises:
            subprocess.SubprocessError: If command fails
        """
        try:
            return subprocess.run(
                cmd,
                capture_output=capture_output,
                text=True,
                timeout=timeout,
                check=True
            )
        except subprocess.TimeoutExpired as e:
            raise subprocess.SubprocessError(f"Command timed out: {' '.join(cmd)}")
        except subprocess.CalledProcessError as e:
            raise subprocess.SubprocessError(f"Command failed: {' '.join(cmd)}, error: {e.stderr}")
    
    @staticmethod
    def check_gpu_availability() -> Dict[str, Union[bool, str]]:
        """
        Check GPU availability for different backends
        
        Returns:
            Dict with GPU availability information
        """
        gpu_info = {
            'cuda': False,
            'cuda_version': None,
            'rocm': False,
            'rocm_version': None,
            'mps': False,
            'recommended': 'cpu'
        }
        
        # Check CUDA
        try:
            result = subprocess.run(
                ['nvidia-smi', '--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                gpu_info['cuda'] = True
                gpu_info['recommended'] = 'cuda'
                
                # Try to get CUDA version
                try:
                    nvcc_result = subprocess.run(
                        ['nvcc', '--version'],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    if nvcc_result.returncode == 0:
                        for line in nvcc_result.stdout.split('\n'):
                            if 'release' in line.lower():
                                version = line.split('release')[1].strip().split(',')[0].strip()
                                gpu_info['cuda_version'] = version
                                break
                except (subprocess.SubprocessError, FileNotFoundError):
                    pass
                    
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
        
        # Check ROCm
        try:
            result = subprocess.run(
                ['rocm-smi', '--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                gpu_info['rocm'] = True
                if not gpu_info['cuda']:  # Prefer CUDA over ROCm
                    gpu_info['recommended'] = 'rocm'
                    
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
        
        # Check MPS (Apple Silicon)
        import platform
        if platform.system().lower() == 'darwin':
            try:
                result = subprocess.run(
                    ['sysctl', '-n', 'machdep.cpu.brand_string'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    cpu_info = result.stdout.lower()
                    if 'apple' in cpu_info or 'intel' in cpu_info:
                        gpu_info['mps'] = True
                        if not gpu_info['cuda'] and not gpu_info['rocm']:
                            gpu_info['recommended'] = 'mps'
                            
            except (subprocess.SubprocessError, FileNotFoundError):
                pass
        
        return gpu_info
    
    @staticmethod
    def get_system_memory() -> Optional[int]:
        """
        Get system memory in MB
        
        Returns:
            int: System memory in MB, or None if unable to determine
        """
        import platform
        
        try:
            if platform.system().lower() == 'linux':
                with open('/proc/meminfo', 'r') as f:
                    for line in f:
                        if line.startswith('MemTotal:'):
                            # Extract memory in kB and convert to MB
                            mem_kb = int(line.split()[1])
                            return mem_kb // 1024
                            
            elif platform.system().lower() == 'darwin':
                result = subprocess.run(
                    ['sysctl', '-n', 'hw.memsize'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    mem_bytes = int(result.stdout.strip())
                    return mem_bytes // (1024 * 1024)
                    
            elif platform.system().lower() == 'windows':
                result = subprocess.run(
                    ['wmic', 'computersystem', 'get', 'TotalPhysicalMemory', '/value'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    for line in result.stdout.split('\n'):
                        if 'TotalPhysicalMemory=' in line:
                            mem_bytes = int(line.split('=')[1].strip())
                            return mem_bytes // (1024 * 1024)
                            
        except (subprocess.SubprocessError, FileNotFoundError, ValueError):
            pass
            
        return None
    
    @staticmethod
    def check_disk_space(path: Union[str, Path], required_mb: int = 5120) -> Tuple[bool, int]:
        """
        Check available disk space at given path
        
        Args:
            path: Path to check
            required_mb: Required space in MB (default: 5120 MB = 5 GB)
            
        Returns:
            Tuple of (has_enough_space, available_mb)
        """
        try:
            import shutil
            
            path = Path(path)
            if not path.exists():
                path = path.parent
                
            total, used, free = shutil.disk_usage(str(path))
            free_mb = free // (1024 * 1024)
            
            return free_mb >= required_mb, free_mb
            
        except Exception:
            return False, 0
    
    @staticmethod
    def get_package_version(package_name: str, import_name: Optional[str] = None) -> Optional[str]:
        """
        Get version of an installed package
        
        Args:
            package_name: Name of the package
            import_name: Name used for import. If None, uses package_name
            
        Returns:
            str: Package version or None if not found
        """
        import_name = import_name or package_name.replace('-', '_')
        
        try:
            module = importlib.import_module(import_name)
            
            # Try different version attributes
            version_attrs = ['__version__', 'version', 'VERSION']
            for attr in version_attrs:
                if hasattr(module, attr):
                    version = getattr(module, attr)
                    if isinstance(version, str):
                        return version
                    elif hasattr(version, '__str__'):
                        return str(version)
                        
        except ImportError:
            pass
            
        # Fallback: try using pkg_resources or importlib.metadata
        try:
            import pkg_resources
            return pkg_resources.get_distribution(package_name).version
        except (ImportError, pkg_resources.DistributionNotFound):
            pass
            
        try:
            import importlib.metadata
            return importlib.metadata.version(package_name)
        except (ImportError, importlib.metadata.PackageNotFoundError):
            pass
            
        return None
    
    @staticmethod
    def create_requirements_list(dependencies: List[str]) -> str:
        """
        Create a requirements.txt formatted string from dependency list
        
        Args:
            dependencies: List of dependency specifications
            
        Returns:
            str: Requirements.txt formatted content
        """
        content = "# Generated requirements file\n"
        content += "# Install with: pip install -r requirements.txt\n\n"
        
        for dep in dependencies:
            content += f"{dep}\n"
            
        return content
    
    @staticmethod
    def validate_python_version(min_version: Tuple[int, int, int] = (3, 8, 0)) -> Tuple[bool, str]:
        """
        Validate that Python version meets minimum requirements
        
        Args:
            min_version: Minimum required version as tuple
            
        Returns:
            Tuple of (is_valid, message)
        """
        current = sys.version_info[:3]
        
        if current >= min_version:
            return True, f"Python {'.'.join(map(str, current))} meets requirements"
        else:
            return False, (
                f"Python {'.'.join(map(str, min_version))} or higher required, "
                f"but found {'.'.join(map(str, current))}"
            )


class DependencyChecker:
    """Helper class for checking specific dependencies"""
    
    # Core dependencies mapping
    CORE_PACKAGES = {
        'numpy': 'numpy',
        'opencv-python': 'cv2',
        'pillow': 'PIL',
        'scipy': 'scipy',
        'matplotlib': 'matplotlib',
        'requests': 'requests',
        'tqdm': 'tqdm',
    }
    
    # AI/ML dependencies mapping
    AI_PACKAGES = {
        'torch': 'torch',
        'torchvision': 'torchvision',
        'torchaudio': 'torchaudio',
        'transformers': 'transformers',
        'accelerate': 'accelerate',
        'safetensors': 'safetensors',
    }
    
    @classmethod
    def check_core_dependencies(cls) -> Dict[str, bool]:
        """Check availability of core dependencies"""
        results = {}
        for package, import_name in cls.CORE_PACKAGES.items():
            results[package] = EnvironmentUtils.is_dependency_installed(package, import_name)
        return results
    
    @classmethod
    def check_ai_dependencies(cls) -> Dict[str, bool]:
        """Check availability of AI/ML dependencies"""
        results = {}
        for package, import_name in cls.AI_PACKAGES.items():
            results[package] = EnvironmentUtils.is_dependency_installed(package, import_name)
        return results
    
    @classmethod
    def check_all_dependencies(cls) -> Dict[str, Dict[str, bool]]:
        """Check all dependencies grouped by category"""
        return {
            'core': cls.check_core_dependencies(),
            'ai': cls.check_ai_dependencies(),
        }
    
    @classmethod
    def get_missing_dependencies(cls) -> Dict[str, List[str]]:
        """Get list of missing dependencies by category"""
        all_deps = cls.check_all_dependencies()
        missing = {}
        
        for category, deps in all_deps.items():
            missing[category] = [
                package for package, installed in deps.items()
                if not installed
            ]
            
        return missing
    
    @classmethod
    def generate_installation_commands(cls) -> Dict[str, List[str]]:
        """Generate pip install commands for missing dependencies"""
        missing = cls.get_missing_dependencies()
        commands = {}
        
        if missing['core']:
            commands['core'] = ['pip', 'install'] + missing['core']
            
        if missing['ai']:
            # AI dependencies might need special handling for GPU versions
            gpu_info = EnvironmentUtils.check_gpu_availability()
            ai_cmd = ['pip', 'install']
            
            if gpu_info['cuda']:
                ai_cmd.extend(['--extra-index-url', 'https://download.pytorch.org/whl/cu118'])
            elif gpu_info['recommended'] == 'cpu':
                ai_cmd.extend(['--extra-index-url', 'https://download.pytorch.org/whl/cpu'])
                
            ai_cmd.extend(missing['ai'])
            commands['ai'] = ai_cmd
            
        return commands
