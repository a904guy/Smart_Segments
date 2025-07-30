"""
Cross-Platform Compatibility Layer for Smart Segments Plugin

This module provides platform-specific implementations for path handling,
Python detection, file operations, and other OS-specific tasks to ensure
consistent behavior across Windows, Linux, and macOS.
"""

import os
import sys
import platform
import subprocess
import shutil
from pathlib import Path, PurePath, PurePosixPath, PureWindowsPath
from typing import Dict, List, Optional, Tuple, Union, Any
from enum import Enum


class PlatformType(Enum):
    """Supported platform types"""
    WINDOWS = "windows"
    LINUX = "linux"
    MACOS = "macos"
    UNKNOWN = "unknown"


class CrossPlatformPathHandler:
    """Handles platform-specific path operations"""
    
    def __init__(self):
        self.platform = self._detect_platform()
        self.separator = self._get_path_separator()
        self.line_ending = self._get_line_ending()
    
    @staticmethod
    def _detect_platform() -> PlatformType:
        """Detect the current platform"""
        system = platform.system().lower()
        if system == "windows":
            return PlatformType.WINDOWS
        elif system == "linux":
            return PlatformType.LINUX
        elif system == "darwin":
            return PlatformType.MACOS
        else:
            return PlatformType.UNKNOWN
    
    def _get_path_separator(self) -> str:
        """Get platform-specific path separator"""
        return os.sep
    
    def _get_line_ending(self) -> str:
        """Get platform-specific line ending"""
        if self.platform == PlatformType.WINDOWS:
            return "\r\n"
        else:
            return "\n"
    
    def normalize_path(self, path: Union[str, Path]) -> Path:
        """
        Normalize path for current platform
        
        Args:
            path: Path to normalize
            
        Returns:
            Path: Normalized path object
        """
        path_obj = Path(path)
        
        # Convert to absolute path if relative
        if not path_obj.is_absolute():
            path_obj = path_obj.resolve()
        
        # Handle Windows drive letters
        if self.platform == PlatformType.WINDOWS:
            return path_obj
        else:
            # Convert Windows-style paths on Unix systems
            path_str = str(path_obj)
            if '\\' in path_str:
                path_str = path_str.replace('\\', '/')
                return Path(path_str)
            return path_obj
    
    def join_paths(self, *paths: Union[str, Path]) -> Path:
        """
        Join multiple paths using platform-appropriate separators
        
        Args:
            *paths: Paths to join
            
        Returns:
            Path: Joined path
        """
        if not paths:
            return Path()
        
        # Convert all paths to strings and join
        path_parts = [str(p) for p in paths]
        joined = Path(path_parts[0])
        
        for part in path_parts[1:]:
            joined = joined / part
            
        return self.normalize_path(joined)
    
    def get_home_directory(self) -> Path:
        """Get user home directory"""
        return Path.home()
    
    def get_temp_directory(self) -> Path:
        """Get system temporary directory"""
        import tempfile
        return Path(tempfile.gettempdir())
    
    def get_app_data_directory(self, app_name: str = "SmartSegments") -> Path:
        """
        Get platform-specific application data directory
        
        Args:
            app_name: Name of the application
            
        Returns:
            Path: Application data directory
        """
        if self.platform == PlatformType.WINDOWS:
            # Windows: %APPDATA%\AppName
            appdata = os.environ.get('APPDATA')
            if appdata:
                return Path(appdata) / app_name
            else:
                return self.get_home_directory() / "AppData" / "Roaming" / app_name
        
        elif self.platform == PlatformType.MACOS:
            # macOS: ~/Library/Application Support/AppName
            return self.get_home_directory() / "Library" / "Application Support" / app_name
        
        else:
            # Linux/Unix: ~/.config/appname (lowercase)
            config_home = os.environ.get('XDG_CONFIG_HOME')
            if config_home:
                return Path(config_home) / app_name.lower()
            else:
                return self.get_home_directory() / ".config" / app_name.lower()
    
    def get_cache_directory(self, app_name: str = "SmartSegments") -> Path:
        """
        Get platform-specific cache directory
        
        Args:
            app_name: Name of the application
            
        Returns:
            Path: Cache directory
        """
        if self.platform == PlatformType.WINDOWS:
            # Windows: %LOCALAPPDATA%\AppName\Cache
            localappdata = os.environ.get('LOCALAPPDATA')
            if localappdata:
                return Path(localappdata) / app_name / "Cache"
            else:
                return self.get_home_directory() / "AppData" / "Local" / app_name / "Cache"
        
        elif self.platform == PlatformType.MACOS:
            # macOS: ~/Library/Caches/AppName
            return self.get_home_directory() / "Library" / "Caches" / app_name
        
        else:
            # Linux/Unix: ~/.cache/appname (lowercase)
            cache_home = os.environ.get('XDG_CACHE_HOME')
            if cache_home:
                return Path(cache_home) / app_name.lower()
            else:
                return self.get_home_directory() / ".cache" / app_name.lower()
    
    def is_case_sensitive_filesystem(self) -> bool:
        """Check if the filesystem is case-sensitive"""
        if self.platform == PlatformType.WINDOWS:
            return False
        elif self.platform == PlatformType.MACOS:
            # macOS can be either, but default is case-insensitive
            return False
        else:
            # Linux and other Unix systems are typically case-sensitive
            return True
    
    def get_executable_extension(self) -> str:
        """Get platform-specific executable extension"""
        if self.platform == PlatformType.WINDOWS:
            return ".exe"
        else:
            return ""
    
    def convert_path_for_subprocess(self, path: Union[str, Path]) -> str:
        """
        Convert path to format suitable for subprocess calls
        
        Args:
            path: Path to convert
            
        Returns:
            str: Converted path string
        """
        path_str = str(self.normalize_path(path))
        
        # On Windows, wrap paths with spaces in quotes
        if self.platform == PlatformType.WINDOWS and ' ' in path_str:
            return f'"{path_str}"'
        
        return path_str


class PythonDetector:
    """Detects Python installations across different platforms"""
    
    def __init__(self):
        self.path_handler = CrossPlatformPathHandler()
        self.platform = self.path_handler.platform
    
    def find_python_executables(self) -> List[Dict[str, Any]]:
        """
        Find all Python executables on the system
        
        Returns:
            List[Dict]: List of Python installation info
        """
        pythons = []
        
        # Check current Python
        current_python = self._get_current_python_info()
        if current_python:
            pythons.append(current_python)
        
        # Search in common locations
        search_paths = self._get_python_search_paths()
        
        for search_path in search_paths:
            found_pythons = self._search_directory_for_python(search_path)
            for python in found_pythons:
                # Avoid duplicates
                if not any(p['executable'] == python['executable'] for p in pythons):
                    pythons.append(python)
        
        # Search in PATH
        path_pythons = self._search_path_for_python()
        for python in path_pythons:
            if not any(p['executable'] == python['executable'] for p in pythons):
                pythons.append(python)
        
        return pythons
    
    def _get_current_python_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the current Python interpreter"""
        try:
            return {
                'executable': sys.executable,
                'version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                'version_info': sys.version_info,
                'is_virtual_env': self._is_virtual_environment(),
                'virtual_env_path': self._get_virtual_env_path(),
                'architecture': platform.architecture()[0],
                'is_current': True
            }
        except Exception:
            return None
    
    def _get_python_search_paths(self) -> List[Path]:
        """Get platform-specific Python search paths"""
        paths = []
        
        if self.platform == PlatformType.WINDOWS:
            # Windows Python locations
            paths.extend([
                Path("C:/Python*"),
                Path("C:/Program Files/Python*"),
                Path("C:/Program Files (x86)/Python*"),
                self.path_handler.get_home_directory() / "AppData/Local/Programs/Python",
                Path("C:/ProgramData/Microsoft/Windows/Start Menu/Programs/Python*"),
            ])
            
            # Check Windows Store Python
            try:
                local_packages = self.path_handler.get_home_directory() / "AppData/Local/Packages"
                if local_packages.exists():
                    for item in local_packages.iterdir():
                        if "python" in item.name.lower():
                            paths.append(item / "LocalCache/local-packages/Python*/Scripts")
            except Exception:
                pass
        
        elif self.platform == PlatformType.MACOS:
            # macOS Python locations
            paths.extend([
                Path("/usr/bin"),
                Path("/usr/local/bin"),
                Path("/opt/homebrew/bin"),
                Path("/System/Library/Frameworks/Python.framework/Versions"),
                Path("/Library/Frameworks/Python.framework/Versions"),
                self.path_handler.get_home_directory() / ".pyenv/versions",
                Path("/opt/local/bin"),  # MacPorts
            ])
        
        else:
            # Linux/Unix Python locations
            paths.extend([
                Path("/usr/bin"),
                Path("/usr/local/bin"),
                Path("/opt/python*/bin"),
                self.path_handler.get_home_directory() / ".pyenv/versions",
                Path("/snap/bin"),  # Snap packages
                Path("/usr/lib/python*"),
            ])
        
        return [p for p in paths if p.exists()]
    
    def _search_directory_for_python(self, directory: Path) -> List[Dict[str, Any]]:
        """Search a directory for Python executables"""
        pythons = []
        
        if not directory.exists():
            return pythons
        
        try:
            python_names = ["python", "python3", "python2"]
            if self.platform == PlatformType.WINDOWS:
                python_names.extend(["python.exe", "python3.exe", "python2.exe"])
            
            for item in directory.rglob("*"):
                if item.is_file() and item.name.lower() in [n.lower() for n in python_names]:
                    python_info = self._get_python_info(item)
                    if python_info:
                        pythons.append(python_info)
        
        except (PermissionError, OSError):
            pass
        
        return pythons
    
    def _search_path_for_python(self) -> List[Dict[str, Any]]:
        """Search PATH environment variable for Python"""
        pythons = []
        
        try:
            python_names = ["python", "python3", "python2"]
            
            for name in python_names:
                python_path = shutil.which(name)
                if python_path:
                    python_info = self._get_python_info(Path(python_path))
                    if python_info:
                        pythons.append(python_info)
        
        except Exception:
            pass
        
        return pythons
    
    def _get_python_info(self, executable: Path) -> Optional[Dict[str, Any]]:
        """Get information about a Python executable"""
        try:
            # Run python --version to get version info
            result = subprocess.run(
                [str(executable), "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                return None
            
            version_str = result.stdout.strip() or result.stderr.strip()
            version_str = version_str.replace("Python ", "")
            
            # Parse version
            version_parts = version_str.split(".")
            if len(version_parts) < 2:
                return None
            
            major = int(version_parts[0])
            minor = int(version_parts[1])
            micro = int(version_parts[2]) if len(version_parts) > 2 else 0
            
            # Get architecture info
            arch_result = subprocess.run(
                [str(executable), "-c", "import platform; print(platform.architecture()[0])"],
                capture_output=True,
                text=True,
                timeout=10
            )
            architecture = arch_result.stdout.strip() if arch_result.returncode == 0 else "unknown"
            
            return {
                'executable': str(executable),
                'version': version_str,
                'version_info': (major, minor, micro),
                'is_virtual_env': False,  # Cannot determine for external Python
                'virtual_env_path': None,
                'architecture': architecture,
                'is_current': str(executable) == sys.executable
            }
        
        except (subprocess.SubprocessError, ValueError, IndexError):
            return None
    
    def _is_virtual_environment(self) -> bool:
        """Check if current Python is in a virtual environment"""
        return (
            hasattr(sys, 'real_prefix') or
            (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
        )
    
    def _get_virtual_env_path(self) -> Optional[str]:
        """Get virtual environment path if in one"""
        if not self._is_virtual_environment():
            return None
        
        venv_path = os.environ.get('VIRTUAL_ENV')
        if venv_path:
            return venv_path
        
        return sys.prefix
    
    def get_recommended_python(self, min_version: Tuple[int, int] = (3, 8)) -> Optional[Dict[str, Any]]:
        """
        Get recommended Python installation
        
        Args:
            min_version: Minimum required version (major, minor)
            
        Returns:
            Dict: Python installation info or None
        """
        pythons = self.find_python_executables()
        
        # Filter by minimum version
        compatible_pythons = [
            p for p in pythons
            if p['version_info'][:2] >= min_version
        ]
        
        if not compatible_pythons:
            return None
        
        # Prefer current Python if compatible
        current_python = next((p for p in compatible_pythons if p.get('is_current')), None)
        if current_python:
            return current_python
        
        # Otherwise, prefer newest version
        return max(compatible_pythons, key=lambda p: p['version_info'])


class PortableFileOperations:
    """Portable file operations that work across platforms"""
    
    def __init__(self):
        self.path_handler = CrossPlatformPathHandler()
    
    def copy_file(self, src: Union[str, Path], dst: Union[str, Path], 
                  preserve_metadata: bool = True) -> bool:
        """
        Copy file with platform-appropriate method
        
        Args:
            src: Source file path
            dst: Destination file path
            preserve_metadata: Whether to preserve file metadata
            
        Returns:
            bool: Success status
        """
        try:
            src_path = self.path_handler.normalize_path(src)
            dst_path = self.path_handler.normalize_path(dst)
            
            # Ensure destination directory exists
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            
            if preserve_metadata:
                shutil.copy2(src_path, dst_path)
            else:
                shutil.copy(src_path, dst_path)
            
            return True
        
        except (OSError, shutil.Error):
            return False
    
    def copy_directory(self, src: Union[str, Path], dst: Union[str, Path], 
                      ignore_patterns: Optional[List[str]] = None) -> bool:
        """
        Copy directory recursively
        
        Args:
            src: Source directory path
            dst: Destination directory path
            ignore_patterns: List of patterns to ignore
            
        Returns:
            bool: Success status
        """
        try:
            src_path = self.path_handler.normalize_path(src)
            dst_path = self.path_handler.normalize_path(dst)
            
            ignore_func = None
            if ignore_patterns:
                ignore_func = shutil.ignore_patterns(*ignore_patterns)
            
            shutil.copytree(src_path, dst_path, ignore=ignore_func, dirs_exist_ok=True)
            return True
        
        except (OSError, shutil.Error):
            return False
    
    def move_file(self, src: Union[str, Path], dst: Union[str, Path]) -> bool:
        """
        Move file or directory
        
        Args:
            src: Source path
            dst: Destination path
            
        Returns:
            bool: Success status
        """
        try:
            src_path = self.path_handler.normalize_path(src)
            dst_path = self.path_handler.normalize_path(dst)
            
            # Ensure destination directory exists
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            
            shutil.move(src_path, dst_path)
            return True
        
        except (OSError, shutil.Error):
            return False
    
    def delete_file(self, path: Union[str, Path]) -> bool:
        """
        Delete file safely
        
        Args:
            path: File path to delete
            
        Returns:
            bool: Success status
        """
        try:
            file_path = self.path_handler.normalize_path(path)
            
            if file_path.is_file():
                file_path.unlink()
                return True
            
            return False
        
        except OSError:
            return False
    
    def delete_directory(self, path: Union[str, Path], ignore_errors: bool = True) -> bool:
        """
        Delete directory recursively
        
        Args:
            path: Directory path to delete
            ignore_errors: Whether to ignore deletion errors
            
        Returns:
            bool: Success status
        """
        try:
            dir_path = self.path_handler.normalize_path(path)
            
            if dir_path.is_dir():
                shutil.rmtree(dir_path, ignore_errors=ignore_errors)
                return True
            
            return False
        
        except OSError:
            return False
    
    def create_directory(self, path: Union[str, Path], parents: bool = True, 
                        exist_ok: bool = True) -> bool:
        """
        Create directory with proper permissions
        
        Args:
            path: Directory path to create
            parents: Whether to create parent directories
            exist_ok: Whether to ignore if directory exists
            
        Returns:
            bool: Success status
        """
        try:
            dir_path = self.path_handler.normalize_path(path)
            dir_path.mkdir(parents=parents, exist_ok=exist_ok)
            return True
        
        except OSError:
            return False
    
    def read_text_file(self, path: Union[str, Path], encoding: str = "utf-8") -> Optional[str]:
        """
        Read text file with platform-appropriate encoding
        
        Args:
            path: File path to read
            encoding: Text encoding to use
            
        Returns:
            str: File contents or None if failed
        """
        try:
            file_path = self.path_handler.normalize_path(path)
            return file_path.read_text(encoding=encoding)
        
        except (OSError, UnicodeDecodeError):
            return None
    
    def write_text_file(self, path: Union[str, Path], content: str, 
                       encoding: str = "utf-8", line_ending: Optional[str] = None) -> bool:
        """
        Write text file with platform-appropriate line endings
        
        Args:
            path: File path to write
            content: Content to write
            encoding: Text encoding to use
            line_ending: Line ending to use (None for platform default)
            
        Returns:
            bool: Success status
        """
        try:
            file_path = self.path_handler.normalize_path(path)
            
            # Ensure directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Normalize line endings if specified
            if line_ending is not None:
                content = content.replace('\r\n', '\n').replace('\r', '\n')
                if line_ending != '\n':
                    content = content.replace('\n', line_ending)
            elif line_ending is None:
                # Use platform default line ending
                content = content.replace('\r\n', '\n').replace('\r', '\n')
                content = content.replace('\n', self.path_handler.line_ending)
            
            file_path.write_text(content, encoding=encoding)
            return True
        
        except (OSError, UnicodeEncodeError):
            return False
    
    def get_file_permissions(self, path: Union[str, Path]) -> Optional[int]:
        """
        Get file permissions (Unix-style)
        
        Args:
            path: File path
            
        Returns:
            int: File permissions or None if failed
        """
        try:
            file_path = self.path_handler.normalize_path(path)
            return file_path.stat().st_mode & 0o777
        
        except OSError:
            return None
    
    def set_file_permissions(self, path: Union[str, Path], permissions: int) -> bool:
        """
        Set file permissions (Unix-style, no-op on Windows)
        
        Args:
            path: File path
            permissions: Permissions to set
            
        Returns:
            bool: Success status
        """
        try:
            if self.path_handler.platform == PlatformType.WINDOWS:
                # Windows doesn't use Unix-style permissions
                return True
            
            file_path = self.path_handler.normalize_path(path)
            file_path.chmod(permissions)
            return True
        
        except OSError:
            return False
    
    def is_executable(self, path: Union[str, Path]) -> bool:
        """
        Check if file is executable
        
        Args:
            path: File path to check
            
        Returns:
            bool: True if executable
        """
        try:
            file_path = self.path_handler.normalize_path(path)
            
            if self.path_handler.platform == PlatformType.WINDOWS:
                # On Windows, check file extension
                executable_extensions = ['.exe', '.bat', '.cmd', '.com', '.scr']
                return file_path.suffix.lower() in executable_extensions
            else:
                # On Unix systems, check execute permission
                return os.access(file_path, os.X_OK)
        
        except OSError:
            return False
    
    def make_executable(self, path: Union[str, Path]) -> bool:
        """
        Make file executable
        
        Args:
            path: File path
            
        Returns:
            bool: Success status
        """
        try:
            if self.path_handler.platform == PlatformType.WINDOWS:
                # Windows doesn't need explicit execute permissions
                return True
            
            file_path = self.path_handler.normalize_path(path)
            current_permissions = file_path.stat().st_mode
            new_permissions = current_permissions | 0o111  # Add execute for all
            file_path.chmod(new_permissions)
            return True
        
        except OSError:
            return False


# Convenience instances for easy import
path_handler = CrossPlatformPathHandler()
python_detector = PythonDetector()
file_ops = PortableFileOperations()
