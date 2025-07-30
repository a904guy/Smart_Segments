"""
System-specific utilities for Smart Segments Plugin

This module provides platform-specific system operations, resource management,
and system integration utilities.
"""

import os
import sys
import platform
import subprocess
import signal
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass

from .platform_utils import path_handler, PlatformType

# Optional psutil import for system monitoring
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    psutil = None


@dataclass
class SystemInfo:
    """System information data class"""
    platform: PlatformType
    os_name: str
    os_version: str
    architecture: str
    cpu_count: int
    memory_total: int  # MB
    memory_available: int  # MB
    disk_space_free: int  # MB
    python_version: str
    python_executable: str


@dataclass
class ProcessInfo:
    """Process information data class"""
    pid: int
    name: str
    status: str
    cpu_percent: float
    memory_percent: float
    memory_mb: float
    create_time: float
    cmdline: List[str]


class SystemResourceManager:
    """Manages system resources and monitoring"""
    
    def __init__(self):
        self.platform = path_handler.platform
    
    def get_system_info(self) -> SystemInfo:
        """
        Get comprehensive system information
        
        Returns:
            SystemInfo: System information object
        """
        if HAS_PSUTIL:
            # Get memory info
            memory = psutil.virtual_memory()
            
            # Get disk space for current directory
            disk_usage = psutil.disk_usage('.')
            
            return SystemInfo(
                platform=self.platform,
                os_name=platform.system(),
                os_version=platform.release(),
                architecture=platform.machine(),
                cpu_count=psutil.cpu_count(),
                memory_total=memory.total // (1024 * 1024),
                memory_available=memory.available // (1024 * 1024),
                disk_space_free=disk_usage.free // (1024 * 1024),
                python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                python_executable=sys.executable
            )
        else:
            # Fallback without psutil
            import shutil
            
            # Get basic info without psutil
            total, used, free = shutil.disk_usage('.')
            
            return SystemInfo(
                platform=self.platform,
                os_name=platform.system(),
                os_version=platform.release(),
                architecture=platform.machine(),
                cpu_count=os.cpu_count() or 1,
                memory_total=0,  # Cannot determine without psutil
                memory_available=0,  # Cannot determine without psutil
                disk_space_free=free // (1024 * 1024),
                python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                python_executable=sys.executable
            )
    
    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get current memory usage information
        
        Returns:
            Dict: Memory usage statistics in MB
        """
        if not HAS_PSUTIL:
            return {
                'total': 0,
                'available': 0,
                'used': 0,
                'percent': 0,
                'free': 0
            }
        
        memory = psutil.virtual_memory()
        return {
            'total': memory.total / (1024 * 1024),
            'available': memory.available / (1024 * 1024),
            'used': memory.used / (1024 * 1024),
            'percent': memory.percent,
            'free': memory.free / (1024 * 1024) if hasattr(memory, 'free') else 0
        }
    
    def get_cpu_usage(self, interval: float = 1.0) -> Dict[str, float]:
        """
        Get CPU usage information
        
        Args:
            interval: Measurement interval in seconds
            
        Returns:
            Dict: CPU usage statistics
        """
        if not HAS_PSUTIL:
            cpu_count = os.cpu_count() or 1
            return {
                'overall': 0,
                'per_core': [0] * cpu_count,
                'count': cpu_count,
                'count_logical': cpu_count
            }
        
        cpu_percent = psutil.cpu_percent(interval=interval, percpu=True)
        return {
            'overall': psutil.cpu_percent(interval=0),
            'per_core': cpu_percent,
            'count': psutil.cpu_count(),
            'count_logical': psutil.cpu_count(logical=True)
        }
    
    def get_disk_usage(self, path: Union[str, Path] = '.') -> Dict[str, float]:
        """
        Get disk usage for a path
        
        Args:
            path: Path to check disk usage for
            
        Returns:
            Dict: Disk usage statistics in MB
        """
        if not HAS_PSUTIL:
            import shutil
            total, used, free = shutil.disk_usage(str(path))
            return {
                'total': total / (1024 * 1024),
                'used': used / (1024 * 1024),
                'free': free / (1024 * 1024),
                'percent': (used / total) * 100 if total > 0 else 0
            }
        
        usage = psutil.disk_usage(str(path))
        return {
            'total': usage.total / (1024 * 1024),
            'used': usage.used / (1024 * 1024),
            'free': usage.free / (1024 * 1024),
            'percent': (usage.used / usage.total) * 100
        }
    
    def monitor_process(self, pid: int) -> Optional[ProcessInfo]:
        """
        Get information about a specific process
        
        Args:
            pid: Process ID to monitor
            
        Returns:
            ProcessInfo: Process information or None if not found
        """
        if not HAS_PSUTIL:
            # Fallback: Basic process check without detailed info
            try:
                os.kill(pid, 0)  # Check if process exists
                return ProcessInfo(
                    pid=pid,
                    name="unknown",
                    status="unknown",
                    cpu_percent=0.0,
                    memory_percent=0.0,
                    memory_mb=0.0,
                    create_time=0.0,
                    cmdline=[]
                )
            except (OSError, ProcessLookupError):
                return None
        
        try:
            process = psutil.Process(pid)
            
            return ProcessInfo(
                pid=process.pid,
                name=process.name(),
                status=process.status(),
                cpu_percent=process.cpu_percent(),
                memory_percent=process.memory_percent(),
                memory_mb=process.memory_info().rss / (1024 * 1024),
                create_time=process.create_time(),
                cmdline=process.cmdline()
            )
        
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return None
    
    def find_processes_by_name(self, name: str) -> List[ProcessInfo]:
        """
        Find processes by name
        
        Args:
            name: Process name to search for
            
        Returns:
            List[ProcessInfo]: List of matching processes
        """
        if not HAS_PSUTIL:
            # Fallback: Platform-specific process enumeration
            processes = []
            try:
                if self.platform == PlatformType.WINDOWS:
                    # Windows: Use tasklist command
                    result = subprocess.run(
                        ['tasklist', '/fo', 'csv'],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    if result.returncode == 0:
                        lines = result.stdout.strip().split('\n')[1:]  # Skip header
                        for line in lines:
                            if name.lower() in line.lower():
                                parts = line.split(',')
                                if len(parts) >= 2:
                                    try:
                                        pid = int(parts[1].strip('"'))
                                        process_info = self.monitor_process(pid)
                                        if process_info:
                                            processes.append(process_info)
                                    except ValueError:
                                        continue
                else:
                    # Unix-like: Use ps command
                    result = subprocess.run(
                        ['ps', 'aux'],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    if result.returncode == 0:
                        lines = result.stdout.strip().split('\n')[1:]  # Skip header
                        for line in lines:
                            if name.lower() in line.lower():
                                parts = line.split()
                                if len(parts) >= 2:
                                    try:
                                        pid = int(parts[1])
                                        process_info = self.monitor_process(pid)
                                        if process_info:
                                            processes.append(process_info)
                                    except ValueError:
                                        continue
            except Exception:
                pass
            
            return processes
        
        processes = []
        
        for proc in psutil.process_iter(['pid', 'name', 'status']):
            try:
                if name.lower() in proc.info['name'].lower():
                    process_info = self.monitor_process(proc.info['pid'])
                    if process_info:
                        processes.append(process_info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        return processes
    
    def terminate_process(self, pid: int, timeout: int = 5) -> bool:
        """
        Terminate a process gracefully
        
        Args:
            pid: Process ID to terminate
            timeout: Timeout in seconds
            
        Returns:
            bool: True if process terminated successfully
        """
        if not HAS_PSUTIL:
            # Fallback: Platform-specific process termination
            try:
                if self.platform == PlatformType.WINDOWS:
                    # Windows: Use taskkill command
                    result = subprocess.run(
                        ['taskkill', '/PID', str(pid), '/F'],
                        capture_output=True,
                        text=True,
                        timeout=timeout
                    )
                    return result.returncode == 0
                else:
                    # Unix-like: Use os.kill with SIGTERM, then SIGKILL
                    try:
                        os.kill(pid, signal.SIGTERM)
                        
                        # Wait a bit for graceful termination
                        import time
                        for _ in range(timeout * 10):  # Check every 0.1 seconds
                            try:
                                os.kill(pid, 0)  # Check if process still exists
                                time.sleep(0.1)
                            except (OSError, ProcessLookupError):
                                return True  # Process terminated
                        
                        # Force kill if still running
                        os.kill(pid, signal.SIGKILL)
                        return True
                        
                    except (OSError, ProcessLookupError):
                        return True  # Process already terminated or doesn't exist
                        
            except Exception:
                return False
        
        try:
            process = psutil.Process(pid)
            process.terminate()
            
            # Wait for process to terminate
            try:
                process.wait(timeout=timeout)
                return True
            except psutil.TimeoutExpired:
                # Force kill if graceful termination failed
                process.kill()
                return True
                
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return False


class PlatformSpecificOperations:
    """Platform-specific operations and integrations"""
    
    def __init__(self):
        self.platform = path_handler.platform
        self.resource_manager = SystemResourceManager()
    
    def get_environment_variables(self) -> Dict[str, str]:
        """Get all environment variables"""
        return dict(os.environ)
    
    def set_environment_variable(self, name: str, value: str, persistent: bool = False) -> bool:
        """
        Set environment variable
        
        Args:
            name: Variable name
            value: Variable value
            persistent: Whether to make persistent (platform-dependent)
            
        Returns:
            bool: Success status
        """
        try:
            os.environ[name] = value
            
            if persistent:
                return self._set_persistent_env_var(name, value)
            
            return True
        
        except Exception:
            return False
    
    def _set_persistent_env_var(self, name: str, value: str) -> bool:
        """Set persistent environment variable (platform-specific)"""
        try:
            if self.platform == PlatformType.WINDOWS:
                # Windows: Use setx command
                result = subprocess.run(
                    ['setx', name, value],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                return result.returncode == 0
            
            elif self.platform == PlatformType.MACOS:
                # macOS: Add to ~/.bash_profile and ~/.zshrc
                home = path_handler.get_home_directory()
                export_line = f"export {name}='{value}'\n"
                
                for profile in ['.bash_profile', '.zshrc']:
                    profile_path = home / profile
                    try:
                        with open(profile_path, 'a') as f:
                            f.write(export_line)
                    except OSError:
                        pass
                
                return True
            
            else:
                # Linux: Add to ~/.bashrc and ~/.profile
                home = path_handler.get_home_directory()
                export_line = f"export {name}='{value}'\n"
                
                for profile in ['.bashrc', '.profile']:
                    profile_path = home / profile
                    try:
                        with open(profile_path, 'a') as f:
                            f.write(export_line)
                    except OSError:
                        pass
                
                return True
        
        except Exception:
            return False
    
    def get_system_path(self) -> List[str]:
        """Get system PATH as list of directories"""
        path_env = os.environ.get('PATH', '')
        
        if self.platform == PlatformType.WINDOWS:
            separator = ';'
        else:
            separator = ':'
        
        return [p for p in path_env.split(separator) if p]
    
    def add_to_system_path(self, directory: Union[str, Path], persistent: bool = False) -> bool:
        """
        Add directory to system PATH
        
        Args:
            directory: Directory to add
            persistent: Whether to make persistent
            
        Returns:
            bool: Success status
        """
        try:
            dir_path = str(path_handler.normalize_path(directory))
            current_path = self.get_system_path()
            
            # Don't add if already in PATH
            if dir_path in current_path:
                return True
            
            # Add to current session PATH
            if self.platform == PlatformType.WINDOWS:
                separator = ';'
            else:
                separator = ':'
            
            new_path = separator.join([dir_path] + current_path)
            os.environ['PATH'] = new_path
            
            if persistent:
                return self.set_environment_variable('PATH', new_path, persistent=True)
            
            return True
        
        except Exception:
            return False
    
    def get_default_browser(self) -> Optional[str]:
        """Get default web browser"""
        try:
            if self.platform == PlatformType.WINDOWS:
                # Windows: Query registry
                result = subprocess.run([
                    'reg', 'query',
                    'HKEY_CURRENT_USER\\Software\\Microsoft\\Windows\\Shell\\Associations\\UrlAssociations\\http\\UserChoice',
                    '/v', 'ProgId'
                ], capture_output=True, text=True, timeout=10)
                
                if result.returncode == 0:
                    for line in result.stdout.split('\n'):
                        if 'ProgId' in line:
                            return line.split()[-1]
            
            elif self.platform == PlatformType.MACOS:
                # macOS: Use defaults command
                result = subprocess.run([
                    'defaults', 'read',
                    'com.apple.LaunchServices/com.apple.launchservices.secure',
                    'LSHandlers'
                ], capture_output=True, text=True, timeout=10)
                
                if result.returncode == 0:
                    # Parse for HTTP handler
                    return "Safari"  # Default fallback
            
            else:
                # Linux: Check xdg-settings
                result = subprocess.run([
                    'xdg-settings', 'get', 'default-web-browser'
                ], capture_output=True, text=True, timeout=10)
                
                if result.returncode == 0:
                    return result.stdout.strip()
        
        except Exception:
            pass
        
        return None
    
    def open_file_with_default_app(self, file_path: Union[str, Path]) -> bool:
        """
        Open file with default application
        
        Args:
            file_path: Path to file to open
            
        Returns:
            bool: True if opened successfully
        """
        try:
            file_path = path_handler.normalize_path(file_path)
            
            if self.platform == PlatformType.WINDOWS:
                os.startfile(str(file_path))
            elif self.platform == PlatformType.MACOS:
                subprocess.run(['open', str(file_path)], timeout=10)
            else:
                subprocess.run(['xdg-open', str(file_path)], timeout=10)
            
            return True
        
        except Exception:
            return False
    
    def get_file_associations(self, extension: str) -> List[str]:
        """
        Get applications associated with file extension
        
        Args:
            extension: File extension (with or without dot)
            
        Returns:
            List[str]: List of associated application names
        """
        if not extension.startswith('.'):
            extension = f'.{extension}'
        
        applications = []
        
        try:
            if self.platform == PlatformType.WINDOWS:
                # Windows: Query registry
                result = subprocess.run([
                    'assoc', extension
                ], capture_output=True, text=True, timeout=10)
                
                if result.returncode == 0:
                    file_type = result.stdout.strip().split('=')[-1]
                    
                    # Get associated command
                    cmd_result = subprocess.run([
                        'ftype', file_type
                    ], capture_output=True, text=True, timeout=10)
                    
                    if cmd_result.returncode == 0:
                        applications.append(cmd_result.stdout.strip())
            
            elif self.platform == PlatformType.MACOS:
                # macOS: Use mdls to get associations
                result = subprocess.run([
                    'mdls', '-name', 'kMDItemContentType',
                    '-name', 'kMDItemContentTypeTree',
                    f'test{extension}'
                ], capture_output=True, text=True, timeout=10)
                
                # This is a simplified approach
                applications.append("Default macOS application")
            
            else:
                # Linux: Use xdg-mime
                result = subprocess.run([
                    'xdg-mime', 'query', 'default',
                    f'application/{extension[1:]}'
                ], capture_output=True, text=True, timeout=10)
                
                if result.returncode == 0:
                    applications.append(result.stdout.strip())
        
        except Exception:
            pass
        
        return applications
    
    def create_desktop_shortcut(self, name: str, target: Union[str, Path], 
                              icon: Optional[Union[str, Path]] = None,
                              description: str = "") -> bool:
        """
        Create desktop shortcut
        
        Args:
            name: Shortcut name
            target: Target executable or file
            icon: Icon file path
            description: Shortcut description
            
        Returns:
            bool: True if created successfully
        """
        try:
            target_path = path_handler.normalize_path(target)
            desktop = path_handler.get_home_directory() / "Desktop"
            
            if self.platform == PlatformType.WINDOWS:
                # Windows: Create .lnk file
                import winshell
                from win32com.client import Dispatch
                
                shortcut_path = desktop / f"{name}.lnk"
                shell = Dispatch('WScript.Shell')
                shortcut = shell.CreateShortCut(str(shortcut_path))
                shortcut.Targetpath = str(target_path)
                shortcut.WorkingDirectory = str(target_path.parent)
                
                if icon:
                    shortcut.IconLocation = str(path_handler.normalize_path(icon))
                
                shortcut.save()
                return True
            
            elif self.platform == PlatformType.LINUX:
                # Linux: Create .desktop file
                shortcut_path = desktop / f"{name}.desktop"
                
                content = f"""[Desktop Entry]
Version=1.0
Type=Application
Name={name}
Comment={description}
Exec={target_path}
Terminal=false
"""
                
                if icon:
                    content += f"Icon={path_handler.normalize_path(icon)}\n"
                
                return path_handler.write_text_file(shortcut_path, content)
            
            else:
                # macOS: Create alias (simplified)
                # This would require more complex implementation with Apple Events
                return False
        
        except Exception:
            return False
    
    def get_installed_applications(self) -> List[Dict[str, str]]:
        """
        Get list of installed applications
        
        Returns:
            List[Dict]: List of application information
        """
        applications = []
        
        try:
            if self.platform == PlatformType.WINDOWS:
                # Windows: Query WMI for installed programs
                result = subprocess.run([
                    'wmic', 'product', 'get', 'name,version,vendor',
                    '/format:csv'
                ], capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')[1:]  # Skip header
                    for line in lines:
                        if line.strip():
                            parts = line.split(',')
                            if len(parts) >= 4:
                                applications.append({
                                    'name': parts[1].strip(),
                                    'version': parts[3].strip(),
                                    'vendor': parts[2].strip()
                                })
            
            elif self.platform == PlatformType.MACOS:
                # macOS: List applications in /Applications
                apps_dir = Path('/Applications')
                if apps_dir.exists():
                    for app_path in apps_dir.glob('*.app'):
                        applications.append({
                            'name': app_path.stem,
                            'path': str(app_path),
                            'version': 'Unknown'
                        })
            
            else:
                # Linux: Check common application directories
                app_dirs = [
                    Path('/usr/share/applications'),
                    Path('/usr/local/share/applications'),
                    path_handler.get_home_directory() / '.local/share/applications'
                ]
                
                for app_dir in app_dirs:
                    if app_dir.exists():
                        for desktop_file in app_dir.glob('*.desktop'):
                            try:
                                content = desktop_file.read_text()
                                name = None
                                for line in content.split('\n'):
                                    if line.startswith('Name='):
                                        name = line.split('=', 1)[1]
                                        break
                                
                                if name:
                                    applications.append({
                                        'name': name,
                                        'desktop_file': str(desktop_file),
                                        'version': 'Unknown'
                                    })
                            except Exception:
                                continue
        
        except Exception:
            pass
        
        return applications


# Convenience instances
resource_manager = SystemResourceManager()
platform_ops = PlatformSpecificOperations()
