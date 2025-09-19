#!/usr/bin/env python3
"""
Create Krita Python Plugin ZIP for Smart Segments Plugin
"""

import os
import shutil
import zipfile
from pathlib import Path

def create_plugin_package():
    """Create a Krita Python plugin ZIP file for distribution"""
    
    print("Starting plugin package creation...")
    
    # Configuration
    plugin_name = "smart_segments"
    version = "1.0.2"
    package_name = f"SmartSegments_v{version}.zip"
    
    # Paths
    project_root = Path(__file__).parent
    temp_dir = project_root / "temp_package"
    package_path = project_root / "dist" / package_name
    
    print(f"Creating package: {package_name}")
    
    # Create directories
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(project_root / "dist", exist_ok=True)
    
    try:
        print("Copying plugin files...")
        
        # Create nested directory structure like working plugins
        package_dir = temp_dir / package_name.replace('.zip', '')
        package_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Copy plugin folder to nested directory (excluding cache files)
        plugin_src = project_root / "smart_segments"
        plugin_dst = package_dir / "smart_segments"
        if not plugin_src.exists():
            raise FileNotFoundError(f"Plugin source directory not found: {plugin_src}")
        shutil.copytree(plugin_src, plugin_dst, ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.pyo', '.pytest_cache'))
        
        print("Copying desktop file...")
        
        # 2. Copy existing desktop file with correct Krita naming convention
        desktop_src = project_root / "smart_segments.desktop"
        desktop_dst = package_dir / "smart_segments.desktop"
        if not desktop_src.exists():
            raise FileNotFoundError(f"Desktop file not found: {desktop_src}")
        shutil.copy2(desktop_src, desktop_dst)
        
        print("Creating README...")
        
        # 3. Create README for the package
        readme_content = f"""# Smart Segments v{version} - Krita Plugin

## Description
AI-powered intelligent segmentation tool that uses advanced machine learning models to automatically detect and segment objects in your artwork.

## Installation
1. Download this ZIP file
2. Open Krita
3. Go to Tools → Scripts → Import Python Plugin from File...
4. Select this ZIP file
5. Restart Krita
6. Find the plugin in Tools → Scripts → Smart Segments

## First Use
When you first run the plugin, it will automatically:
- Check system requirements
- Create a virtual environment
- Download AI models (~1 GB)
- Install dependencies

This process requires an internet connection and may take 10-15 minutes.

## Usage
- **Smart Segments**: Interactive segmentation with clickable preview
- **Smart Settings Technicals**: View plugin status and configuration

## System Requirements
- Krita 5.0+ with Python support
- 4GB+ RAM recommended
- 5-6 GB free disk space
- Internet connection for initial setup

## Support
- GitHub: https://github.com/a904guy/Smart_Segments
- Issues: https://github.com/a904guy/Smart_Segments/issues

## License
MIT License - see LICENSE file for details
"""
        
        with open(temp_dir / "README.md", "w") as f:
            f.write(readme_content)
        
        print("Copying additional files...")
        
        # 4. Copy additional files to root level
        if (project_root / "LICENSE").exists():
            shutil.copy2(project_root / "LICENSE", temp_dir / "LICENSE")
        
        # Copy the manual file to plugin directory where desktop file expects it
        if (project_root / "Manual.html").exists():
            shutil.copy2(project_root / "Manual.html", package_dir / "smart_segments" / "Manual.html")
        
        print("Creating ZIP file...")
        
        # 5. Create the package using command-line zip (like the working method)
        import subprocess
        result = subprocess.run([
            'zip', '-r', str(package_path), package_name.replace('.zip', '')
        ], cwd=temp_dir, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise Exception(f"ZIP creation failed: {result.stderr}")
        
        print(f"✅ Package created successfully: {package_path}")
        print(f"📦 Package size: {package_path.stat().st_size / (1024*1024):.1f} MB")
        print(f"\nInstallation Instructions:")
        print(f"1. Open Krita")
        print(f"2. Go to Tools → Scripts → Import Python Plugin from File...")
        print(f"3. Select {package_path.name}")
        print(f"4. Restart Krita")
        print(f"5. Find the plugin in Tools → Scripts → Smart Segments")
        
        return package_path
        
    finally:
        # Cleanup
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    create_plugin_package()
