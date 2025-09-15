"""
SAM2 Model Loader
Handles automatic download and loading of SAM2 models with GPU acceleration support
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from urllib.request import urlretrieve
from urllib.error import URLError

try:
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError as e:
    logging.warning(f"SAM2 dependencies not available: {e}")
    torch = None
    build_sam2 = None
    SAM2ImagePredictor = None


class SAM2ModelLoader:
    """
    SAM2 Model Loader with automatic download capabilities
    """
    
    # Model configurations
    MODEL_CONFIGS = {
        'sam2_hiera_large': {
            'config': 'sam2_hiera_l.yaml',
            'checkpoint_url': 'https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt'
        },
        'sam2_hiera_base_plus': {
            'config': 'sam2_hiera_b+.yaml',
            'checkpoint_url': 'https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt'
        },
        'sam2_hiera_small': {
            'config': 'sam2_hiera_s.yaml',
            'checkpoint_url': 'https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt'
        },
        'sam2_hiera_tiny': {
            'config': 'sam2_hiera_t.yaml',
            'checkpoint_url': 'https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt'
        }
    }
    
    def __init__(self, models_dir: Optional[str] = None):
        """
        Initialize SAM2 model loader
        
        Args:
            models_dir: Directory to store models (defaults to project models directory)
        """
        if models_dir is None:
            # Use Krita config directory for models
            import platform
            if platform.system() == "Windows":
                project_root = Path(os.path.expanduser("~")) / "AppData" / "Roaming" / "krita"
            elif platform.system() == "Darwin":  # macOS
                project_root = Path(os.path.expanduser("~")) / "Library" / "Application Support" / "krita"
            else:  # Linux and others
                # Try XDG_CONFIG_HOME first, fallback to ~/.config
                xdg_config = os.environ.get('XDG_CONFIG_HOME')
                if xdg_config:
                    project_root = Path(xdg_config) / "krita"
                else:
                    project_root = Path(os.path.expanduser("~")) / ".config" / "krita"
            models_dir = project_root / "models"
        
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging with same handler as main logger
        self.logger = logging.getLogger("smart_segments.model_loader")
        # Ensure it has the same handler as the main logger
        if not self.logger.handlers:
            main_logger = logging.getLogger("smart_segments_plugin")
            if main_logger.handlers:
                for handler in main_logger.handlers:
                    self.logger.addHandler(handler)
            self.logger.setLevel(logging.DEBUG)
        
        # Model cache
        self._model_cache: Dict[str, Any] = {}
        
    
    def _download_with_progress(self, url: str, dest_path: Path) -> bool:
        """
        Download file with progress reporting
        
        Args:
            url: URL to download from
            dest_path: Destination path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            def reporthook(block_num, block_size, total_size):
                if total_size > 0:
                    percent = min(100, (block_num * block_size * 100) // total_size)
                    if block_num % 100 == 0:  # Report every 100 blocks
                        self.logger.info(f"Download progress: {percent}%")
            
            self.logger.info(f"Downloading {url} to {dest_path}")
            urlretrieve(url, dest_path, reporthook)
            return True
            
        except URLError as e:
            self.logger.error(f"Failed to download {url}: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error downloading {url}: {e}")
            return False
    
    def download_model(self, model_name: str = 'sam2_hiera_large', force_redownload: bool = False) -> bool:
        """
        Download SAM2 model if not already present
        
        Args:
            model_name: Model variant to download
            force_redownload: Force redownload even if file exists
            
        Returns:
            True if model is available (downloaded or already present)
        """
        if model_name not in self.MODEL_CONFIGS:
            self.logger.error(f"Unknown model: {model_name}")
            return False
        
        config = self.MODEL_CONFIGS[model_name]
        checkpoint_path = self.models_dir / f"{model_name}.pt"
        
        # Check if model already exists
        if not force_redownload and checkpoint_path.exists():
            self.logger.info(f"Model {model_name} already exists")
            return True
        
        # Download model
        self.logger.info(f"Downloading {model_name} model...")
        if not self._download_with_progress(config['checkpoint_url'], checkpoint_path):
            return False
        
        self.logger.info(f"Model {model_name} downloaded successfully")
        return True
    
    def get_available_models(self) -> list:
        """
        Get list of available model variants
        
        Returns:
            List of available model names
        """
        return list(self.MODEL_CONFIGS.keys())
    
    def is_model_available(self, model_name: str) -> bool:
        """
        Check if model is available locally
        
        Args:
            model_name: Model variant to check
            
        Returns:
            True if model is available locally
        """
        if model_name not in self.MODEL_CONFIGS:
            return False
        
        checkpoint_path = self.models_dir / f"{model_name}.pt"
        
        return checkpoint_path.exists()
    
    def load_model(self, model_name: str = 'sam2_hiera_large', device: Optional[str] = None) -> Optional[Any]:
        """
        Load SAM2 model for inference
        
        Args:
            model_name: Model variant to load
            device: Device to load model on ('cuda', 'cpu', or None for auto)
            
        Returns:
            Loaded SAM2 model or None if failed
        """
        if torch is None or build_sam2 is None:
            self.logger.error("SAM2 dependencies not available")
            return None
        
        # Check cache first
        cache_key = f"{model_name}_{device}"
        if cache_key in self._model_cache:
            self.logger.info(f"Using cached model {model_name}")
            return self._model_cache[cache_key]
        
        # Ensure model is downloaded
        if not self.is_model_available(model_name):
            self.logger.info(f"Model {model_name} not available, attempting download...")
            if not self.download_model(model_name):
                self.logger.error(f"Failed to download model {model_name}")
                return None
        
        # Determine device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            config = self.MODEL_CONFIGS[model_name]
            checkpoint_path = self.models_dir / f"{model_name}.pt"
            
            self.logger.info(f"Loading {model_name} from {checkpoint_path} on {device}...")
            
            # Verify checkpoint file exists
            if not checkpoint_path.exists():
                self.logger.error(f"Checkpoint file not found at {checkpoint_path}")
                return None
            
            self.logger.info(f"Checkpoint file size: {checkpoint_path.stat().st_size / 1024 / 1024:.2f} MB")
            
            # Build model using SAM2 config - need full path to config file
            # Find sam2 package location
            import sam2
            sam2_path = Path(sam2.__file__).parent
            config_path = sam2_path / "configs" / "sam2" / config['config']
            
            if not config_path.exists():
                self.logger.error(f"Config file not found at {config_path}")
                return None
            
            self.logger.info(f"Building SAM2 model with config: {config_path}")
            model = build_sam2(str(config_path), str(checkpoint_path), device=device)
            
            # Cache the model
            self._model_cache[cache_key] = model
            
            self.logger.info(f"Model {model_name} loaded successfully")
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to load model {model_name}: {e}", exc_info=True)
            return None
    
    def clear_cache(self):
        """
        Clear model cache to free memory
        """
        self._model_cache.clear()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.logger.info("Model cache cleared")
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a model
        
        Args:
            model_name: Model variant name
            
        Returns:
            Dictionary with model information or None if not found
        """
        if model_name not in self.MODEL_CONFIGS:
            return None
        
        config = self.MODEL_CONFIGS[model_name]
        checkpoint_path = self.models_dir / f"{model_name}.pt"
        
        info = {
            'name': model_name,
            'config': config['config'],
            'url': config['checkpoint_url'],
            'local_path': str(checkpoint_path),
            'available': self.is_model_available(model_name),
            'size_mb': checkpoint_path.stat().st_size / (1024 * 1024) if checkpoint_path.exists() else 0
        }
        
        return info

