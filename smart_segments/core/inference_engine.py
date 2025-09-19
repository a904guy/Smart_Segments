"""
SAM2 Inference Engine
GPU-accelerated inference engine for SAM2 model with memory management
"""

import logging
import numpy as np
from typing import Optional, Tuple, List, Dict, Any, Union
from pathlib import Path

try:
    import torch
    import cv2
    from PIL import Image
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
except ImportError as e:
    logging.warning(f"Inference engine dependencies not available: {e}")
    torch = None
    cv2 = None
    Image = None
    SAM2ImagePredictor = None
    SAM2AutomaticMaskGenerator = None

from .model_loader import SAM2ModelLoader


class SAM2InferenceEngine:
    """
    SAM2 Inference Engine with GPU acceleration and memory management
    """
    
    def __init__(self, model_name: str = 'sam2_hiera_large', device: Optional[str] = None):
        """
        Initialize inference engine
        
        Args:
            model_name: SAM2 model variant to use
            device: Device to run inference on ('cuda', 'cpu', or auto-detect)
        """
        # Use a named logger that will be picked up by the main logging config
        self.logger = logging.getLogger("smart_segments.inference_engine")
        # Ensure it has the same handler as the main logger
        if not self.logger.handlers:
            main_logger = logging.getLogger("smart_segments_plugin")
            if main_logger.handlers:
                for handler in main_logger.handlers:
                    self.logger.addHandler(handler)
            self.logger.setLevel(logging.DEBUG)
        
        # Check dependencies
        if torch is None or SAM2ImagePredictor is None:
            raise RuntimeError("Required dependencies not available")
        
        # Initialize model loader
        self.model_loader = SAM2ModelLoader()
        self.model_name = model_name
        
        # Device selection with CUDA compatibility checking
        if device is None:
            from ..utils.system_utils import CUDACompatibilityChecker
            recommended_device, reason = CUDACompatibilityChecker.get_recommended_device()
            self.device = recommended_device
            self.logger.info(f"Auto-selected device: {self.device} ({reason})")
        else:
            self.device = device
        
        self.logger.info(f"Initializing inference engine on {self.device}")
        
        # Initialize predictor
        self.predictor: Optional[SAM2ImagePredictor] = None
        self.current_image: Optional[np.ndarray] = None
        self.image_embeddings: Optional[torch.Tensor] = None
        
        # Memory management settings
        self.max_cache_size = 512 * 1024 * 1024  # 512MB cache limit
        self.enable_mixed_precision = self.device == "cuda"
        
        # Performance monitoring
        self.inference_times: List[float] = []
        
    def _ensure_model_loaded(self) -> bool:
        """
        Ensure SAM2 model is loaded and predictor is initialized
        
        Returns:
            True if model is ready, False otherwise
        """
        if self.predictor is not None:
            return True
        
        try:
            # Load model with device fallback
            self.logger.info(f"Loading model {self.model_name} on device {self.device}...")
            model = self.model_loader.load_model(self.model_name, self.device)
            
            # If CUDA failed and we were using CUDA, try CPU fallback
            if model is None and self.device == "cuda":
                self.logger.warning("CUDA model loading failed, attempting CPU fallback...")
                self.device = "cpu"
                self.enable_mixed_precision = False  # Disable mixed precision for CPU
                model = self.model_loader.load_model(self.model_name, self.device)
                
            if model is None:
                self.logger.error("Failed to load SAM2 model on both CUDA and CPU")
                return False
            
            self.logger.info(f"Model loaded successfully, initializing predictor...")
            
            # Disable JIT scripting for Krita compatibility
            # Krita's embedded Python has issues with inspect.getsource
            import torch
            original_script = torch.jit.script
            torch.jit.script = lambda x, *args, **kwargs: x  # Return unscripted module
            
            try:
                # Initialize predictor
                self.predictor = SAM2ImagePredictor(model)
            finally:
                # Restore original scripting function
                torch.jit.script = original_script
            
            # Enable mixed precision if supported
            if self.enable_mixed_precision and hasattr(self.predictor, 'model'):
                self.predictor.model.half()
                self.logger.info("Enabled mixed precision (FP16)")
            
            self.logger.info(f"SAM2 predictor initialized successfully with {self.model_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize predictor: {e}", exc_info=True)
            return False
    
    def _preprocess_image(self, image: Union[np.ndarray, str, Path]) -> Optional[np.ndarray]:
        """
        Preprocess image for SAM2 inference
        
        Args:
            image: Input image (numpy array, file path, or PIL Image)
            
        Returns:
            Preprocessed image as numpy array or None if failed
        """
        try:
            if isinstance(image, (str, Path)):
                # Load from file
                if cv2 is not None:
                    img = cv2.imread(str(image))
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    else:
                        # Fallback to PIL
                        img = np.array(Image.open(image).convert('RGB'))
                else:
                    img = np.array(Image.open(image).convert('RGB'))
            elif isinstance(image, np.ndarray):
                img = image.copy()
                # Ensure RGB format
                if len(img.shape) == 3 and img.shape[2] == 3:
                    # Assume BGR if opencv format, convert to RGB
                    if img.dtype == np.uint8 and np.max(img) > 1:
                        pass  # Already in correct format
                elif len(img.shape) == 3 and img.shape[2] == 4:
                    # RGBA to RGB
                    img = img[:, :, :3]
                else:
                    raise ValueError(f"Unsupported image shape: {img.shape}")
            else:
                raise ValueError(f"Unsupported image type: {type(image)}")
            
            # Validate image
            if img is None or img.size == 0:
                raise ValueError("Empty or invalid image")
            
            # Ensure proper data type
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
            
            return img
            
        except Exception as e:
            self.logger.error(f"Failed to preprocess image: {e}")
            return None
    
    def set_image(self, image: Union[np.ndarray, str, Path]) -> bool:
        """
        Set image for segmentation
        
        Args:
            image: Input image
            
        Returns:
            True if image was set successfully
        """
        self.logger.info(f"set_image called with image type: {type(image)}")
        if not self._ensure_model_loaded():
            self.logger.error("Model loading failed in set_image")
            return False
        
        # Preprocess image
        processed_image = self._preprocess_image(image)
        if processed_image is None:
            return False
        
        try:
            import time
            start_time = time.time()
            
            # Set image in predictor (this computes embeddings)
            with torch.cuda.amp.autocast(enabled=self.enable_mixed_precision):
                self.predictor.set_image(processed_image)
            
            # Store current image
            self.current_image = processed_image
            
            # Record timing
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            
            self.logger.info(f"Image set successfully (embedding time: {inference_time:.3f}s)")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to set image: {e}", exc_info=True)
            return False
    
    def predict(self, 
                points: List[Tuple[int, int]], 
                labels: List[int],
                box: Optional[Tuple[int, int, int, int]] = None,
                mask_input: Optional[np.ndarray] = None,
                multimask_output: bool = True,
                return_logits: bool = False) -> Optional[Dict[str, Any]]:
        """
        Perform segmentation prediction
        
        Args:
            points: List of (x, y) click coordinates
            labels: List of point labels (1 for foreground, 0 for background)
            box: Optional bounding box as (x1, y1, x2, y2)
            mask_input: Optional input mask from previous prediction
            multimask_output: Whether to return multiple mask proposals
            return_logits: Whether to return raw logits
            
        Returns:
            Dictionary containing masks, scores, and logits
        """
        if self.predictor is None or self.current_image is None:
            self.logger.error("Model not loaded or image not set")
            return None
        
        if len(points) != len(labels):
            self.logger.error("Points and labels must have same length")
            return None
        
        try:
            import time
            start_time = time.time()
            
            # Convert inputs to numpy arrays
            point_coords = np.array(points, dtype=np.float32) if points else None
            point_labels = np.array(labels, dtype=np.int32) if labels else None
            
            # Convert box to numpy array if provided
            box_coords = np.array(box, dtype=np.float32) if box is not None else None
            
            # Perform prediction
            with torch.cuda.amp.autocast(enabled=self.enable_mixed_precision):
                masks, scores, logits = self.predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    box=box_coords,
                    mask_input=mask_input,
                    multimask_output=multimask_output,
                    return_logits=return_logits
                )
            
            # Record timing
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            
            # Prepare result
            result = {
                'masks': masks,
                'scores': scores,
                'inference_time': inference_time
            }
            
            if return_logits:
                result['logits'] = logits
            
            self.logger.info(f"Prediction completed (inference time: {inference_time:.3f}s)")
            return result
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            return None
    
    def predict_single_point(self, 
                           x: int, 
                           y: int, 
                           is_positive: bool = True,
                           multimask_output: bool = True) -> Optional[Dict[str, Any]]:
        """
        Convenience method for single point prediction
        
        Args:
            x: X coordinate of click
            y: Y coordinate of click
            is_positive: Whether this is a positive (foreground) point
            multimask_output: Whether to return multiple mask proposals
            
        Returns:
            Prediction result dictionary
        """
        points = [(x, y)]
        labels = [1 if is_positive else 0]
        
        return self.predict(points, labels, multimask_output=multimask_output)
    
    def predict_multi_points(self, 
                           positive_points: List[Tuple[int, int]],
                           negative_points: Optional[List[Tuple[int, int]]] = None,
                           multimask_output: bool = False) -> Optional[Dict[str, Any]]:
        """
        Convenience method for multi-point prediction
        
        Args:
            positive_points: List of positive (foreground) points
            negative_points: List of negative (background) points
            multimask_output: Whether to return multiple mask proposals
            
        Returns:
            Prediction result dictionary
        """
        points = positive_points.copy()
        labels = [1] * len(positive_points)
        
        if negative_points:
            points.extend(negative_points)
            labels.extend([0] * len(negative_points))
        
        return self.predict(points, labels, multimask_output=multimask_output)
    
    def predict_everything(self) -> Optional[List[Dict[str, Any]]]:
        """
        Automatically segments all objects in the image using SamAutomaticMaskGenerator.
        Returns a list of mask dictionaries, each containing 'segmentation', 'area', 'bbox', etc.
        
        Returns:
            List of mask dictionaries or None if failed
        """
        if SAM2AutomaticMaskGenerator is None:
            self.logger.error("SAM2AutomaticMaskGenerator not available")
            return None
            
        if self.current_image is None:
            self.logger.error("No image set for automatic segmentation")
            return None
            
        if not self._ensure_model_loaded():
            return None
            
        try:
            import time
            start_time = time.time()
            
            # Load the raw model (not the predictor wrapper)
            model = self.model_loader.load_model(self.model_name, self.device)
            if model is None:
                self.logger.error("Failed to load raw model for automatic segmentation")
                return None
                
            # Disable JIT scripting for Krita compatibility
            import torch
            original_script = torch.jit.script
            torch.jit.script = lambda x, *args, **kwargs: x  # Return unscripted module
            
            try:
                # Create automatic mask generator
                mask_generator = SAM2AutomaticMaskGenerator(
                    model=model,
                    points_per_side=32,
                    pred_iou_thresh=0.88,
                    stability_score_thresh=0.95,
                    crop_n_layers=0,
                    crop_n_points_downscale_factor=1,
                    min_mask_region_area=100,  # Minimum area in pixels
                )
            finally:
                # Restore original scripting function
                torch.jit.script = original_script
            
            self.logger.info("Running automatic mask generation...")
            
            # Generate masks
            with torch.cuda.amp.autocast(enabled=self.enable_mixed_precision):
                masks = mask_generator.generate(self.current_image)
            
            # Record timing
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            
            self.logger.info(f"Generated {len(masks)} masks in {inference_time:.3f}s")
            return masks
            
        except Exception as e:
            self.logger.error(f"Automatic mask generation failed: {e}")
            return None
    
    def get_best_mask(self, prediction_result: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Get the best mask from prediction result
        
        Args:
            prediction_result: Result from predict method
            
        Returns:
            Best mask as numpy array or None
        """
        if not prediction_result or 'masks' not in prediction_result:
            return None
        
        masks = prediction_result['masks']
        scores = prediction_result.get('scores', None)
        
        if masks is None or len(masks) == 0:
            return None
        
        # Return highest scoring mask if scores available
        if scores is not None and len(scores) > 0:
            best_idx = np.argmax(scores)
            return masks[best_idx]
        
        # Otherwise return first mask
        return masks[0]
    
    def clear_cache(self):
        """
        Clear inference cache and free GPU memory
        """
        self.current_image = None
        self.image_embeddings = None
        
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Clear model cache
        if self.model_loader:
            self.model_loader.clear_cache()
        
        self.logger.info("Inference cache cleared")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.inference_times:
            return {}
        
        times = np.array(self.inference_times)
        
        stats = {
            'total_inferences': len(times),
            'mean_time': float(np.mean(times)),
            'median_time': float(np.median(times)),
            'min_time': float(np.min(times)),
            'max_time': float(np.max(times)),
            'std_time': float(np.std(times)),
            'device': self.device,
            'model': self.model_name,
            'mixed_precision': self.enable_mixed_precision
        }
        
        return stats
    
    def reset_performance_stats(self):
        """
        Reset performance statistics
        """
        self.inference_times.clear()
        self.logger.info("Performance statistics reset")
    
    def __del__(self):
        """
        Cleanup when engine is destroyed
        """
        try:
            self.clear_cache()
        except:
            pass
