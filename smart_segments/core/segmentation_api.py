"""
Segmentation API
High-level API for SAM2 segmentation with click coordinate handling
"""

import logging
import numpy as np
from typing import Optional, Tuple, List, Dict, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from .inference_engine import SAM2InferenceEngine


class ClickType(Enum):
    """Types of clicks for segmentation"""
    POSITIVE = "positive"  # Foreground click
    NEGATIVE = "negative"  # Background click


class SelectionMode(Enum):
    """Selection modes for multi-selection logic"""
    SINGLE = "single"      # Single click selection
    MULTI = "multi"        # Multi-selection mode (shift+click)
    REPLACE = "replace"    # Replace current selection


@dataclass
class ClickEvent:
    """Represents a user click event"""
    x: int
    y: int
    click_type: ClickType
    selection_mode: SelectionMode
    timestamp: float = field(default_factory=lambda: __import__('time').time())


@dataclass
class SegmentationResult:
    """Result of segmentation operation"""
    mask: np.ndarray
    confidence: float
    click_events: List[ClickEvent]
    processing_time: float
    model_used: str
    image_shape: Tuple[int, int]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'mask': self.mask.tolist() if self.mask is not None else None,
            'confidence': self.confidence,
            'click_events': [
                {
                    'x': click.x,
                    'y': click.y,
                    'click_type': click.click_type.value,
                    'selection_mode': click.selection_mode.value,
                    'timestamp': click.timestamp
                }
                for click in self.click_events
            ],
            'processing_time': self.processing_time,
            'model_used': self.model_used,
            'image_shape': self.image_shape
        }


class SegmentationSession:
    """Manages a segmentation session with click history and state"""
    
    def __init__(self, image: Union[np.ndarray, str, Path], session_id: Optional[str] = None):
        """
        Initialize segmentation session
        
        Args:
            image: Input image for segmentation
            session_id: Optional session identifier
        """
        self.session_id = session_id or f"session_{int(__import__('time').time())}"
        self.image = image
        self.click_history: List[ClickEvent] = []
        self.current_mask: Optional[np.ndarray] = None
        self.mask_history: List[np.ndarray] = []
        self.created_at = __import__('time').time()
        
        # Image properties
        self.image_shape: Optional[Tuple[int, int]] = None
        
    def add_click(self, click_event: ClickEvent):
        """Add click event to session history"""
        self.click_history.append(click_event)
    
    def get_positive_clicks(self) -> List[Tuple[int, int]]:
        """Get all positive click coordinates"""
        return [(click.x, click.y) for click in self.click_history 
                if click.click_type == ClickType.POSITIVE]
    
    def get_negative_clicks(self) -> List[Tuple[int, int]]:
        """Get all negative click coordinates"""
        return [(click.x, click.y) for click in self.click_history 
                if click.click_type == ClickType.NEGATIVE]
    
    def clear_history(self):
        """Clear click and mask history"""
        self.click_history.clear()
        self.mask_history.clear()
        self.current_mask = None
    
    def undo_last_click(self) -> bool:
        """Remove last click event"""
        if self.click_history:
            self.click_history.pop()
            return True
        return False


class SegmentationAPI:
    """
    High-level segmentation API with multi-selection logic
    """
    
    def __init__(self, 
                 model_name: str = 'sam2_hiera_large',
                 device: Optional[str] = None,
                 max_sessions: int = 10):
        """
        Initialize segmentation API
        
        Args:
            model_name: SAM2 model variant to use
            device: Device for inference ('cuda', 'cpu', or auto)
            max_sessions: Maximum number of concurrent sessions
        """
        self.logger = logging.getLogger(__name__)
        
        # Initialize inference engine
        self.inference_engine = SAM2InferenceEngine(model_name, device)
        self.model_name = model_name
        
        # Session management
        self.sessions: Dict[str, SegmentationSession] = {}
        self.max_sessions = max_sessions
        self.current_session_id: Optional[str] = None
        
        # API statistics
        self.total_segmentations = 0
        self.successful_segmentations = 0
        
        self.logger.info(f"Segmentation API initialized with {model_name}")
    
    def create_session(self, 
                      image: Union[np.ndarray, str, Path], 
                      session_id: Optional[str] = None) -> str:
        """
        Create new segmentation session
        
        Args:
            image: Input image
            session_id: Optional custom session ID
            
        Returns:
            Session ID
        """
        # Clean up old sessions if at limit
        if len(self.sessions) >= self.max_sessions:
            oldest_session = min(self.sessions.keys(), 
                               key=lambda k: self.sessions[k].created_at)
            self.close_session(oldest_session)
        
        # Create new session
        session = SegmentationSession(image, session_id)
        session_id = session.session_id
        
        # Set image in inference engine
        if not self.inference_engine.set_image(image):
            raise RuntimeError("Failed to set image in inference engine")
        
        # Store image shape
        if isinstance(image, np.ndarray):
            session.image_shape = image.shape[:2]
        else:
            # Will be set when image is processed
            pass
        
        self.sessions[session_id] = session
        self.current_session_id = session_id
        
        self.logger.info(f"Created session {session_id}")
        return session_id
    
    def get_session(self, session_id: str) -> Optional[SegmentationSession]:
        """Get session by ID"""
        return self.sessions.get(session_id)
    
    def close_session(self, session_id: str) -> bool:
        """Close and remove session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            if self.current_session_id == session_id:
                self.current_session_id = None
            self.logger.info(f"Closed session {session_id}")
            return True
        return False
    
    def segment_single_click(self, 
                           x: int, 
                           y: int, 
                           is_positive: bool = True,
                           session_id: Optional[str] = None) -> Optional[SegmentationResult]:
        """
        Perform segmentation with single click
        
        Args:
            x: X coordinate of click
            y: Y coordinate of click
            is_positive: Whether this is a positive (foreground) click
            session_id: Session ID (uses current if None)
            
        Returns:
            Segmentation result or None if failed
        """
        session_id = session_id or self.current_session_id
        if not session_id or session_id not in self.sessions:
            self.logger.error(f"Invalid session ID: {session_id}")
            return None
        
        session = self.sessions[session_id]
        
        # Create click event
        click_event = ClickEvent(
            x=x, y=y,
            click_type=ClickType.POSITIVE if is_positive else ClickType.NEGATIVE,
            selection_mode=SelectionMode.SINGLE
        )
        
        return self._process_segmentation(session, click_event)
    
    def segment_multi_click(self, 
                          x: int, 
                          y: int, 
                          is_positive: bool = True,
                          replace_selection: bool = False,
                          session_id: Optional[str] = None) -> Optional[SegmentationResult]:
        """
        Perform segmentation with multi-click logic (shift+click behavior)
        
        Args:
            x: X coordinate of click
            y: Y coordinate of click
            is_positive: Whether this is a positive (foreground) click
            replace_selection: Whether to replace current selection
            session_id: Session ID (uses current if None)
            
        Returns:
            Segmentation result or None if failed
        """
        session_id = session_id or self.current_session_id
        if not session_id or session_id not in self.sessions:
            self.logger.error(f"Invalid session ID: {session_id}")
            return None
        
        session = self.sessions[session_id]
        
        # Determine selection mode
        if replace_selection:
            selection_mode = SelectionMode.REPLACE
            session.clear_history()  # Clear previous clicks
        else:
            selection_mode = SelectionMode.MULTI
        
        # Create click event
        click_event = ClickEvent(
            x=x, y=y,
            click_type=ClickType.POSITIVE if is_positive else ClickType.NEGATIVE,
            selection_mode=selection_mode
        )
        
        return self._process_segmentation(session, click_event)
    
    def _process_segmentation(self, 
                            session: SegmentationSession, 
                            click_event: ClickEvent) -> Optional[SegmentationResult]:
        """
        Process segmentation for a session and click event
        
        Args:
            session: Segmentation session
            click_event: Click event to process
            
        Returns:
            Segmentation result or None if failed
        """
        import time
        start_time = time.time()
        
        try:
            self.total_segmentations += 1
            
            # Add click to session
            session.add_click(click_event)
            
            # Get all positive and negative clicks
            positive_points = session.get_positive_clicks()
            negative_points = session.get_negative_clicks()
            
            # Perform prediction
            if len(positive_points) == 1 and len(negative_points) == 0:
                # Single positive point
                result = self.inference_engine.predict_single_point(
                    positive_points[0][0], positive_points[0][1], 
                    is_positive=True, multimask_output=True
                )
            else:
                # Multi-point prediction
                result = self.inference_engine.predict_multi_points(
                    positive_points, negative_points, multimask_output=False
                )
            
            if result is None:
                self.logger.error("Inference engine returned no result")
                return None
            
            # Get best mask
            mask = self.inference_engine.get_best_mask(result)
            if mask is None:
                self.logger.error("No valid mask in prediction result")
                return None
            
            # Calculate confidence (use max score if available)
            confidence = float(np.max(result.get('scores', [0.0])))
            
            # Update session
            session.current_mask = mask
            session.mask_history.append(mask.copy())
            
            # Set image shape if not already set
            if session.image_shape is None:
                session.image_shape = mask.shape
            
            # Create result
            processing_time = time.time() - start_time
            segmentation_result = SegmentationResult(
                mask=mask,
                confidence=confidence,
                click_events=session.click_history.copy(),
                processing_time=processing_time,
                model_used=self.model_name,
                image_shape=session.image_shape
            )
            
            self.successful_segmentations += 1
            self.logger.info(f"Segmentation completed in {processing_time:.3f}s")
            
            return segmentation_result
            
        except Exception as e:
            self.logger.error(f"Segmentation failed: {e}")
            return None
    
    def undo_last_click(self, session_id: Optional[str] = None) -> bool:
        """
        Undo last click in session
        
        Args:
            session_id: Session ID (uses current if None)
            
        Returns:
            True if click was undone
        """
        session_id = session_id or self.current_session_id
        if not session_id or session_id not in self.sessions:
            return False
        
        session = self.sessions[session_id]
        if session.undo_last_click():
            # Re-run segmentation with remaining clicks
            if session.click_history:
                last_click = session.click_history[-1]
                self._process_segmentation(session, last_click)
            else:
                session.current_mask = None
            return True
        
        return False
    
    def clear_session(self, session_id: Optional[str] = None) -> bool:
        """
        Clear all clicks and masks in session
        
        Args:
            session_id: Session ID (uses current if None)
            
        Returns:
            True if session was cleared
        """
        session_id = session_id or self.current_session_id
        if not session_id or session_id not in self.sessions:
            return False
        
        session = self.sessions[session_id]
        session.clear_history()
        self.logger.info(f"Cleared session {session_id}")
        return True
    
    def get_current_mask(self, session_id: Optional[str] = None) -> Optional[np.ndarray]:
        """
        Get current mask for session
        
        Args:
            session_id: Session ID (uses current if None)
            
        Returns:
            Current mask or None
        """
        session_id = session_id or self.current_session_id
        if not session_id or session_id not in self.sessions:
            return None
        
        return self.sessions[session_id].current_mask
    
    def get_session_info(self, session_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get information about session
        
        Args:
            session_id: Session ID (uses current if None)
            
        Returns:
            Session information dictionary
        """
        session_id = session_id or self.current_session_id
        if not session_id or session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        
        return {
            'session_id': session.session_id,
            'image_shape': session.image_shape,
            'click_count': len(session.click_history),
            'positive_clicks': len(session.get_positive_clicks()),
            'negative_clicks': len(session.get_negative_clicks()),
            'has_mask': session.current_mask is not None,
            'created_at': session.created_at
        }
    
    def segment_everything(self, session_id: Optional[str] = None) -> Optional[List[Dict[str, Any]]]:
        """
        Automatically segment all objects in the current image
        
        Args:
            session_id: Session ID (uses current if None)
            
        Returns:
            List of mask dictionaries, each containing segmentation mask and metadata
        """
        session_id = session_id or self.current_session_id
        if not session_id or session_id not in self.sessions:
            self.logger.error(f"Invalid session ID: {session_id}")
            return None
            
        try:
            self.total_segmentations += 1
            
            # Use inference engine to generate all masks
            mask_results = self.inference_engine.predict_everything()
            
            if mask_results is None:
                self.logger.error("Failed to generate automatic masks")
                return None
                
            self.successful_segmentations += 1
            self.logger.info(f"Successfully generated {len(mask_results)} masks")
            
            return mask_results
            
        except Exception as e:
            self.logger.error(f"Segment everything failed: {e}")
            return None
    
    def get_api_stats(self) -> Dict[str, Any]:
        """
        Get API usage statistics
        
        Returns:
            Statistics dictionary
        """
        stats = {
            'total_segmentations': self.total_segmentations,
            'successful_segmentations': self.successful_segmentations,
            'success_rate': (self.successful_segmentations / max(1, self.total_segmentations)) * 100,
            'active_sessions': len(self.sessions),
            'max_sessions': self.max_sessions,
            'model_name': self.model_name
        }
        
        # Add inference engine stats
        engine_stats = self.inference_engine.get_performance_stats()
        stats.update({f'engine_{k}': v for k, v in engine_stats.items()})
        
        return stats
    
    def clear_cache(self):
        """Clear all caches and free memory"""
        self.inference_engine.clear_cache()
        self.logger.info("API cache cleared")
    
    def shutdown(self):
        """Shutdown API and cleanup resources"""
        # Close all sessions
        for session_id in list(self.sessions.keys()):
            self.close_session(session_id)
        
        # Clear caches
        self.clear_cache()
        
        self.logger.info("Segmentation API shutdown")
