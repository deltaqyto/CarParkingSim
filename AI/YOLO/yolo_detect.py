import os
import numpy as np
import cv2
import time
import pygame

# Add YOLO imports
from ultralytics import YOLO


class YOLODetector:
    """Unified YOLO detector that works with both pygame screens and numpy arrays"""
    
    def __init__(self, model_path, confidence_threshold=0.5):
        """Initialize YOLO detector"""
        self.model = None
        self.labels = {}
        self.confidence_threshold = confidence_threshold
        self.detection_active = False
        
        try:
            self.model = YOLO(model_path, task='detect')
            self.labels = self.model.names
            self.detection_active = True
            print(f"YOLO model loaded successfully from {model_path}")
            print(f"Available classes: {list(self.labels.values())}")
        except Exception as e:
            print(f"Failed to load YOLO model: {e}")
            print("YOLO detection will be disabled")
    
    def detect_from_pygame_screen(self, screen):
        """Run YOLO detection on pygame screen surface"""
        if not self.detection_active or self.model is None or screen is None:
            return []
        
        try:
            # Capture pygame screen and convert to array
            screen_array = self._capture_pygame_screen(screen)
            
            # Use the unified detection method
            return self._run_detection(screen_array)
            
        except Exception as e:
            print(f"Detection error: {e}")
            return []
    
    def detect_from_array(self, image_array):
        """Run YOLO detection on a numpy array (BGR format)"""
        if not self.detection_active or self.model is None:
            return []
        
        try:
            # Use the unified detection method
            return self._run_detection(image_array)
            
        except Exception as e:
            print(f"Detection error: {e}")
            return []
    
    def _run_detection(self, image_array):
        """Unified detection method that works with any BGR numpy array"""
        try:
            # Run YOLO inference
            results = self.model(image_array, verbose=False)
            detections = results[0].boxes
            
            detected_objects = []
            
            if detections is not None:
                for i in range(len(detections)):
                    # Extract detection data
                    xyxy_tensor = detections[i].xyxy.cpu()
                    xyxy = xyxy_tensor.numpy().squeeze()
                    
                    confidence = detections[i].conf.item()
                    
                    if confidence < self.confidence_threshold:
                        continue
                    
                    class_id = int(detections[i].cls.item())
                    class_name = self.labels.get(class_id, f"class_{class_id}")
                    
                    # Store detection info
                    detection_info = {
                        'bbox': xyxy.astype(int),  # [xmin, ymin, xmax, ymax]
                        'confidence': confidence,
                        'class_id': class_id,
                        'class_name': class_name,
                        'center': ((xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2),
                        'width': xyxy[2] - xyxy[0],
                        'height': xyxy[3] - xyxy[1]
                    }
                    
                    detected_objects.append(detection_info)
            
            return detected_objects
            
        except Exception as e:
            print(f"YOLO inference error: {e}")
            return []
    
    def _capture_pygame_screen(self, screen):
        """
        Capture the current Pygame screen as a numpy array
        
        Args:
            screen (pygame.Surface): Pygame screen surface
        
        Returns:
            numpy.ndarray: Captured screen as a numpy array in BGR format
        """
        # Convert Pygame surface to numpy array
        screen_array = pygame.surfarray.array3d(screen)
        
        # Transpose to get standard image format (Height, Width, Channels)
        screen_array = screen_array.transpose([1, 0, 2])
        
        # Convert from RGB to BGR (OpenCV default)
        screen_array = cv2.cvtColor(screen_array, cv2.COLOR_RGB2BGR)
        
        return screen_array
    
    def toggle_detection(self):
        """Toggle detection on/off"""
        if self.model is not None:
            self.detection_active = not self.detection_active
            return self.detection_active
        return False
    
    def is_active(self):
        """Check if detection is active"""
        return self.detection_active and self.model is not None