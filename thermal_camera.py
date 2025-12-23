#!/usr/bin/env python3
"""
Thermal Camera Module for Geoscan 201 UAV
Handles infrared (IR) camera, thermal image processing, and night vision
for geodesy and survey applications.
"""

import cv2
import numpy as np
import threading
import time
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class ThermalImageData:
    """Thermal image metadata and processing results"""
    timestamp: float
    raw_frame: np.ndarray
    thermal_processed: np.ndarray
    temperature_min: float
    temperature_max: float
    temperature_avg: float
    night_enhanced: Optional[np.ndarray] = None
    roi_data: Optional[dict] = None


class ThermalCamera:
    """Infrared/Thermal camera interface for Geoscan 201"""
    
    def __init__(self, camera_id: int = 0, sensor_type: str = 'MLX90640'):
        """
        Initialize thermal camera.
        
        Args:
            camera_id: Camera device ID
            sensor_type: Type of thermal sensor (MLX90640, FLIR, etc.)
        """
        self.camera_id = camera_id
        self.sensor_type = sensor_type
        self.running = False
        self.frame_rate = 30
        self.resolution = (640, 480)
        
        # Temperature calibration
        self.temp_offset = 0.0
        self.emissivity = 0.95
        
        # Night vision settings
        self.night_mode_enabled = False
        self.night_mode_threshold = 50
        
    def connect(self) -> bool:
        """Connect to thermal camera"""
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            if self.cap.isOpened():
                print(f"Thermal camera {self.camera_id} connected")
                return True
        except Exception as e:
            print(f"Error connecting to camera: {e}")
        return False
    
    def capture_thermal_frame(self) -> Optional[np.ndarray]:
        """Capture raw thermal frame"""
        if not hasattr(self, 'cap'):
            return None
            
        ret, frame = self.cap.read()
        if ret:
            return frame
        return None
    
    def apply_thermal_colormap(self, frame: np.ndarray) -> np.ndarray:
        """Apply thermal pseudocolor mapping to raw frame"""
        # Convert to grayscale and normalize
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        
        # Apply thermal colormap (Iron or Turbo for better temperature visualization)
        thermal_color = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
        return thermal_color
    
    def calculate_temperatures(self, frame: np.ndarray) -> Tuple[float, float, float]:
        """Calculate min, max, and average temperatures from frame"""
        # Simulate temperature calculation from thermal data
        # Real implementation would use actual thermal sensor calibration
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(float)
        
        # Map pixel values to temperature range (-40°C to 150°C)
        temp_min = -40 + (gray.min() / 255) * 190 + self.temp_offset
        temp_max = -40 + (gray.max() / 255) * 190 + self.temp_offset
        temp_avg = -40 + (gray.mean() / 255) * 190 + self.temp_offset
        
        return temp_min, temp_max, temp_avg
    
    def enhance_night_vision(self, frame: np.ndarray) -> np.ndarray:
        """Enhance image for night vision/low-light conditions"""
        # Convert to LAB color space for better enhancement
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge back
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # Apply slight noise reduction
        enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        return enhanced
    
    def detect_hot_regions(self, frame: np.ndarray, threshold: int = 200) -> list:
        """Detect hot regions/anomalies in thermal image"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Threshold to find hot regions
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        regions = []
        for contour in contours:
            if cv2.contourArea(contour) > 100:
                x, y, w, h = cv2.boundingRect(contour)
                regions.append({
                    'bbox': (x, y, w, h),
                    'area': cv2.contourArea(contour),
                    'center': (x + w//2, y + h//2)
                })
        
        return sorted(regions, key=lambda r: r['area'], reverse=True)
    
    def process_thermal_frame(self, frame: np.ndarray) -> ThermalImageData:
        """Process thermal frame and extract data"""
        timestamp = time.time()
        
        # Apply thermal colormap
        thermal_colored = self.apply_thermal_colormap(frame)
        
        # Calculate temperatures
        temp_min, temp_max, temp_avg = self.calculate_temperatures(frame)
        
        # Night vision enhancement (if enabled)
        night_enhanced = None
        if self.night_mode_enabled:
            night_enhanced = self.enhance_night_vision(frame)
        
        # Detect hot regions
        hot_regions = self.detect_hot_regions(thermal_colored, self.night_mode_threshold)
        
        return ThermalImageData(
            timestamp=timestamp,
            raw_frame=frame,
            thermal_processed=thermal_colored,
            temperature_min=temp_min,
            temperature_max=temp_max,
            temperature_avg=temp_avg,
            night_enhanced=night_enhanced,
            roi_data={'hot_regions': hot_regions}
        )
    
    def save_thermal_image(self, image: np.ndarray, filename: Optional[str] = None):
        """Save thermal image to disk"""
        if filename is None:
            filename = f"thermal_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        cv2.imwrite(filename, image)
        print(f"Thermal image saved: {filename}")
    
    def set_emissivity(self, value: float):
        """Set emissivity for temperature calculation (0.0 - 1.0)"""
        self.emissivity = max(0.0, min(1.0, value))
    
    def calibrate_temperature_offset(self, reference_temp: float, measured_pixel_value: int):
        """Calibrate temperature offset using reference temperature"""
        # Calculate offset based on reference
        expected_value = (-40 + (reference_temp + 40) / 190 * 255)
        self.temp_offset = reference_temp - (-40 + (measured_pixel_value / 255) * 190)
    
    def disconnect(self):
        """Disconnect from thermal camera"""
        if hasattr(self, 'cap'):
            self.cap.release()
            print("Thermal camera disconnected")


class NightVisionMode:
    """Enhanced night vision processing for low-light geodesy surveys"""
    
    def __init__(self, sensitivity: float = 0.8):
        self.sensitivity = sensitivity
        self.ir_boost = 1.5
        self.noise_reduction_level = 2
    
    def process(self, frame: np.ndarray) -> np.ndarray:
        """Apply night vision processing"""
        # Increase brightness
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 2] = hsv[:, :, 2] * (1.0 + self.sensitivity)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
        enhanced = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        # Denoise
        for _ in range(self.noise_reduction_level):
            enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
        
        return enhanced
