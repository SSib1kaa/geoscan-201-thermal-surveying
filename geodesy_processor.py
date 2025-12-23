#!/usr/bin/env python3
"""
Geodesy Data Processor for Geoscan 201
Processes thermal/thermal imagery for mapping, surveying, and GIS applications.
Includes orthorectification, georeferencing, and thermal mosaic generation.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple
import math


@dataclass
class GeoPoint:
    """Geographic coordinate point"""
    latitude: float
    longitude: float
    altitude: float
    accuracy: float = 5.0  # meters


@dataclass
class SurveyMetadata:
    """Survey mission metadata"""
    mission_name: str
    site_location: GeoPoint
    survey_date: str
    altitude_agl: float  # Above Ground Level
    flight_speed: float  # m/s
    focal_length: float  # mm
    sensor_width: float  # mm
    camera_angle: float  # degrees


class GeoscanSurveyProcessor:
    """Process thermal survey data from Geoscan 201 UAV"""
    
    def __init__(self, metadata: SurveyMetadata):
        self.metadata = metadata
        self.images = []
        self.camera_calibration = None
        self.orthorectified_mosaic = None
        
        # Ground resolution (GSD - Ground Sample Distance)
        self.gsd = self.calculate_gsd()
    
    def calculate_gsd(self) -> float:
        """
        Calculate Ground Sample Distance (pixel resolution at ground level).
        GSD = (altitude * sensor_width) / (focal_length * image_width)
        """
        image_width = 640  # pixels
        gsd = (self.metadata.altitude_agl * self.metadata.sensor_width) / (
            self.metadata.focal_length * image_width
        )
        return gsd  # cm per pixel
    
    def calculate_footprint(self) -> Tuple[float, float]:
        """Calculate image footprint size at ground level (width, height)"""
        image_width = 640
        image_height = 480
        
        footprint_width = image_width * self.gsd / 100  # convert to meters
        footprint_height = image_height * self.gsd / 100
        
        return footprint_width, footprint_height
    
    def geographic_to_projected(self, lat: float, lon: float, 
                                zone: int = 37) -> Tuple[float, float]:
        """
        Convert geographic coordinates (lat/lon) to UTM projected coordinates.
        Uses WGS84 datum and UTM projection.
        """
        # Simplified UTM conversion (real implementation would use pyproj)
        # UTM zone 37 for Moscow region
        lon_origin = (zone - 1) * 6 - 180 + 3
        
        # Transverse Mercator projection
        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)
        lon_origin_rad = math.radians(lon_origin)
        
        # Earth radius
        R = 6371000  # meters
        
        # Simplified projection
        easting = R * (lon_rad - lon_origin_rad) * math.cos(lat_rad)
        northing = R * lat_rad
        
        return easting, northing
    
    def orthorectify_image(self, image: np.ndarray, image_pose: dict) -> np.ndarray:
        """
        Orthorectify thermal image using camera pose and DEM data.
        
        Args:
            image: Input thermal image
            image_pose: Dictionary with 'position' (GPS), 'roll', 'pitch', 'yaw'
        
        Returns:
            Orthorectified image
        """
        # Simplified orthorectification
        # Real implementation would use more sophisticated methods
        
        height, width = image.shape[:2]
        
        # Create output image (projected ground coordinates)
        ortho = np.zeros_like(image)
        
        # Get camera angles
        pitch = image_pose.get('pitch', 0)
        roll = image_pose.get('roll', 0)
        yaw = image_pose.get('yaw', 0)
        
        # Apply perspective correction based on camera angle
        # This is simplified; real orthorectification is more complex
        center_x, center_y = width // 2, height // 2
        
        for y in range(height):
            for x in range(width):
                # Account for camera pitch and roll
                corrected_x = int(x + (y - center_y) * math.tan(math.radians(pitch)) / 10)
                corrected_y = int(y + (x - center_x) * math.tan(math.radians(roll)) / 10)
                
                if 0 <= corrected_x < width and 0 <= corrected_y < height:
                    ortho[corrected_y, corrected_x] = image[y, x]
        
        return ortho
    
    def create_thermal_mosaic(self, images: List[dict]) -> np.ndarray:
        """
        Create thermal mosaic from multiple orthorectified images.
        
        Args:
            images: List of dicts with 'image' and 'pose' keys
        
        Returns:
            Mosaic image
        """
        if not images:
            return None
        
        # Get dimensions
        footprint_w, footprint_h = self.calculate_footprint()
        
        # Create mosaic canvas
        # Size in meters / GSD to get pixel dimensions
        mosaic_width = int(footprint_w * len(images) / (self.gsd / 100))
        mosaic_height = int(footprint_h / (self.gsd / 100))
        
        mosaic = np.zeros((mosaic_height, mosaic_width, 3), dtype=np.uint8)
        
        # Place each orthorectified image
        for idx, img_data in enumerate(images):
            ortho = self.orthorectify_image(img_data['image'], img_data['pose'])
            
            x_offset = int(idx * footprint_w / (self.gsd / 100))
            h, w = ortho.shape[:2]
            
            if x_offset + w <= mosaic_width and h <= mosaic_height:
                mosaic[0:h, x_offset:x_offset+w] = ortho
        
        return mosaic
    
    def calculate_coverage_area(self, num_images: int, flight_pattern: str = 'grid') -> float:
        """
        Calculate total survey coverage area.
        
        Args:
            num_images: Number of images taken
            flight_pattern: 'grid' or 'spiral'
        
        Returns:
            Coverage area in square kilometers
        """
        footprint_w, footprint_h = self.calculate_footprint()
        
        if flight_pattern == 'grid':
            # Assume grid pattern with 30% overlap
            area = num_images * footprint_w * footprint_h * 0.7
        else:  # spiral
            # Spiral pattern covers less area per image
            area = num_images * footprint_w * footprint_h * 0.5
        
        return area / 1e6  # convert to km^2
    
    def detect_thermal_anomalies(self, thermal_image: np.ndarray, 
                                 threshold_percent: float = 90.0) -> List[dict]:
        """
        Detect thermal anomalies in survey image.
        Useful for infrastructure inspection, heat loss detection, etc.
        """
        # Convert to grayscale for thermal processing
        if len(thermal_image.shape) == 3:
            gray = np.mean(thermal_image, axis=2)
        else:
            gray = thermal_image
        
        # Find pixels above threshold percentile
        threshold_value = np.percentile(gray, threshold_percent)
        anomalies = np.where(gray > threshold_value)
        
        # Cluster nearby anomalies
        anomaly_points = []
        for y, x in zip(anomalies[0], anomalies[1]):
            anomaly_points.append({
                'x': x,
                'y': y,
                'intensity': float(gray[y, x]),
                'normalized_intensity': float(gray[y, x] / 255.0)
            })
        
        return anomaly_points
    
    def export_geotiff(self, image: np.ndarray, output_path: str, 
                       bounds: dict):
        """
        Export image as GeoTIFF with georeferencing information.
        
        Args:
            image: Image array
            output_path: Output file path
            bounds: Dict with 'north', 'south', 'east', 'west' in WGS84
        """
        # In real implementation, would use rasterio library
        # This is a simplified version showing the concept
        print(f"Exporting GeoTIFF to {output_path}")
        print(f"Geographic bounds: {bounds}")
        print(f"GSD: {self.gsd:.2f} cm/pixel")
    
    def generate_orthomosaic_report(self) -> dict:
        """Generate orthomosaic processing report"""
        report = {
            'mission_name': self.metadata.mission_name,
            'gsd': self.gsd,
            'footprint': self.calculate_footprint(),
            'flight_altitude': self.metadata.altitude_agl,
            'camera_angle': self.metadata.camera_angle,
            'processing_status': 'completed'
        }
        return report
