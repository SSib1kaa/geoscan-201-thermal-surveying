
def example_thermal_camera():
    """Example: Initialize and use thermal camera"""
    print("=" * 50)
    print("Example 1: Thermal Camera Setup")
    print("=" * 50)
    
    camera = ThermalCamera(camera_id=0, sensor_type='MLX90640')
    
    if camera.connect():
        print("✓ Thermal camera connected successfully")
        
        camera.set_emissivity(0.95)
        print("✓ Emissivity set to 0.95")
        
        camera.night_mode_enabled = True
        print("✓ Night vision mode enabled")
        
        dummy_frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        
        result = camera.process_thermal_frame(dummy_frame)
        
        print(f"\nThermal Data:")
        print(f"  Temperature Range: {result.temperature_min:.1f}°C to {result.temperature_max:.1f}°C")
        print(f"  Average Temperature: {result.temperature_avg:.1f}°C")
        print(f"  Hot Regions Detected: {len(result.roi_data['hot_regions'])}")
        
        camera.disconnect()
        print("\n✓ Camera disconnected")


def example_night_vision():
    """Example: Night vision image enhancement"""
    print("\n" + "=" * 50)
    print("Example 2: Night Vision Mode")
    print("=" * 50)

    night_vision = NightVisionMode(sensitivity=0.8)
    print("✓ Night vision mode initialized")
    print(f"  Sensitivity: {night_vision.sensitivity}")
    print(f"  IR Boost: {night_vision.ir_boost}x")
    print(f"  Noise Reduction Level: {night_vision.noise_reduction_level}")
    
    low_light_image = np.random.randint(30, 100, (480, 640, 3), dtype=np.uint8)
    
    print("✓ Low-light image captured")
    print(f"  Brightness range: {low_light_image.min()}-{low_light_image.max()}")
    print("\n⏳ Applying night vision enhancement...")
    enhanced = night_vision.process(low_light_image)
    print("✓ Enhancement complete")
    print(f"  Enhanced brightness range: {enhanced.min()}-{enhanced.max()}")


def example_geodesy_survey():
    """Example: Geodesy survey processing"""
    print("\n" + "=" * 50)
    print("Example 3: Geodesy Survey Processing")
    print("=" * 50)

    survey_location = GeoPoint(
        latitude=55.7558,
        longitude=37.6173,
        altitude=100
    )
    
    metadata = SurveyMetadata(
        mission_name="Moscow_Building_Survey_2024",
        site_location=survey_location,
        survey_date="2024-12-23",
        altitude_agl=100,  # 100 meters above ground
        flight_speed=10,   # m/s
        focal_length=4.0,  # mm
        sensor_width=6.0,  # mm
        camera_angle=45    # 45 degrees nadir
    )
    
    processor = GeoscanSurveyProcessor(metadata)
    
    print(f"✓ Survey initialized: {metadata.mission_name}")
    print(f"\nSurvey Parameters:")
    print(f"  Location: {survey_location.latitude}, {survey_location.longitude}")
    print(f"  Altitude AGL: {metadata.altitude_agl} meters")
    print(f"  Flight Speed: {metadata.flight_speed} m/s")
    
    print(f"\nCalculated Parameters:")
    print(f"  Ground Sample Distance (GSD): {processor.gsd:.2f} cm/pixel")
    
    footprint_w, footprint_h = processor.calculate_footprint()
    print(f"  Image Footprint: {footprint_w:.1f}m × {footprint_h:.1f}m")
    
    num_images = 100
    coverage = processor.calculate_coverage_area(num_images, flight_pattern='grid')
    print(f"  Coverage Area ({num_images} images, grid pattern): {coverage:.2f} km²")
   
    sample_thermal_image = np.random.randint(50, 200, (480, 640), dtype=np.uint8)
    anomalies = processor.detect_thermal_anomalies(sample_thermal_image, threshold_percent=90)
    print(f"\nThermal Anomalies Detected: {len(anomalies)}")
    
    report = processor.generate_orthomosaic_report()
    print(f"\nOrthomosaic Report:")
    print(f"  Status: {report['processing_status']}")
    print(f"  GSD: {report['gsd']:.2f} cm/pixel")


def main():
    """Run all examples"""
    print("\n")
    print("╔═" + "═" * 48 + "═╗")
    print("║  Geoscan 201 Thermal & Night Vision System  ║")
    print("║      Example Usage & Demonstrations         ║")
    print("╚═" + "═" * 48 + "═╝")
    
    try:
        example_thermal_camera()
        example_night_vision()
        example_geodesy_survey()
        
        print("\n" + "=" * 50)
        print("✓ All examples completed successfully!")
        print("=" * 50 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
