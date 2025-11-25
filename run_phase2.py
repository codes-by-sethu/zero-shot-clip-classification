#!/usr/bin/env python3
"""
Phase 2 Command Line Test
Run: python run_phase2.py
"""

import sys
import os
import requests
from PIL import Image

# Add src to path
sys.path.append('src')

from src.pipeline import ZeroShotDetectionPipeline

def main():
    print("=== Phase 2: Pipeline Integration Test (Command Line) ===")
    
    # 1. Initialize the integrated pipeline
    print("1. Initializing Zero-Shot Detection Pipeline...")
    pipeline = ZeroShotDetectionPipeline(yolo_model='yolov8n.pt')
    
    # 2. Download test image if not exists
    test_url = "https://ultralytics.com/images/bus.jpg"
    test_image_path = "test_image.jpg"
    
    if not os.path.exists(test_image_path):
        print("2. Downloading test image...")
        img_data = requests.get(test_url).content
        with open(test_image_path, 'wb') as handler:
            handler.write(img_data)
        print("   Download complete!")
    else:
        print("2. Test image already exists.")
    
    # 3. Define what we want to detect
    print("3. Defining detection targets...")
    detection_targets = [
        "bus", "person", "car", "traffic light", 
        "stop sign", "building", "tree", "road"
    ]
    
    print(f"   Looking for: {', '.join(detection_targets)}")
    
    # 4. Run the complete pipeline
    print("4. Running zero-shot detection pipeline...")
    print("   This may take a moment...")
    results = pipeline.detect(test_image_path, detection_targets)
    
    # 5. Print comprehensive results
    print(f"\n" + "="*50)
    print("DETECTION RESULTS")
    print("="*50)
    print(f"Image: {results['image_path']}")
    print(f"Target objects: {', '.join(results['original_prompts'])}")
    print(f"Found {len(results['detections'])} objects:\n")
    
    for i, detection in enumerate(results['detections']):
        print(f"Object {i+1}:")
        print(f"  Position: {detection['box'].astype(int)}")
        print(f"  Classification: '{detection['clip_class']}'")
        print(f"  YOLO Confidence: {detection['yolo_confidence']:.3f}")
        print(f"  CLIP Confidence: {detection['clip_confidence']:.3f}")
        print()
    
    # 6. Save detailed visualization
    print("5. Generating visualization...")
    os.makedirs('results/visualizations', exist_ok=True)
    output_path = 'results/visualizations/phase2_cmd_detections.jpg'
    pipeline.visualize_detections(results, output_path)
    
    print("\n✅ Phase 2 completed successfully!")
    print(f"   - Visualization saved to: {output_path}")
    print("   - Integrated pipeline is working")
    print("   - Ready for real zero-shot detection!")

if __name__ == "__main__":
    main()