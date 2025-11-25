#!/usr/bin/env python3
"""
Phase 1 Test Script
Tests basic YOLO detection and CLIP classification functionality
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from yolo_detector import YOLODetector
from clip_classifier import CLIPClassifier
import requests
from PIL import Image
import matplotlib.pyplot as plt

def main():
    print("=== Phase 1: Foundation Test ===")
    
    # Initialize models
    print("1. Initializing models...")
    yolo = YOLODetector('yolov8n.pt')  # Using nano for speed
    clip = CLIPClassifier()
    
    # Test image
    test_url = "https://ultralytics.com/images/bus.jpg"
    
    # Download test image
    print("2. Downloading test image...")
    img_data = requests.get(test_url).content
    with open('test_image.jpg', 'wb') as handler:
        handler.write(img_data)
    
    # YOLO detection
    print("3. Running YOLO detection...")
    results = yolo.detect('test_image.jpg')
    boxes, scores, class_ids = yolo.extract_detections(results)
    
    print(f"   Detected {len(boxes)} objects")
    
    # CLIP classification test
    print("4. Testing CLIP classification...")
    image = Image.open('test_image.jpg')
    text_prompts = ["a bus", "a car", "a person", "a street", "a building"]
    
    # Test CLIP on full image
    clip_results = clip.classify(image, text_prompts)
    print(f"   Full image classified as: '{clip_results['best_class']}'")
    
    # Test on first detection crop
    if len(boxes) > 0:
        x1, y1, x2, y2 = boxes[0].astype(int)
        crop = image.crop((x1, y1, x2, y2))
        crop_results = clip.classify(crop, text_prompts)
        print(f"   First detection classified as: '{crop_results['best_class']}'")
    
    # Save visualization
    print("5. Saving visualizations...")
    os.makedirs('results/visualizations', exist_ok=True)
    yolo.visualize_detections('test_image.jpg', results, 'results/visualizations/yolo_detections.jpg')
    clip.visualize_similarity(clip_results['scores'], text_prompts, 'results/visualizations/clip_similarity.jpg')
    
    print("\n✅ Phase 1 completed successfully!")
    print("   - YOLO detector is working")
    print("   - CLIP classifier is working") 
    print("   - Basic integration tested")
    print("   - Visualizations saved to results/visualizations/")

if __name__ == "__main__":
    main()