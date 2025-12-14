#!/usr/bin/env python3
"""
Download and prepare COCO dataset for zero-shot evaluation
"""

import os
import requests
import zipfile
from tqdm import tqdm
import json

def download_coco():
    """Download COCO 2017 dataset"""
    
    print("Setting up COCO dataset for zero-shot evaluation...")
    
    # Create directories
    os.makedirs('phase3/datasets/coco', exist_ok=True)
    os.makedirs('phase3/datasets/coco/images', exist_ok=True)
    os.makedirs('phase3/datasets/coco/annotations', exist_ok=True)
    
    # URLs for COCO 2017
    urls = {
        'train_images': 'http://images.cocodataset.org/zips/train2017.zip',
        'val_images': 'http://images.cocodataset.org/zips/val2017.zip',
        'annotations': 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
    }
    
    # For testing, we'll just download a small subset
    print("For testing, we'll use a small sample. For full dataset:")
    print("Visit: https://cocodataset.org/#download")
    
    # Create sample dataset for testing
    create_sample_dataset()
    
def create_sample_dataset():
    """Create a small sample dataset for testing"""
    
    print("\nCreating sample dataset for quick testing...")
    
    sample_dir = 'phase3/datasets/coco_sample'
    os.makedirs(sample_dir, exist_ok=True)
    
    # Sample images (we'll use our test image + a few others)
    sample_images = ['test_image.jpg']  # Our existing test image
    
    # Create sample annotations
    sample_annotations = {
        "info": {"description": "COCO Sample Dataset"},
        "images": [{"id": 1, "file_name": "test_image.jpg", "width": 640, "height": 480}],
        "categories": [
            {"id": 1, "name": "person", "supercategory": "person"},
            {"id": 2, "name": "bicycle", "supercategory": "vehicle"},
            {"id": 3, "name": "car", "supercategory": "vehicle"},
            {"id": 5, "name": "bus", "supercategory": "vehicle"},
            {"id": 9, "name": "stop sign", "supercategory": "outdoor"}
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 5, "bbox": [22, 231, 783, 525]},  # bus
            {"id": 2, "image_id": 1, "category_id": 1, "bbox": [48, 398, 197, 504]},   # person
            {"id": 3, "image_id": 1, "category_id": 1, "bbox": [221, 405, 123, 452]},  # person
            {"id": 4, "image_id": 1, "category_id": 9, "bbox": [0, 550, 63, 323]}      # stop sign
        ]
    }
    
    # Save annotations
    with open(f'{sample_dir}/annotations.json', 'w') as f:
        json.dump(sample_annotations, f, indent=2)
    
    print(f"Sample dataset created at: {sample_dir}")
    print(f"Categories: {[c['name'] for c in sample_annotations['categories']]}")
    
    return sample_dir

def get_zero_shot_split():
    """Split COCO classes into seen and unseen for zero-shot evaluation"""
    
    # All COCO classes (80 classes)
    coco_classes = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 
        'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 
        'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 
        'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 
        'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 
        'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 
        'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 
        'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 
        'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    
    # Split for zero-shot evaluation (60 seen, 20 unseen)
    seen_classes = coco_classes[:60]
    unseen_classes = coco_classes[60:]
    
    print("\nZero-Shot Split:")
    print(f"Seen classes: {len(seen_classes)} (first 60)")
    print(f"Unseen classes: {len(unseen_classes)} (last 20)")
    print(f"Unseen examples: {unseen_classes[:5]}...")
    
    return seen_classes, unseen_classes

if __name__ == "__main__":
    download_coco()
    seen, unseen = get_zero_shot_split()
    
    print("\n✅ COCO dataset setup complete!")
    print("Ready for zero-shot evaluation!")