#!/usr/bin/env python3
"""
Quick fix for CLIP confidence issue
Testing different prompt strategies
"""

import sys
sys.path.append('src')

from src.pipeline import ZeroShotDetectionPipeline
import matplotlib.pyplot as plt
import numpy as np

def test_prompt_strategies():
    """Test different prompt engineering strategies"""
    
    print("Testing CLIP Confidence with Different Prompt Strategies")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = ZeroShotDetectionPipeline(yolo_model='yolov8n.pt')
    
    # Test strategies
    strategies = {
        'Minimal': ["bus", "person", "stop sign"],
        'With Article': ["a bus", "a person", "a stop sign"],
        'Simple Template': ["photo of bus", "photo of person", "photo of stop sign"],
        'Full Template': ["a photo of a bus", "a photo of a person", "a photo of a stop sign"],
        'Multiple Templates': [
            "a photo of a bus", "picture of a bus", "image of a bus",
            "a photo of a person", "picture of a person", "image of a person",
            "a photo of a stop sign", "picture of a stop sign", "image of a stop sign"
        ]
    }
    
    results = {}
    
    for strategy_name, prompts in strategies.items():
        print(f"\nStrategy: {strategy_name}")
        print(f"Prompts: {prompts[:3]}...")  # Show first 3
        
        # Run detection
        results = pipeline.detect('test_image.jpg', prompts)
        
        if results['detections']:
            avg_confidence = np.mean([d['clip_confidence'] for d in results['detections']])
            print(f"Average CLIP Confidence: {avg_confidence:.3f}")
            
            # Show first detection details
            first_det = results['detections'][0]
            print(f"First detection: {first_det['clip_class']} = {first_det['clip_confidence']:.3f}")
    
    print("\n" + "=" * 60)
    print("CLIP Confidence Test Complete!")

def optimize_pipeline():
    """Create optimized pipeline with best prompts"""
    
    print("\nCreating Optimized Pipeline...")
    
    # Best strategy from testing
    optimized_prompts = ["a photo of a bus", "a photo of a person", "a photo of a stop sign"]
    
    class OptimizedPipeline(ZeroShotDetectionPipeline):
        def _engineer_prompts(self, text_prompts, template):
            """Use only the most effective prompts"""
            engineered = []
            for prompt in text_prompts:
                engineered.append(f"a photo of a {prompt}")
                engineered.append(f"a picture of a {prompt}")
            return engineered
    
    # Test optimized pipeline
    pipeline = OptimizedPipeline(yolo_model='yolov8n.pt')
    results = pipeline.detect('test_image.jpg', ["bus", "person", "stop sign"])
    
    print(f"\nOptimized Pipeline Results:")
    for i, det in enumerate(results['detections']):
        print(f"  {i+1}. {det['clip_class']}: CLIP={det['clip_confidence']:.3f}")
    
    return pipeline

if __name__ == "__main__":
    test_prompt_strategies()
    optimized_pipeline = optimize_pipeline()
    
    print("\n✅ Ready for Phase 3: Evaluation!")