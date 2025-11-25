import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from .yolo_detector import YOLODetector
from .clip_classifier import CLIPClassifier

class ZeroShotDetectionPipeline:
    def __init__(self, yolo_model='yolov8l.pt', clip_model='ViT-B-32', conf_threshold=0.25):
        """
        Integrated pipeline for zero-shot object detection
        
        Args:
            yolo_model: YOLO model type
            clip_model: CLIP model type  
            conf_threshold: Detection confidence threshold
        """
        self.yolo_detector = YOLODetector(yolo_model, conf_threshold)
        self.clip_classifier = CLIPClassifier(clip_model)
        self.conf_threshold = conf_threshold
        
        print("Zero-Shot Detection Pipeline initialized!")
    
    def detect(self, image_path, text_prompts, prompt_template="a photo of a {}"):
        """
        Main detection method with improved prompt handling
        """
        # Apply prompt engineering
        engineered_prompts = self._engineer_prompts(text_prompts, prompt_template)
        
        # Get YOLO detections
        yolo_results = self.yolo_detector.detect(image_path)
        boxes, scores, yolo_class_ids = self.yolo_detector.extract_detections(yolo_results)
        
        # Filter by confidence
        valid_indices = scores >= self.conf_threshold
        boxes = boxes[valid_indices]
        scores = scores[valid_indices]
        yolo_class_ids = yolo_class_ids[valid_indices]
        
        print(f"Found {len(boxes)} valid detections")
        
        # Load image for cropping
        image = Image.open(image_path)
        
        # Process each detection with CLIP
        clip_results = []
        for i, (box, yolo_score) in enumerate(zip(boxes, scores)):
            # Crop the detection
            crop = self._crop_image(image, box)
            
            # Use ALL engineered prompts for better accuracy
            clip_result = self.clip_classifier.classify(crop, engineered_prompts)
            
            # Find the best matching object from our original targets
            best_class = "unknown"
            best_score = 0
            
            # Check which of our target objects has the highest score
            for j, target in enumerate(text_prompts):
                # Sum scores from all prompt variations for this target
                target_score = 0
                target_count = 0
                
                for k, prompt in enumerate(engineered_prompts):
                    if target in prompt:
                        target_score += clip_result['scores'][k]
                        target_count += 1
                
                if target_count > 0:
                    avg_score = target_score / target_count
                    if avg_score > best_score:
                        best_score = avg_score
                        best_class = target
            
            # If no good match found, use CLIP's best class
            if best_class == "unknown" or best_score < 0.1:
                best_class = clip_result['best_class']
                best_score = clip_result['best_score']
                # Try to extract object name from prompt
                for target in text_prompts:
                    if target in best_class:
                        best_class = target
                        break
            
            clip_results.append({
                'box': box,
                'yolo_confidence': yolo_score,
                'clip_class': best_class,
                'clip_confidence': best_score,
                'all_scores': clip_result['scores'],
                'crop': crop
            })
            
            print(f"Detection {i+1}: YOLO={yolo_score:.3f}, CLIP='{best_class}' ({best_score:.3f})")
        
        return {
            'image_path': image_path,
            'detections': clip_results,
            'text_prompts': engineered_prompts,
            'original_prompts': text_prompts,
            'yolo_results': yolo_results
        }
    
    def _engineer_prompts(self, text_prompts, template):
        """Better prompt engineering for higher CLIP confidence"""
        engineered = []
        
        # Use multiple templates for robustness
        templates = [
            "a photo of a {}",
            "a picture of a {}", 
            "an image of a {}",
            "a {} in the scene",
            "a {} in the image",
            "a close up of a {}",
            "a large {}",
            "a small {}"
        ]
        
        # Create variations for each object
        for prompt in text_prompts:
            for temp in templates[:3]:  # Use first 3 templates
                engineered.append(temp.format(prompt))
        
        # Add context-specific prompts
        engineered.extend([
            "a street scene",
            "an urban environment", 
            "outdoor photography"
        ])
        
        print(f"Generated {len(engineered)} prompts for: {text_prompts}")
        return engineered
    
    def _crop_image(self, image, box):
        """Crop image based on bounding box coordinates"""
        x1, y1, x2, y2 = box.astype(int)
        # Ensure coordinates are within image bounds
        width, height = image.size
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)
        
        return image.crop((x1, y1, x2, y2))
    
    def visualize_detections(self, results, save_path=None):
        """
        Simplified visualization that works with any number of detections
        """
        image = Image.open(results['image_path'])
        detections = results['detections']
        
        if len(detections) == 0:
            print("No detections to visualize")
            return
        
        # Create a simple 2-part figure
        fig = plt.figure(figsize=(20, 10))
        
        # Left: Main image with bounding boxes
        ax1 = plt.subplot(1, 2, 1)
        ax1.imshow(image)
        
        for i, detection in enumerate(detections):
            box = detection['box']
            x1, y1, x2, y2 = box.astype(int)
            
            # Create rectangle
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                   linewidth=2, edgecolor='red', facecolor='none')
            ax1.add_patch(rect)
            
            # Add label
            label = f"{detection['clip_class']}\nYOLO: {detection['yolo_confidence']:.2f}\nCLIP: {detection['clip_confidence']:.2f}"
            ax1.text(x1, y1-10, label, fontsize=8, color='red', 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
        
        ax1.set_title('Zero-Shot Detections\n(Rectangles show detected objects)')
        ax1.axis('off')
        
        # Right: Show first 4 crops in a 2x2 grid
        ax2 = plt.subplot(1, 2, 2)
        ax2.axis('off')  # Hide the main right subplot
        
        # Show crops in a grid on the right side
        num_crops = min(4, len(detections))
        if num_crops > 0:
            # Create a grid for crops within the right subplot
            grid = plt.GridSpec(2, 2, wspace=0.3, hspace=0.3)
            
            for i in range(num_crops):
                row = i // 2
                col = i % 2
                ax_crop = fig.add_subplot(grid[row, col])
                
                detection = detections[i]
                ax_crop.imshow(detection['crop'])
                ax_crop.set_title(f"{detection['clip_class']}\nCLIP: {detection['clip_confidence']:.3f}", 
                               fontsize=9)
                ax_crop.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300, facecolor='white')
            print(f"Visualization saved to: {save_path}")
        
        plt.show()

# Test function
def test_pipeline():
    """Test the integrated pipeline"""
    print("Testing Zero-Shot Detection Pipeline...")
    
    # Initialize pipeline
    pipeline = ZeroShotDetectionPipeline(yolo_model='yolov8n.pt')  # Use nano for speed
    
    # Test prompts - we can detect ANYTHING without training!
    text_prompts = ["bus", "person", "car", "traffic light", "stop sign"]
    
    # Test image
    test_image = "https://ultralytics.com/images/bus.jpg"
    
    # Run detection
    results = pipeline.detect(test_image, text_prompts)
    
    print(f"\n=== SUMMARY ===")
    print(f"Processed {len(results['detections'])} detections")
    for i, detection in enumerate(results['detections']):
        print(f"{i+1}. {detection['clip_class']} - YOLO: {detection['yolo_confidence']:.3f}, CLIP: {detection['clip_confidence']:.3f}")
    
    # Visualize results
    pipeline.visualize_detections(results, 'results/visualizations/pipeline_detections.jpg')

if __name__ == "__main__":
    test_pipeline()