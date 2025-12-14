import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# Handle imports for different project structures
try:
    from .yolo_detector import YOLODetector
    from .clip_classifier import CLIPClassifier
except ImportError:
    # Fallback for direct execution
    from yolo_detector import YOLODetector
    from clip_classifier import CLIPClassifier

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
    
    def detect(self, image_path, text_prompts, clip_threshold=0.05):
        """
        Main detection method with improved prompt handling
        
        Args:
            image_path: Path to image
            text_prompts: List of target object descriptions
            clip_threshold: Minimum CLIP confidence score
            
        Returns:
            Dictionary with detection results
        """
        # Apply improved prompt engineering
        engineered_prompts = self._engineer_prompts(text_prompts)
        
        # Get YOLO detections
        yolo_results = self.yolo_detector.detect(image_path)
        boxes, scores, yolo_class_ids = self.yolo_detector.extract_detections(yolo_results)
        
        # Filter by confidence
        valid_indices = scores >= self.conf_threshold
        boxes = boxes[valid_indices]
        scores = scores[valid_indices]
        yolo_class_ids = yolo_class_ids[valid_indices]
        
        print(f"Found {len(boxes)} valid YOLO detections")
        
        # Load image for cropping
        image = Image.open(image_path).convert('RGB')
        
        # Process each detection with CLIP
        clip_results = []
        for i, (box, yolo_score) in enumerate(zip(boxes, scores)):
            # Crop the detection
            crop = self._crop_image(image, box)
            
            # Use CLIP to classify with ALL engineered prompts
            clip_result = self.clip_classifier.classify(crop, engineered_prompts)
            
            # Find the best matching object from our original targets
            best_class, best_score = self._find_best_match(text_prompts, engineered_prompts, clip_result)
            
            # Apply CLIP confidence threshold
            if best_score < clip_threshold:
                # Skip low-confidence detections
                continue
            
            clip_results.append({
                'box': box.tolist() if isinstance(box, np.ndarray) else box,
                'yolo_confidence': float(yolo_score),
                'clip_class': best_class,
                'clip_confidence': float(best_score),
                'crop': crop
            })
            
            print(f"Detection {i+1}: YOLO={yolo_score:.3f}, CLIP='{best_class}' ({best_score:.3f})")
        
        # Apply Non-Maximum Suppression to remove duplicate detections
        clip_results = self._apply_nms(clip_results, iou_threshold=0.5)
        
        print(f"Final: {len(clip_results)} zero-shot detections after filtering")
        
        return {
            'image_path': image_path,
            'detections': clip_results,
            'text_prompts': text_prompts,
            'engineered_prompts': engineered_prompts,
            'num_yolo_detections': len(boxes)
        }
    
    def _engineer_prompts(self, text_prompts):
        """
        Advanced prompt engineering for higher CLIP confidence
        
        Args:
            text_prompts: List of target object descriptions
            
        Returns:
            List of engineered prompts
        """
        engineered = []
        
        for prompt in text_prompts:
            prompt_lower = prompt.lower()
            
            # Person-related prompts
            if any(word in prompt_lower for word in ['person', 'human', 'people', 'man', 'woman', 'child']):
                engineered.extend([
                    f"a photo of a {prompt}",
                    f"a picture of a {prompt}",
                    f"an image of a {prompt}",
                    f"a {prompt} walking",
                    f"a {prompt} standing",
                    f"close up of a {prompt}",
                    f"full body of a {prompt}",
                    f"{prompt} in urban environment"
                ])
            
            # Vehicle-related prompts
            elif any(word in prompt_lower for word in ['car', 'vehicle', 'bus', 'truck', 'van', 'automobile']):
                engineered.extend([
                    f"a photo of a {prompt}",
                    f"a picture of a {prompt}",
                    f"an image of a {prompt}",
                    f"a {prompt} on the road",
                    f"a {prompt} in traffic",
                    f"a {prompt} parked",
                    f"street view with {prompt}",
                    f"traffic with {prompt}"
                ])
            
            # Sign-related prompts
            elif any(word in prompt_lower for word in ['sign', 'signal', 'traffic light', 'stop sign']):
                engineered.extend([
                    f"a photo of a {prompt}",
                    f"a picture of a {prompt}",
                    f"an image of a {prompt}",
                    f"a traffic {prompt}",
                    f"a street {prompt}",
                    f"a road {prompt}",
                    f"{prompt} on pole",
                    f"traffic control {prompt}"
                ])
            
            # Building-related prompts
            elif any(word in prompt_lower for word in ['building', 'house', 'structure']):
                engineered.extend([
                    f"a photo of a {prompt}",
                    f"a picture of a {prompt}",
                    f"an image of a {prompt}",
                    f"a large {prompt}",
                    f"an urban {prompt}",
                    f"a city {prompt}",
                    f"architecture of {prompt}"
                ])
            
            # Generic prompts for other objects
            else:
                engineered.extend([
                    f"a photo of a {prompt}",
                    f"a picture of a {prompt}",
                    f"an image of a {prompt}",
                    f"a {prompt} in the scene",
                    f"a {prompt} in the image",
                    f"close up of a {prompt}",
                    f"{prompt} object"
                ])
        
        # Add context prompts
        engineered.extend([
            "street scene",
            "urban environment",
            "outdoor photography",
            "city street",
            "road traffic"
        ])
        
        # Remove duplicates and ensure reasonable length
        engineered = list(set(engineered))[:30]  # Max 30 prompts for efficiency
        
        print(f"Generated {len(engineered)} prompts for: {text_prompts}")
        return engineered
    
    def _find_best_match(self, target_prompts, engineered_prompts, clip_result):
        """
        Find the best matching target prompt for a detection
        
        Args:
            target_prompts: Original target prompts
            engineered_prompts: All engineered prompts
            clip_result: CLIP classification result
            
        Returns:
            Tuple of (best_class, best_score)
        """
        best_class = "unknown"
        best_score = 0
        
        # For each target prompt, calculate average score across its variations
        for target in target_prompts:
            target_variations = [p for p in engineered_prompts if target.lower() in p.lower()]
            
            if not target_variations:
                continue
                
            # Get indices of these variations
            indices = [engineered_prompts.index(p) for p in target_variations]
            
            # Calculate average score
            avg_score = np.mean([clip_result['scores'][i] for i in indices])
            
            if avg_score > best_score:
                best_score = avg_score
                best_class = target
        
        # If no good match found, use CLIP's original best class
        if best_class == "unknown" or best_score < 0.05:
            best_class = clip_result['best_class']
            best_score = clip_result['best_score']
            
            # Try to map CLIP's class to one of our targets
            for target in target_prompts:
                if target.lower() in best_class.lower():
                    best_class = target
                    break
        
        return best_class, best_score
    
    def _crop_image(self, image, box):
        """Crop image based on bounding box coordinates"""
        x1, y1, x2, y2 = box
        # Ensure coordinates are within image bounds
        width, height = image.size
        x1 = int(max(0, x1))
        y1 = int(max(0, y1))
        x2 = int(min(width, x2))
        y2 = int(min(height, y2))
        
        # Ensure valid crop dimensions
        if x2 <= x1 or y2 <= y1:
            return image
        
        return image.crop((x1, y1, x2, y2))
    
    def _apply_nms(self, detections, iou_threshold=0.5):
        """
        Apply Non-Maximum Suppression to remove duplicate detections
        
        Args:
            detections: List of detection dictionaries
            iou_threshold: IoU threshold for suppression
            
        Returns:
            Filtered detections
        """
        if not detections:
            return detections
        
        # Sort by CLIP confidence
        detections = sorted(detections, key=lambda x: x['clip_confidence'], reverse=True)
        
        filtered = []
        while detections:
            # Take the highest confidence detection
            current = detections.pop(0)
            filtered.append(current)
            
            # Remove overlapping detections
            to_keep = []
            for det in detections:
                iou = self._calculate_iou(current['box'], det['box'])
                if iou < iou_threshold:
                    to_keep.append(det)
            
            detections = to_keep
        
        return filtered
    
    def _calculate_iou(self, box1, box2):
        """Calculate Intersection over Union"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def visualize_detections(self, results, save_path=None):
        """
        Create comprehensive visualization of detections
        
        Args:
            results: Detection results from detect() method
            save_path: Optional path to save visualization
        """
        image = Image.open(results['image_path'])
        detections = results['detections']
        
        if len(detections) == 0:
            print("No detections to visualize")
            return
        
        # Create figure
        fig = plt.figure(figsize=(20, 10))
        
        # Left: Main image with bounding boxes
        ax1 = plt.subplot(1, 2, 1)
        ax1.imshow(image)
        
        # Color coding based on confidence
        for detection in detections:
            box = detection['box']
            x1, y1, x2, y2 = box
            
            # Color: Green for high confidence, Yellow for medium, Red for low
            conf = detection['clip_confidence']
            if conf > 0.1:
                color = 'green'
            elif conf > 0.05:
                color = 'yellow'
            else:
                color = 'red'
            
            # Create rectangle
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                   linewidth=2, edgecolor=color, facecolor='none')
            ax1.add_patch(rect)
            
            # Add label
            label = f"{detection['clip_class']}\nYOLO: {detection['yolo_confidence']:.2f}\nCLIP: {detection['clip_confidence']:.2f}"
            ax1.text(x1, y1-15, label, fontsize=8, color=color, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        ax1.set_title(f'Zero-Shot Detections ({len(detections)} objects)', fontsize=14)
        ax1.axis('off')
        
        # Right: Show crops in a grid
        ax2 = plt.subplot(1, 2, 2)
        ax2.axis('off')
        
        # Show up to 9 crops in a 3x3 grid
        num_crops = min(9, len(detections))
        if num_crops > 0:
            cols = 3
            rows = (num_crops + cols - 1) // cols
            
            grid = plt.GridSpec(rows, cols, wspace=0.3, hspace=0.5)
            
            for i in range(num_crops):
                row = i // cols
                col = i % cols
                ax_crop = fig.add_subplot(grid[row, col])
                
                detection = detections[i]
                ax_crop.imshow(detection['crop'])
                
                # Truncate long class names
                class_name = detection['clip_class']
                if len(class_name) > 15:
                    class_name = class_name[:12] + "..."
                
                ax_crop.set_title(f"{class_name}\nCLIP: {detection['clip_confidence']:.3f}", 
                               fontsize=9)
                ax_crop.axis('off')
        
        plt.suptitle('Zero-Shot Object Detection Results', fontsize=16, y=0.95)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', 
                       exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', dpi=150, facecolor='white')
            print(f"Visualization saved to: {save_path}")
        
        plt.show()
    
    def evaluate_performance(self, image_path, target_classes, yolo_as_ground_truth=True):
        """
        Evaluate pipeline performance using YOLO as pseudo-ground truth
        
        Args:
            image_path: Path to test image
            target_classes: List of target classes to detect
            yolo_as_ground_truth: Use YOLO detections as ground truth
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Run zero-shot detection
        results = self.detect(image_path, target_classes)
        
        # Get YOLO detections as ground truth
        yolo_results = self.yolo_detector.detect(image_path)
        yolo_boxes, yolo_scores, yolo_classes = self.yolo_detector.extract_detections(yolo_results)
        
        # Convert YOLO class IDs to names
        yolo_class_names = []
        if hasattr(yolo_results, 'names'):
            for class_id in yolo_classes:
                yolo_class_names.append(yolo_results.names[int(class_id)])
        
        # Calculate evaluation metrics
        metrics = self._calculate_metrics(results['detections'], yolo_boxes, yolo_class_names)
        
        return {
            'image_path': image_path,
            'target_classes': target_classes,
            'zero_shot_results': results,
            'yolo_detections': list(zip(yolo_boxes, yolo_class_names, yolo_scores)),
            'metrics': metrics
        }
    
    def _calculate_metrics(self, zero_shot_dets, yolo_boxes, yolo_classes):
        """Calculate precision, recall, and F1 score"""
        if not zero_shot_dets:
            return {'precision': 0, 'recall': 0, 'f1_score': 0, 'accuracy': 0, 'avg_iou': 0}
        
        # Match detections
        matches = []
        matched_yolo = set()
        
        for zs_det in zero_shot_dets:
            best_iou = 0
            best_idx = -1
            
            for j, (yolo_box, yolo_class) in enumerate(zip(yolo_boxes, yolo_classes)):
                if j in matched_yolo:
                    continue
                
                iou = self._calculate_iou(zs_det['box'], yolo_box)
                if iou > best_iou and iou > 0.5:
                    best_iou = iou
                    best_idx = j
            
            if best_idx >= 0:
                matches.append({
                    'zero_shot_class': zs_det['clip_class'],
                    'yolo_class': yolo_classes[best_idx],
                    'iou': best_iou,
                    'is_correct': self._classes_match(zs_det['clip_class'], yolo_classes[best_idx])
                })
                matched_yolo.add(best_idx)
        
        # Calculate metrics
        correct_matches = sum(1 for m in matches if m['is_correct'])
        precision = correct_matches / len(zero_shot_dets) if zero_shot_dets else 0
        recall = correct_matches / len(yolo_boxes) if yolo_boxes else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'accuracy': correct_matches / len(matches) if matches else 0,
            'avg_iou': np.mean([m['iou'] for m in matches]) if matches else 0,
            'num_zero_shot': len(zero_shot_dets),
            'num_yolo': len(yolo_boxes),
            'num_matches': len(matches),
            'num_correct': correct_matches
        }
    
    def _classes_match(self, class1, class2):
        """Check if two classes match (flexible matching)"""
        class1 = str(class1).lower()
        class2 = str(class2).lower()
        
        # Remove prefixes
        prefixes = ['a ', 'an ', 'the ', 'photo of ', 'picture of ', 'image of ']
        for prefix in prefixes:
            if class1.startswith(prefix):
                class1 = class1[len(prefix):]
            if class2.startswith(prefix):
                class2 = class2[len(prefix):]
        
        # Check synonyms
        synonyms = {
            'person': ['human', 'people', 'man', 'woman', 'child', 'pedestrian'],
            'car': ['vehicle', 'automobile', 'sedan'],
            'bus': ['bus', 'coach'],
            'stop sign': ['stop sign', 'traffic sign', 'sign', 'street sign'],
            'traffic light': ['traffic light', 'stoplight', 'signal']
        }
        
        for key, syn_list in synonyms.items():
            if class1 == key and class2 in syn_list:
                return True
            if class2 == key and class1 in syn_list:
                return True
        
        # Check partial matches
        return (class1 in class2) or (class2 in class1)


# Test function
def test_pipeline():
    """Test the integrated pipeline"""
    print("Testing Improved Zero-Shot Detection Pipeline...")
    
    # Initialize pipeline with better settings
    pipeline = ZeroShotDetectionPipeline(
        yolo_model='yolov8n.pt',  # Use nano for speed
        conf_threshold=0.25
    )
    
    # Test 1: Standard objects
    print("\n=== TEST 1: Standard Objects ===")
    text_prompts = ["bus", "person", "stop sign"]
    
    # Test image (use local image if available)
    test_image = "test_image.jpg" if os.path.exists("test_image.jpg") else "https://ultralytics.com/images/bus.jpg"
    
    # Run detection
    results = pipeline.detect(test_image, text_prompts, clip_threshold=0.05)
    
    print(f"\nFound {len(results['detections'])} detections:")
    for i, detection in enumerate(results['detections']):
        print(f"{i+1}. {detection['clip_class']} - YOLO: {detection['yolo_confidence']:.3f}, CLIP: {detection['clip_confidence']:.3f}")
    
    # Visualize results
    if results['detections']:
        pipeline.visualize_detections(results, 'results/visualizations/pipeline_detections.jpg')
    
    # Test 2: Evaluation
    print("\n=== TEST 2: Performance Evaluation ===")
    eval_results = pipeline.evaluate_performance(test_image, text_prompts)
    
    print(f"Precision: {eval_results['metrics']['precision']:.3f}")
    print(f"Recall: {eval_results['metrics']['recall']:.3f}")
    print(f"F1 Score: {eval_results['metrics']['f1_score']:.3f}")
    print(f"Accuracy: {eval_results['metrics']['accuracy']:.3f}")
    
    return results


if __name__ == "__main__":
    test_pipeline()