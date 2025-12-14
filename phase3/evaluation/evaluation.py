#!/usr/bin/env python3
import os
import sys
import json
import numpy as np

# Add phase3 folder to path so we can import evaluation.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'phase3')))

# Add src to path for pipeline import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from pipeline import ZeroShotDetectionPipeline
from PIL import Image

class RealTimeEvaluator:
    def __init__(self, pipeline):
        self.pipeline = pipeline
    
    def evaluate_zero_shot(self, image_path, target_classes):
        """
        Evaluate zero-shot performance by comparing with YOLO's native detection
        YOLO detections serve as pseudo-ground truth for validation
        """
        print(f"\nEvaluating zero-shot detection on: {image_path}")
        print(f"Target classes: {target_classes}")
        
        # Step 1: Get YOLO native detections (pseudo-ground truth)
        print("\n1. Getting YOLO native detections...")
        yolo_results = self.pipeline.yolo_detector.detect(image_path)
        yolo_boxes, yolo_scores, yolo_classes = self.pipeline.yolo_detector.extract_detections(yolo_results)
        
        # Convert YOLO class IDs to names
        yolo_class_names = []
        if hasattr(yolo_results, 'names'):
            for class_id in yolo_classes:
                yolo_class_names.append(yolo_results.names[int(class_id)])
        elif hasattr(yolo_results[0], 'names'):  # Handle list results
            for class_id in yolo_classes:
                yolo_class_names.append(yolo_results[0].names[int(class_id)])
        else:
            # Fallback to COCO class names
            from pipeline import COCO_CLASSES
            for class_id in yolo_classes:
                if int(class_id) < len(COCO_CLASSES):
                    yolo_class_names.append(COCO_CLASSES[int(class_id)])
                else:
                    yolo_class_names.append(f"class_{int(class_id)}")
        
        print(f"   YOLO found {len(yolo_boxes)} objects: {set(yolo_class_names)}")
        
        # Step 2: Run zero-shot detection
        print("\n2. Running zero-shot detection...")
        zero_shot_results = self.pipeline.detect(image_path, target_classes)
        zero_shot_detections = zero_shot_results['detections']
        
        print(f"   Zero-shot found {len(zero_shot_detections)} objects")
        
        # Step 3: Match detections
        print("\n3. Matching detections...")
        matches = self._match_detections(zero_shot_detections, yolo_boxes, yolo_class_names)
        
        # Step 4: Calculate metrics
        print("\n4. Calculating metrics...")
        metrics = self._calculate_metrics(matches, len(zero_shot_detections), len(yolo_boxes))
        
        # Step 5: Visualize comparison
        self._visualize_comparison(image_path, zero_shot_detections, yolo_boxes, yolo_class_names, metrics)
        
        return {
            'image': image_path,
            'target_classes': target_classes,
            'zero_shot_detections': zero_shot_detections,
            'yolo_detections': list(zip(yolo_boxes, yolo_class_names, yolo_scores)),
            'matches': matches,
            'metrics': metrics
        }
    
    def _match_detections(self, zero_shot_dets, yolo_boxes, yolo_classes):
        """Match zero-shot detections with YOLO detections"""
        matches = []
        matched_yolo_indices = set()
        
        for i, zs_det in enumerate(zero_shot_dets):
            best_match_idx = -1
            best_iou = 0
            
            for j, (yolo_box, yolo_class) in enumerate(zip(yolo_boxes, yolo_classes)):
                if j in matched_yolo_indices:
                    continue
                    
                iou = self._calculate_iou(zs_det['box'], yolo_box)
                
                if iou > best_iou and iou > 0.5:  # IoU threshold
                    best_iou = iou
                    best_match_idx = j
            
            if best_match_idx >= 0:
                matches.append({
                    'zero_shot_idx': i,
                    'yolo_idx': best_match_idx,
                    'iou': best_iou,
                    'zero_shot_class': zs_det['clip_class'],
                    'yolo_class': yolo_classes[best_match_idx],
                    'is_correct': self._classes_match(zs_det['clip_class'], yolo_classes[best_match_idx])
                })
                matched_yolo_indices.add(best_match_idx)
        
        return matches
    
    def _classes_match(self, class1, class2):
        """Check if two class names match (flexible matching)"""
        class1_lower = str(class1).lower().strip()
        class2_lower = str(class2).lower().strip()
        
        # Remove articles and common prefixes
        prefixes = ['a ', 'an ', 'the ', 'photo of ', 'picture of ', 'image of ', 'a photo of ', 'a picture of ']
        for prefix in prefixes:
            if class1_lower.startswith(prefix):
                class1_lower = class1_lower[len(prefix):]
            if class2_lower.startswith(prefix):
                class2_lower = class2_lower[len(prefix):]
        
        # Clean up any remaining whitespace
        class1_lower = class1_lower.strip()
        class2_lower = class2_lower.strip()
        
        # Check if one contains the other or they're synonyms
        if class1_lower == class2_lower:
            return True
            
        # Handle common synonyms
        synonyms = {
            'person': ['human', 'people', 'man', 'woman', 'child'],
            'car': ['vehicle', 'automobile', 'sedan', 'truck'],
            'bus': ['bus', 'coach', 'omnibus'],
            'stop sign': ['stop sign', 'traffic sign', 'sign'],
            'truck': ['truck', 'lorry', 'van']
        }
        
        # Check synonyms
        for key, syn_list in synonyms.items():
            if class1_lower == key and class2_lower in syn_list:
                return True
            if class2_lower == key and class1_lower in syn_list:
                return True
        
        # Check partial matches
        return (class1_lower in class2_lower) or (class2_lower in class1_lower)
    
    def _calculate_iou(self, box1, box2):
        """Calculate Intersection over Union"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def _calculate_metrics(self, matches, num_zero_shot, num_yolo):
        """Calculate evaluation metrics"""
        if num_zero_shot == 0 and num_yolo == 0:
            return {'precision': 0, 'recall': 0, 'f1': 0, 'accuracy': 0, 'avg_iou': 0}
        
        correct_matches = sum(1 for m in matches if m['is_correct'])
        false_positives = num_zero_shot - len(matches)
        false_negatives = num_yolo - len(matches)
        
        # Adjust precision calculation to account for unmatched zero-shot detections
        precision = correct_matches / num_zero_shot if num_zero_shot > 0 else 0
        recall = correct_matches / num_yolo if num_yolo > 0 else 0
        
        # Calculate F1 score (harmonic mean of precision and recall)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Accuracy (classification accuracy of matched detections)
        accuracy = correct_matches / len(matches) if matches else 0
        
        # Average IoU of matched detections
        avg_iou = np.mean([m['iou'] for m in matches]) if matches else 0
        
        return {
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'accuracy': float(accuracy),
            'avg_iou': float(avg_iou),
            'num_zero_shot': int(num_zero_shot),
            'num_yolo': int(num_yolo),
            'num_matches': len(matches),
            'num_correct': int(correct_matches),
            'false_positives': int(false_positives),
            'false_negatives': int(false_negatives)
        }
    
    def _visualize_comparison(self, image_path, zero_shot_dets, yolo_boxes, yolo_classes, metrics):
        """Visualize comparison between zero-shot and YOLO detections"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
        except ImportError:
            print("Warning: matplotlib not installed. Skipping visualization.")
            return
        
        try:
            image = Image.open(image_path)
        except FileNotFoundError:
            print(f"Error: Image file not found at {image_path}")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Left: Zero-shot detections
        ax1.imshow(image)
        for det in zero_shot_dets:
            box = det['box']
            rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1],
                                   linewidth=2, edgecolor='green', facecolor='none')
            ax1.add_patch(rect)
            ax1.text(box[0], box[1]-5, f"{det['clip_class']} ({det['clip_score']:.2f})", 
                    color='green', fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
        ax1.set_title('Zero-Shot Detections (CLIP)', fontsize=14)
        ax1.axis('off')
        
        # Right: YOLO native detections
        ax2.imshow(image)
        for box, cls in zip(yolo_boxes, yolo_classes):
            rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1],
                                   linewidth=2, edgecolor='blue', facecolor='none')
            ax2.add_patch(rect)
            ax2.text(box[0], box[1]-5, cls, color='blue', fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
        ax2.set_title('YOLO Native Detections', fontsize=14)
        ax2.axis('off')
        
        # Add metrics text
        metrics_text = (
            f"Precision: {metrics['precision']:.3f}\n"
            f"Recall: {metrics['recall']:.3f}\n"
            f"F1 Score: {metrics['f1_score']:.3f}\n"
            f"Accuracy: {metrics['accuracy']:.3f}\n"
            f"Avg IoU: {metrics['avg_iou']:.3f}\n"
            f"Zero-shot detections: {metrics['num_zero_shot']}\n"
            f"YOLO detections: {metrics['num_yolo']}\n"
            f"Matches: {metrics['num_matches']}\n"
            f"Correct: {metrics['num_correct']}"
        )
        
        plt.figtext(0.5, 0.01, metrics_text, ha='center', fontsize=11,
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        plt.suptitle('Zero-Shot vs YOLO Detection Comparison', fontsize=16, y=0.95)
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        
        # Save results
        os.makedirs('phase3/results', exist_ok=True)
        base_name = os.path.basename(image_path).split('.')[0]
        output_path = f"phase3/results/evaluation_{base_name}.jpg"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n✅ Comparison visualization saved to: {output_path}")
        
        # Save metrics to JSON
        metrics_path = f"phase3/results/metrics_{base_name}.json"
        with open(metrics_path, 'w') as f:
            json.dump({
                'image': image_path,
                'metrics': metrics,
                'timestamp': os.path.getctime(image_path) if os.path.exists(image_path) else None
            }, f, indent=2)
        print(f"✅ Metrics saved to: {metrics_path}")
        
        plt.show()
    
    def batch_evaluate(self, image_paths, target_classes_list):
        """Evaluate multiple images with different target classes"""
        results = []
        for i, (image_path, target_classes) in enumerate(zip(image_paths, target_classes_list)):
            print(f"\n{'='*50}")
            print(f"Evaluating image {i+1}/{len(image_paths)}")
            print(f"{'='*50}")
            result = self.evaluate_zero_shot(image_path, target_classes)
            results.append(result)
        
        # Generate batch summary
        self._generate_batch_summary(results)
        return results
    
    def _generate_batch_summary(self, results):
        """Generate summary report for batch evaluation"""
        print("\n" + "=" * 70)
        print("BATCH EVALUATION SUMMARY")
        print("=" * 70)
        
        avg_metrics = {
            'precision': 0,
            'recall': 0,
            'f1_score': 0,
            'accuracy': 0,
            'avg_iou': 0
        }
        
        for i, result in enumerate(results):
            metrics = result['metrics']
            print(f"\nImage {i+1}: {os.path.basename(result['image'])}")
            print(f"  Target classes: {result['target_classes']}")
            print(f"  Precision: {metrics['precision']:.3f}")
            print(f"  Recall: {metrics['recall']:.3f}")
            print(f"  F1 Score: {metrics['f1_score']:.3f}")
            print(f"  Accuracy: {metrics['accuracy']:.3f}")
            print(f"  Zero-shot detections: {metrics['num_zero_shot']}")
            print(f"  YOLO detections: {metrics['num_yolo']}")
            print(f"  Correct matches: {metrics['num_correct']}")
            
            # Accumulate for averages
            for key in avg_metrics:
                avg_metrics[key] += metrics[key]
        
        # Calculate averages
        num_results = len(results)
        for key in avg_metrics:
            avg_metrics[key] /= num_results
        
        print("\n" + "=" * 70)
        print("OVERALL AVERAGES")
        print("=" * 70)
        print(f"Average Precision: {avg_metrics['precision']:.3f}")
        print(f"Average Recall: {avg_metrics['recall']:.3f}")
        print(f"Average F1 Score: {avg_metrics['f1_score']:.3f}")
        print(f"Average Accuracy: {avg_metrics['accuracy']:.3f}")
        print(f"Average IoU: {avg_metrics['avg_iou']:.3f}")
        print("=" * 70)

def main():
    """Main evaluation function"""
    
    print("=" * 70)
    print("REAL-TIME ZERO-SHOT DETECTION EVALUATION")
    print("No hardcoded data - Uses YOLO as pseudo-ground truth")
    print("=" * 70)
    
    # Initialize pipeline
    pipeline = ZeroShotDetectionPipeline(yolo_model='yolov8n.pt', conf_threshold=0.25)
    evaluator = RealTimeEvaluator(pipeline)
    
    # Test on our image
    image_path = 'test_image.jpg'
    
    # Check if test image exists
    if not os.path.exists(image_path):
        print(f"\n⚠️  Warning: Test image not found at {image_path}")
        print("Please make sure test_image.jpg exists in the current directory.")
        print("You can download a test image or use your own.")
        
        # Try to find any image file
        image_files = [f for f in os.listdir('.') if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if image_files:
            image_path = image_files[0]
            print(f"\nUsing found image: {image_path}")
        else:
            print("\n❌ No image files found. Please add an image to the current directory.")
            return
    
    print(f"\nUsing image: {image_path}")
    
    # Test case 1: Standard objects
    print("\n📊 TEST 1: Standard Objects")
    target_classes = ['bus', 'person', 'stop sign']
    results1 = evaluator.evaluate_zero_shot(image_path, target_classes)
    
    # Test case 2: Additional objects (some not in image)
    print("\n📊 TEST 2: Extended Objects (including non-present)")
    target_classes = ['bus', 'person', 'stop sign', 'car', 'traffic light', 'building']
    results2 = evaluator.evaluate_zero_shot(image_path, target_classes)
    
    # Test case 3: Creative/novel objects
    print("\n📊 TEST 3: Novel Objects (zero-shot capability test)")
    target_classes = ['vehicle', 'human', 'street sign', 'road']
    results3 = evaluator.evaluate_zero_shot(image_path, target_classes)
    
    # Generate summary report
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    
    for test_name, results in [("Standard", results1), ("Extended", results2), ("Novel", results3)]:
        metrics = results['metrics']
        print(f"\n{test_name} Test:")
        print(f"  Target classes: {results['target_classes']}")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall: {metrics['recall']:.3f}")
        print(f"  F1 Score: {metrics['f1_score']:.3f}")
        print(f"  Accuracy: {metrics['accuracy']:.3f}")
        print(f"  Zero-shot detections: {metrics['num_zero_shot']}")
        print(f"  YOLO detections: {metrics['num_yolo']}")
        print(f"  Correct matches: {metrics['num_correct']}/{metrics['num_matches']}")
    
    print("\n" + "=" * 70)
    print("✅ EVALUATION COMPLETE")
    print("All results saved to phase3/results/")
    print("=" * 70)

if __name__ == "__main__":
    main()