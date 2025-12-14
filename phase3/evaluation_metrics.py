#!/usr/bin/env python3
"""
Enhanced evaluation metrics for zero-shot object detection with auto-ground truth generation
"""
import sys
import numpy as np
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import cv2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class EnhancedZeroShotEvaluator:
    def __init__(self, pipeline):
        """
        Initialize evaluator with auto-ground truth generation
        
        Args:
            pipeline: ZeroShotDetectionPipeline instance
        """
        self.pipeline = pipeline
        self.results = []
    
    def evaluate_with_yolo_as_gt(self, image_path, target_classes, use_yolo_as_gt=True):
        """
        Evaluate using YOLO detections as pseudo-ground truth
        
        Args:
            image_path: Path to image
            target_classes: List of target classes
            use_yolo_as_gt: Use YOLO detections as ground truth
            
        Returns:
            Comprehensive evaluation results
        """
        print(f"\nEvaluating: {os.path.basename(image_path)}")
        print(f"Target classes: {target_classes}")
        
        # Step 1: Get YOLO detections (as ground truth)
        print("\n1. Getting YOLO detections as pseudo-ground truth...")
        yolo_results = self.pipeline.yolo_detector.detect(image_path)
        
        # Extract YOLO detections
        if hasattr(yolo_results, 'boxes'):
            # For newer YOLO versions
            yolo_boxes = yolo_results.boxes.xyxy.cpu().numpy() if yolo_results.boxes.xyxy is not None else []
            yolo_scores = yolo_results.boxes.conf.cpu().numpy() if yolo_results.boxes.conf is not None else []
            yolo_class_ids = yolo_results.boxes.cls.cpu().numpy().astype(int) if yolo_results.boxes.cls is not None else []
        else:
            # For older versions or different format
            yolo_boxes, yolo_scores, yolo_class_ids = self.pipeline.yolo_detector.extract_detections(yolo_results)
        
        # Get class names
        if hasattr(yolo_results, 'names'):
            class_names = yolo_results.names
        elif hasattr(self.pipeline.yolo_detector, 'class_names'):
            class_names = self.pipeline.yolo_detector.class_names
        else:
            # Default COCO class names
            class_names = {
                0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
                5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
                10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
                14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
                20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
                25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
                30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
                35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket',
                39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife',
                44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich',
                49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza',
                54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant',
                59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop',
                64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave',
                69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book',
                74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier',
                79: 'toothbrush'
            }
        
        # Create ground truth from YOLO detections
        ground_truth = []
        for i, (box, score, class_id) in enumerate(zip(yolo_boxes, yolo_scores, yolo_class_ids)):
            class_name = class_names.get(int(class_id), f'class_{int(class_id)}')
            # Convert [x1, y1, x2, y2] to [x, y, w, h] format
            x1, y1, x2, y2 = box
            bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
            
            ground_truth.append({
                'bbox': bbox,
                'category_id': int(class_id),
                'category_name': class_name,
                'score': float(score),
                'area': float((x2 - x1) * (y2 - y1))
            })
        
        print(f"   YOLO found {len(ground_truth)} objects: {[gt['category_name'] for gt in ground_truth]}")
        
        # Step 2: Run zero-shot detection
        print("\n2. Running zero-shot detection...")
        zero_shot_results = self.pipeline.detect(image_path, target_classes)
        detections = zero_shot_results['detections']
        
        print(f"   Zero-shot found {len(detections)} detections")
        
        # Step 3: Calculate evaluation metrics
        print("\n3. Calculating metrics...")
        metrics = self._calculate_comprehensive_metrics(detections, ground_truth)
        
        # Step 4: Create ground truth for target classes (filter YOLO detections)
        target_gt = [gt for gt in ground_truth 
                    if any(self._class_matches(gt['category_name'], target_class) 
                          for target_class in target_classes)]
        
        # Step 5: Calculate target-specific metrics
        target_metrics = self._calculate_comprehensive_metrics(detections, target_gt)
        
        # Step 6: Visualize results
        self._visualize_enhanced_evaluation(
            image_path, 
            detections, 
            ground_truth, 
            target_gt,
            metrics,
            target_metrics
        )
        
        # Prepare results
        result = {
            'image_path': image_path,
            'target_classes': target_classes,
            'detections': detections,
            'ground_truth': ground_truth,
            'target_ground_truth': target_gt,
            'metrics': {
                'all_classes': metrics,
                'target_classes': target_metrics
            },
            'timestamp': datetime.now().isoformat()
        }
        
        self.results.append(result)
        
        # Display results
        self._display_evaluation_summary(result)
        
        return result
    
    def _calculate_comprehensive_metrics(self, detections, ground_truth, iou_threshold=0.5):
        """Calculate comprehensive detection metrics"""
        if not ground_truth:
            return {
                'mAP': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'accuracy': 0.0,
                'avg_iou': 0.0,
                'total_detections': len(detections),
                'total_ground_truth': 0,
                'true_positives': 0,
                'false_positives': len(detections),
                'false_negatives': 0,
                'matches_found': 0
            }
        
        # Sort detections by confidence
        sorted_dets = sorted(enumerate(detections), 
                           key=lambda x: x[1].get('clip_confidence', 0.5), 
                           reverse=True)
        
        gt_matched = [False] * len(ground_truth)
        matches = []
        
        # Match detections to ground truth
        for det_idx, detection in sorted_dets:
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt in enumerate(ground_truth):
                if gt_matched[gt_idx]:
                    continue
                
                # Convert ground truth bbox to [x1, y1, x2, y2]
                gt_bbox = gt['bbox']
                if len(gt_bbox) == 4:
                    x, y, w, h = gt_bbox
                    gt_box = [x, y, x + w, y + h]
                else:
                    gt_box = gt_bbox
                
                # Calculate IoU
                iou = self._calculate_iou(detection['box'], gt_box)
                
                if iou > best_iou and iou >= iou_threshold:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_gt_idx >= 0:
                gt = ground_truth[best_gt_idx]
                is_correct = self._class_matches(detection['clip_class'], gt['category_name'])
                
                matches.append({
                    'det_idx': det_idx,
                    'gt_idx': best_gt_idx,
                    'iou': best_iou,
                    'correct': is_correct,
                    'det_class': detection['clip_class'],
                    'gt_class': gt['category_name'],
                    'det_confidence': detection.get('clip_confidence', 0.5),
                    'gt_confidence': gt.get('score', 1.0)
                })
                
                gt_matched[best_gt_idx] = True
        
        # Calculate metrics
        true_positives = sum(1 for m in matches if m['correct'])
        false_positives = len(detections) - len(matches)
        false_negatives = sum(1 for matched in gt_matched if not matched)
        
        precision = true_positives / len(detections) if len(detections) > 0 else 0
        recall = true_positives / len(ground_truth) if len(ground_truth) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = true_positives / len(matches) if matches else 0
        avg_iou = np.mean([m['iou'] for m in matches]) if matches else 0
        
        # Calculate mAP (simplified)
        map_score = self._calculate_simplified_map(matches, detections, ground_truth)
        
        return {
            'mAP': float(map_score),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'accuracy': float(accuracy),
            'avg_iou': float(avg_iou),
            'total_detections': len(detections),
            'total_ground_truth': len(ground_truth),
            'true_positives': int(true_positives),
            'false_positives': int(false_positives),
            'false_negatives': int(false_negatives),
            'matches_found': len(matches),
            'match_details': matches
        }
    
    def _calculate_simplified_map(self, matches, detections, ground_truth):
        """Calculate a simplified mAP for evaluation"""
        if not matches:
            return 0.0
        
        # Group by class
        class_stats = {}
        for match in matches:
            if match['correct']:
                class_name = match['det_class']
                if class_name not in class_stats:
                    class_stats[class_name] = {'tp': 0, 'fp': 0, 'fn': 0}
                class_stats[class_name]['tp'] += 1
        
        # Calculate AP per class
        aps = []
        for class_name, stats in class_stats.items():
            # Count total detections of this class
            total_class_dets = sum(1 for det in detections if det['clip_class'] == class_name)
            
            # Count total ground truth of this class
            total_class_gt = sum(1 for gt in ground_truth 
                               if self._class_matches(gt['category_name'], class_name))
            
            if total_class_dets > 0 and total_class_gt > 0:
                precision = stats['tp'] / total_class_dets
                recall = stats['tp'] / total_class_gt
                ap = precision * recall  # Simplified AP
                aps.append(ap)
        
        return np.mean(aps) if aps else 0.0
    
    def _calculate_iou(self, box1, box2):
        """Calculate Intersection over Union"""
        # Ensure boxes are valid
        box1 = [float(b) for b in box1]
        box2 = [float(b) for b in box2]
        
        # Calculate intersection
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        # Calculate union
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def _class_matches(self, det_class, gt_class):
        """Check if detected class matches ground truth class"""
        det_class = str(det_class).lower().strip()
        gt_class = str(gt_class).lower().strip()
        
        # Remove prefixes
        prefixes = ['a ', 'an ', 'the ', 'photo of ', 'picture of ', 'image of ']
        for prefix in prefixes:
            if det_class.startswith(prefix):
                det_class = det_class[len(prefix):]
            if gt_class.startswith(prefix):
                gt_class = gt_class[len(prefix):]
        
        # Exact match
        if det_class == gt_class:
            return True
        
        # Synonyms
        synonyms = {
            'person': ['human', 'people', 'man', 'woman', 'child', 'pedestrian', 'person'],
            'car': ['vehicle', 'automobile', 'sedan', 'car'],
            'bus': ['bus', 'coach', 'omnibus'],
            'stop sign': ['stop sign', 'traffic sign', 'sign', 'street sign'],
            'truck': ['truck', 'lorry', 'van'],
            'traffic light': ['traffic light', 'stoplight', 'signal'],
            'bicycle': ['bike', 'bicycle', 'cycle'],
            'motorcycle': ['motorcycle', 'motorbike', 'scooter']
        }
        
        # Check synonyms
        for key, syn_list in synonyms.items():
            if det_class == key and gt_class in syn_list:
                return True
            if gt_class == key and det_class in syn_list:
                return True
        
        # Check partial matches
        return (det_class in gt_class) or (gt_class in det_class)
    
    def _visualize_enhanced_evaluation(self, image_path, detections, ground_truth, 
                                      target_gt, metrics, target_metrics):
        """Create comprehensive visualization"""
        try:
            image = Image.open(image_path)
            img_array = np.array(image)
        except:
            print(f"Could not open image: {image_path}")
            return
        
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Original image with all detections and ground truth
        ax1 = plt.subplot(2, 3, 1)
        ax1.imshow(image)
        ax1.set_title('All Detections vs All Ground Truth', fontsize=12)
        ax1.axis('off')
        
        # Draw ground truth (blue)
        for gt in ground_truth:
            bbox = gt['bbox']
            if len(bbox) == 4:
                x, y, w, h = bbox
                rect = Rectangle((x, y), w, h, linewidth=2, 
                               edgecolor='blue', facecolor='none', alpha=0.7)
                ax1.add_patch(rect)
                ax1.text(x, y-5, gt['category_name'], color='blue', fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))
        
        # Draw detections (green for TP, red for FP)
        for det in detections:
            box = det['box']
            # Check if this detection matches any ground truth
            is_match = any(self._calculate_iou(box, [
                gt['bbox'][0], gt['bbox'][1], 
                gt['bbox'][0] + gt['bbox'][2], 
                gt['bbox'][1] + gt['bbox'][3]
            ]) > 0.5 for gt in ground_truth)
            
            color = 'green' if is_match else 'red'
            rect = Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1],
                           linewidth=2, edgecolor=color, facecolor='none', alpha=0.7)
            ax1.add_patch(rect)
            ax1.text(box[0], box[1]-5, f"{det['clip_class']}", color=color, fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))
        
        # 2. Zero-shot detections only
        ax2 = plt.subplot(2, 3, 2)
        ax2.imshow(image)
        ax2.set_title('Zero-Shot Detections Only', fontsize=12)
        ax2.axis('off')
        
        for det in detections:
            box = det['box']
            rect = Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1],
                           linewidth=2, edgecolor='green', facecolor='none')
            ax2.add_patch(rect)
            ax2.text(box[0], box[1]-5, f"{det['clip_class']} ({det.get('clip_confidence', 0):.2f})", 
                    color='green', fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
        
        # 3. YOLO ground truth only
        ax3 = plt.subplot(2, 3, 3)
        ax3.imshow(image)
        ax3.set_title('YOLO Ground Truth Only', fontsize=12)
        ax3.axis('off')
        
        for gt in ground_truth:
            bbox = gt['bbox']
            if len(bbox) == 4:
                x, y, w, h = bbox
                rect = Rectangle((x, y), w, h, linewidth=2, 
                               edgecolor='blue', facecolor='none')
                ax3.add_patch(rect)
                ax3.text(x, y-5, f"{gt['category_name']} ({gt.get('score', 0):.2f})", 
                        color='blue', fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
        
        # 4. Metrics visualization
        ax4 = plt.subplot(2, 3, 4)
        ax4.axis('off')
        
        metrics_text = (
            "OVERALL METRICS (All Classes):\n"
            f"Precision: {metrics['precision']:.3f}\n"
            f"Recall: {metrics['recall']:.3f}\n"
            f"F1 Score: {metrics['f1_score']:.3f}\n"
            f"mAP: {metrics['mAP']:.3f}\n"
            f"Avg IoU: {metrics['avg_iou']:.3f}\n"
            f"TP: {metrics['true_positives']} | FP: {metrics['false_positives']} | FN: {metrics['false_negatives']}\n"
            f"Detections: {metrics['total_detections']} | GT: {metrics['total_ground_truth']}"
        )
        
        ax4.text(0.1, 0.5, metrics_text, fontsize=10, va='center',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        # 5. Target classes metrics
        ax5 = plt.subplot(2, 3, 5)
        ax5.axis('off')
        
        target_text = (
            "TARGET CLASSES METRICS:\n"
            f"Precision: {target_metrics['precision']:.3f}\n"
            f"Recall: {target_metrics['recall']:.3f}\n"
            f"F1 Score: {target_metrics['f1_score']:.3f}\n"
            f"mAP: {target_metrics['mAP']:.3f}\n"
            f"Avg IoU: {target_metrics['avg_iou']:.3f}\n"
            f"TP: {target_metrics['true_positives']} | FP: {target_metrics['false_positives']} | FN: {target_metrics['false_negatives']}\n"
            f"Detections: {target_metrics['total_detections']} | GT: {target_metrics['total_ground_truth']}"
        )
        
        ax5.text(0.1, 0.5, target_text, fontsize=10, va='center',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
        
        # 6. Bar chart of class-wise performance
        ax6 = plt.subplot(2, 3, 6)
        
        # Count detections by class
        det_classes = {}
        for det in detections:
            cls = det['clip_class']
            det_classes[cls] = det_classes.get(cls, 0) + 1
        
        gt_classes = {}
        for gt in ground_truth:
            cls = gt['category_name']
            gt_classes[cls] = gt_classes.get(cls, 0) + 1
        
        classes = sorted(set(list(det_classes.keys()) + list(gt_classes.keys())))
        if classes:
            det_counts = [det_classes.get(cls, 0) for cls in classes]
            gt_counts = [gt_classes.get(cls, 0) for cls in classes]
            
            x = np.arange(len(classes))
            width = 0.35
            
            ax6.bar(x - width/2, det_counts, width, label='Detections', color='green', alpha=0.7)
            ax6.bar(x + width/2, gt_counts, width, label='Ground Truth', color='blue', alpha=0.7)
            
            ax6.set_xlabel('Classes')
            ax6.set_ylabel('Count')
            ax6.set_title('Class-wise Detection Counts')
            ax6.set_xticks(x)
            ax6.set_xticklabels(classes, rotation=45, ha='right')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        
        plt.suptitle(f'Zero-Shot Detection Evaluation - {os.path.basename(image_path)}', 
                    fontsize=16, y=0.98)
        plt.tight_layout()
        
        # Save visualization
        os.makedirs('phase3/evaluation_results', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f"phase3/evaluation_results/comprehensive_eval_{os.path.basename(image_path).split('.')[0]}_{timestamp}.jpg"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Comprehensive visualization saved to: {output_path}")
    
    def _display_evaluation_summary(self, result):
        """Display evaluation summary"""
        metrics = result['metrics']['all_classes']
        target_metrics = result['metrics']['target_classes']
        
        print("\n" + "="*70)
        print("EVALUATION SUMMARY")
        print("="*70)
        
        print(f"\n📊 Overall Performance (All Classes):")
        print(f"   Precision: {metrics['precision']:.3f}")
        print(f"   Recall: {metrics['recall']:.3f}")
        print(f"   F1 Score: {metrics['f1_score']:.3f}")
        print(f"   mAP: {metrics['mAP']:.3f}")
        print(f"   Avg IoU: {metrics['avg_iou']:.3f}")
        print(f"   True Positives: {metrics['true_positives']}")
        print(f"   False Positives: {metrics['false_positives']}")
        print(f"   False Negatives: {metrics['false_negatives']}")
        
        print(f"\n🎯 Target Classes Performance:")
        print(f"   Precision: {target_metrics['precision']:.3f}")
        print(f"   Recall: {target_metrics['recall']:.3f}")
        print(f"   F1 Score: {target_metrics['f1_score']:.3f}")
        print(f"   mAP: {target_metrics['mAP']:.3f}")
        
        print(f"\n🔍 Detection Details:")
        for i, det in enumerate(result['detections']):
            # Find matching ground truth
            match_info = "No match"
            for match in metrics.get('match_details', []):
                if match['det_idx'] == i:
                    match_info = f"Matched: {match['gt_class']} (IoU: {match['iou']:.2f})"
                    break
            
            print(f"   {i+1}. {det['clip_class']} - {match_info}")
            print(f"      CLIP: {det.get('clip_confidence', 0):.3f}, YOLO: {det.get('yolo_confidence', 0):.3f}")
        
        print(f"\n📋 Ground Truth Objects (YOLO):")
        for i, gt in enumerate(result['ground_truth']):
            print(f"   {i+1}. {gt['category_name']} (score: {gt.get('score', 0):.3f})")
        
        print("="*70)
    
    def batch_evaluate(self, image_paths, target_classes_list):
        """Batch evaluate multiple images"""
        all_results = []
        
        for image_path, target_classes in zip(image_paths, target_classes_list):
            try:
                result = self.evaluate_with_yolo_as_gt(image_path, target_classes)
                all_results.append(result)
            except Exception as e:
                print(f"Error evaluating {image_path}: {e}")
                continue
        
        # Generate batch report
        self._generate_batch_report(all_results)
        
        return all_results
    
    def _generate_batch_report(self, all_results):
        """Generate batch evaluation report"""
        if not all_results:
            return
        
        overall_metrics = []
        target_metrics = []
        
        for result in all_results:
            overall_metrics.append(result['metrics']['all_classes'])
            target_metrics.append(result['metrics']['target_classes'])
        
        # Calculate averages
        avg_overall = {}
        avg_target = {}
        
        for key in overall_metrics[0].keys():
            if key not in ['match_details', 'total_detections', 'total_ground_truth', 
                          'true_positives', 'false_positives', 'false_negatives']:
                avg_overall[key] = np.mean([m[key] for m in overall_metrics])
                avg_target[key] = np.mean([m[key] for m in target_metrics])
        
        # Save batch report
        report = {
            'metadata': {
                'evaluation_date': datetime.now().isoformat(),
                'num_images': len(all_results),
                'images': [os.path.basename(r['image_path']) for r in all_results]
            },
            'average_metrics': {
                'overall': avg_overall,
                'target_classes': avg_target
            },
            'per_image_results': [
                {
                    'image': os.path.basename(r['image_path']),
                    'target_classes': r['target_classes'],
                    'overall_metrics': r['metrics']['all_classes'],
                    'target_metrics': r['metrics']['target_classes']
                }
                for r in all_results
            ]
        }
        
        os.makedirs('phase3/evaluation_results', exist_ok=True)
        report_file = f"phase3/evaluation_results/batch_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📊 Batch report saved to: {report_file}")
        
        # Print summary
        print("\n" + "="*70)
        print("BATCH EVALUATION SUMMARY")
        print("="*70)
        print(f"Images evaluated: {len(all_results)}")
        print(f"\nAverage Overall Metrics:")
        print(f"  Precision: {avg_overall.get('precision', 0):.3f}")
        print(f"  Recall: {avg_overall.get('recall', 0):.3f}")
        print(f"  F1 Score: {avg_overall.get('f1_score', 0):.3f}")
        print(f"  mAP: {avg_overall.get('mAP', 0):.3f}")
        print(f"\nAverage Target Classes Metrics:")
        print(f"  Precision: {avg_target.get('precision', 0):.3f}")
        print(f"  Recall: {avg_target.get('recall', 0):.3f}")
        print(f"  F1 Score: {avg_target.get('f1_score', 0):.3f}")
        print(f"  mAP: {avg_target.get('mAP', 0):.3f}")
        print("="*70)

def run_enhanced_evaluation():
    """Run enhanced evaluation"""
    
    print("=" * 70)
    print("ENHANCED ZERO-SHOT DETECTION EVALUATION")
    print("Using YOLO detections as pseudo-ground truth")
    print("=" * 70)
    
    # Import pipeline
    from src.pipeline import ZeroShotDetectionPipeline
    
    # Initialize pipeline
    print("\nInitializing detection pipeline...")
    try:
        pipeline = ZeroShotDetectionPipeline(yolo_model='yolov8n.pt', conf_threshold=0.25)
        evaluator = EnhancedZeroShotEvaluator(pipeline)
        print("✅ Pipeline initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize pipeline: {e}")
        return
    
    # Find test image
    test_image = 'test_image.jpg'
    if not os.path.exists(test_image):
        print(f"\n⚠️  Test image not found: {test_image}")
        # Try to find any image
        image_files = [f for f in os.listdir('.') if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if image_files:
            test_image = image_files[0]
            print(f"Using found image: {test_image}")
        else:
            print("❌ No image files found.")
            return
    
    print(f"\nEvaluating on: {test_image}")
    
    # Define test scenarios
    test_scenarios = [
        {
            'name': 'Standard Objects',
            'classes': ['person', 'bus', 'stop sign']
        },
        {
            'name': 'Extended Objects',
            'classes': ['person', 'car', 'bus', 'stop sign', 'traffic light', 'building']
        },
        {
            'name': 'Novel Descriptions',
            'classes': ['red vehicle', 'people walking', 'street sign', 'large object']
        }
    ]
    
    all_results = []
    
    for scenario in test_scenarios:
        print(f"\n{'='*60}")
        print(f"SCENARIO: {scenario['name']}")
        print(f"{'='*60}")
        
        result = evaluator.evaluate_with_yolo_as_gt(
            test_image, 
            scenario['classes']
        )
        all_results.append(result)
    
    # Final summary
    print("\n" + "="*70)
    print("FINAL COMPARISON")
    print("="*70)
    
    for result in all_results:
        metrics = result['metrics']['target_classes']
        print(f"\n{result['target_classes']}")
        print(f"  F1 Score: {metrics['f1_score']:.3f}")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall: {metrics['recall']:.3f}")
        print(f"  mAP: {metrics['mAP']:.3f}")
    
    print("\n✅ Enhanced evaluation complete!")
    print("Check phase3/evaluation_results/ for visualizations and reports")
    print("="*70)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced zero-shot detection evaluation')
    parser.add_argument('--mode', choices=['single', 'batch'], default='single',
                       help='Evaluation mode')
    parser.add_argument('--image', type=str, help='Image path')
    
    args = parser.parse_args()
    
    if args.mode == 'single':
        run_enhanced_evaluation()
    else:
        # For batch mode, you would need to provide a list of images
        print("Batch mode would require a list of images.")
        print("Running single image evaluation instead.")
        run_enhanced_evaluation()