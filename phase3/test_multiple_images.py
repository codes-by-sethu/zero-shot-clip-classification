#!/usr/bin/env python3
"""
Test zero-shot detection on multiple images automatically
Downloads test images from URLs or uses local images
"""
import os
import sys
import json
import requests
from datetime import datetime
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline import ZeroShotDetectionPipeline
from phase3.evaluation import RealTimeEvaluator

def download_test_images():
    """Download test images from URLs with fallback to local images"""
    
    # Test image URLs (using reliable, smaller images)
    test_images = {
        'bus_street.jpg': 'https://ultralytics.com/images/bus.jpg',
        'city_traffic.jpg': 'https://images.unsplash.com/photo-1544620347-c4fd4a3d5957?w=640&h=480',
        'animals.jpg': 'https://images.unsplash.com/photo-1546182990-dffeafbe841d?w=640&h=480',
        'office.jpg': 'https://images.unsplash.com/photo-1497366754035-f200968a6e72?w=640&h=480',
    }
    
    downloaded = []
    
    print("Setting up test images...")
    
    # First, check for local images
    local_images = [f for f in os.listdir('.') 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    if local_images:
        print(f"Found {len(local_images)} local image(s):")
        for img in local_images[:3]:  # Show first 3
            downloaded.append(img)
            print(f"  ✓ {img}")
        if len(local_images) > 3:
            print(f"  ... and {len(local_images)-3} more")
    
    # Download from URLs if needed
    os.makedirs('phase3/test_images', exist_ok=True)
    
    for filename, url in test_images.items():
        local_path = f"phase3/test_images/{filename}"
        
        if os.path.exists(local_path):
            downloaded.append(local_path)
            print(f"  ✓ {filename} already downloaded")
            continue
            
        print(f"Downloading {filename}...")
        try:
            response = requests.get(url, stream=True, timeout=15)
            if response.status_code == 200:
                with open(local_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                downloaded.append(local_path)
                print(f"  ✓ Downloaded to {local_path}")
                time.sleep(0.5)  # Be nice to servers
            else:
                print(f"  ✗ Failed to download {filename} (HTTP {response.status_code})")
        except requests.exceptions.Timeout:
            print(f"  ✗ Timeout downloading {filename}")
        except Exception as e:
            print(f"  ✗ Error downloading {filename}: {str(e)[:50]}")
    
    return downloaded[:5]  # Return max 5 images for testing

def find_local_images():
    """Find all local images in current directory and subdirectories"""
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')
    local_images = []
    
    for root, dirs, files in os.walk('.'):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        for file in files:
            if file.lower().endswith(image_extensions):
                full_path = os.path.join(root, file)
                # Skip if in phase3/results or other output directories
                if 'phase3/results' not in full_path and 'results/' not in root:
                    local_images.append(full_path)
    
    return local_images[:10]  # Return first 10 found

def run_multi_image_evaluation(max_images=3):
    """Run evaluation on multiple images"""
    
    print("=" * 70)
    print("MULTI-IMAGE ZERO-SHOT DETECTION EVALUATION")
    print("=" * 70)
    
    # Find available images
    print("\n1. Looking for test images...")
    test_images = find_local_images()
    
    if len(test_images) < 2:
        print("Not enough local images found. Downloading some test images...")
        downloaded = download_test_images()
        test_images = downloaded + test_images
    
    if not test_images:
        print("❌ No test images available!")
        print("\nPlease add some images to the current directory or")
        print("make sure you have internet connection for downloading.")
        return []
    
    print(f"\nFound {len(test_images)} test image(s):")
    for img in test_images[:max_images]:
        print(f"  • {os.path.basename(img)}")
    
    # Initialize pipeline and evaluator
    print("\n2. Initializing detection pipeline...")
    try:
        pipeline = ZeroShotDetectionPipeline(yolo_model='yolov8n.pt', conf_threshold=0.25)
        evaluator = RealTimeEvaluator(pipeline)
        print("  ✓ Pipeline initialized successfully")
    except Exception as e:
        print(f"  ✗ Failed to initialize pipeline: {e}")
        return []
    
    # Define comprehensive test scenarios
    test_scenarios = {
        'Urban Scene': {
            'classes': ['person', 'vehicle', 'building', 'traffic light', 'street sign'],
            'description': 'Common urban environment objects'
        },
        'Transportation': {
            'classes': ['bus', 'car', 'bicycle', 'motorcycle', 'truck', 'van'],
            'description': 'Various transportation vehicles'
        },
        'People Focus': {
            'classes': ['person', 'human', 'pedestrian', 'child', 'adult'],
            'description': 'Human-related detections'
        },
        'General Objects': {
            'classes': ['chair', 'table', 'computer', 'book', 'bottle'],
            'description': 'Common indoor objects'
        },
        'Zero-Shot Challenge': {
            'classes': ['red object', 'moving vehicle', 'large building', 'small animal'],
            'description': 'Novel/complex descriptions'
        }
    }
    
    all_results = []
    execution_times = []
    
    # Run tests
    print("\n3. Running evaluations...")
    
    for img_idx, image_path in enumerate(test_images[:max_images]):
        print(f"\n{'='*70}")
        print(f"IMAGE {img_idx+1}/{min(len(test_images), max_images)}: {os.path.basename(image_path)}")
        print(f"{'='*70}")
        
        # Skip if file doesn't exist
        if not os.path.exists(image_path):
            print(f"  ✗ File not found: {image_path}")
            continue
        
        # Test 2-3 scenarios per image (to save time)
        scenarios_to_test = list(test_scenarios.items())[:3]
        
        for scenario_name, scenario_info in scenarios_to_test:
            target_classes = scenario_info['classes']
            print(f"\n📊 Scenario: {scenario_name}")
            print(f"   Description: {scenario_info['description']}")
            print(f"   Target classes: {target_classes}")
            
            try:
                start_time = time.time()
                results = evaluator.evaluate_zero_shot(image_path, target_classes)
                end_time = time.time()
                
                execution_time = end_time - start_time
                execution_times.append(execution_time)
                
                all_results.append({
                    'image': image_path,
                    'scenario': scenario_name,
                    'target_classes': target_classes,
                    'metrics': results['metrics'],
                    'execution_time': execution_time,
                    'timestamp': datetime.now().isoformat()
                })
                
                # Print quick summary
                metrics = results['metrics']
                print(f"   Results: Precision={metrics['precision']:.3f}, "
                      f"Recall={metrics['recall']:.3f}, F1={metrics['f1_score']:.3f}")
                print(f"   Time: {execution_time:.2f}s")
                
            except KeyboardInterrupt:
                print("\n⚠️  Evaluation interrupted by user")
                generate_final_report(all_results, execution_times)
                return all_results
            except Exception as e:
                print(f"   ✗ Error: {str(e)[:100]}")
                continue
    
    # Generate final report
    generate_final_report(all_results, execution_times)
    
    return all_results

def generate_final_report(all_results, execution_times=None):
    """Generate comprehensive report and save to file"""
    
    print("\n" + "=" * 70)
    print("FINAL EVALUATION REPORT")
    print("=" * 70)
    
    if not all_results:
        print("No results to report.")
        return
    
    # Create results directory
    os.makedirs('phase3/results', exist_ok=True)
    
    # Calculate statistics
    num_tests = len(all_results)
    avg_precision = sum(r['metrics']['precision'] for r in all_results) / num_tests
    avg_recall = sum(r['metrics']['recall'] for r in all_results) / num_tests
    avg_f1 = sum(r['metrics']['f1_score'] for r in all_results) / num_tests
    avg_accuracy = sum(r['metrics']['accuracy'] for r in all_results) / num_tests
    avg_iou = sum(r['metrics']['avg_iou'] for r in all_results) / num_tests
    
    if execution_times:
        avg_time = sum(execution_times) / len(execution_times)
        max_time = max(execution_times)
        min_time = min(execution_times)
    else:
        avg_time = max_time = min_time = 0
    
    print(f"\n📈 Overall Performance ({num_tests} tests):")
    print(f"  Average Precision:  {avg_precision:.3f}")
    print(f"  Average Recall:     {avg_recall:.3f}")
    print(f"  Average F1 Score:   {avg_f1:.3f}")
    print(f"  Average Accuracy:   {avg_accuracy:.3f}")
    print(f"  Average IoU:        {avg_iou:.3f}")
    
    if avg_time > 0:
        print(f"  Average Time/Test: {avg_time:.2f}s (min: {min_time:.2f}s, max: {max_time:.2f}s)")
    
    # Find best and worst performing tests
    best_test = max(all_results, key=lambda x: x['metrics']['f1_score'])
    worst_test = min(all_results, key=lambda x: x['metrics']['f1_score'])
    
    print(f"\n🏆 Best Performance:")
    print(f"  Image: {os.path.basename(best_test['image'])}")
    print(f"  Scenario: {best_test['scenario']}")
    print(f"  F1 Score: {best_test['metrics']['f1_score']:.3f}")
    
    print(f"\n📉 Worst Performance:")
    print(f"  Image: {os.path.basename(worst_test['image'])}")
    print(f"  Scenario: {worst_test['scenario']}")
    print(f"  F1 Score: {worst_test['metrics']['f1_score']:.3f}")
    
    print("\n📋 Detailed Results:")
    for i, result in enumerate(all_results, 1):
        metrics = result['metrics']
        print(f"\n  Test {i}:")
        print(f"    Image: {os.path.basename(result['image'])}")
        print(f"    Scenario: {result['scenario']}")
        print(f"    Precision: {metrics['precision']:.3f}")
        print(f"    Recall: {metrics['recall']:.3f}")
        print(f"    F1 Score: {metrics['f1_score']:.3f}")
        print(f"    Detections: {metrics['num_zero_shot']} (YOLO: {metrics['num_yolo']})")
        print(f"    Correct: {metrics['num_correct']}/{metrics['num_matches']}")
        if 'execution_time' in result:
            print(f"    Time: {result['execution_time']:.2f}s")
    
    # Save detailed report to JSON
    report_data = {
        'summary': {
            'total_tests': num_tests,
            'avg_precision': float(avg_precision),
            'avg_recall': float(avg_recall),
            'avg_f1_score': float(avg_f1),
            'avg_accuracy': float(avg_accuracy),
            'avg_iou': float(avg_iou),
            'avg_execution_time': float(avg_time) if execution_times else 0,
            'generated_at': datetime.now().isoformat()
        },
        'best_test': {
            'image': best_test['image'],
            'scenario': best_test['scenario'],
            'metrics': best_test['metrics']
        },
        'worst_test': {
            'image': worst_test['image'],
            'scenario': worst_test['scenario'],
            'metrics': worst_test['metrics']
        },
        'all_tests': all_results
    }
    
    report_file = f"phase3/results/final_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump(report_data, f, indent=2)
    print(f"\n✅ Detailed report saved to: {report_file}")
    
    print("\n" + "=" * 70)
    print("KEY FINDINGS & RECOMMENDATIONS:")
    print("=" * 70)
    
    recommendations = []
    
    if avg_f1 > 0.7:
        recommendations.append("✓ Excellent zero-shot performance achieved!")
    elif avg_f1 > 0.5:
        recommendations.append("✓ Good zero-shot performance")
    elif avg_f1 > 0.3:
        recommendations.append("✓ Moderate zero-shot performance")
    else:
        recommendations.append("⚠️  Zero-shot performance needs improvement")
    
    if avg_precision < avg_recall:
        recommendations.append("⚠️  High false positives - consider increasing confidence threshold")
    elif avg_recall < avg_precision:
        recommendations.append("⚠️  Low recall - some objects missed, try better prompts")
    
    if avg_time > 5:
        recommendations.append("⚠️  Slow execution - consider using smaller YOLO model")
    
    for rec in recommendations:
        print(f"  {rec}")
    
    print(f"""
    
    PROJECT STATUS: {'✅ SUCCESSFUL' if avg_f1 > 0.4 else '⚠️  NEEDS IMPROVEMENT'}
    
    The zero-shot object detection system is {'working well' if avg_f1 > 0.4 else 'functional but needs optimization'}.
    
    Next steps:
    1. {'Fine-tune prompts for better accuracy' if avg_f1 < 0.6 else 'Try more challenging test cases'}
    2. {'Optimize pipeline for faster execution' if avg_time > 3 else 'Experiment with different object categories'}
    3. Test on video streams for real-time performance
    4. Compare with other zero-shot approaches
    
    Remember: Zero-shot detection enables detecting objects not in YOLO's training set!
    This is a significant advantage over traditional object detectors.
    """)

def quick_test():
    """Run a quick test with minimal images and scenarios"""
    print("🚀 Running quick test...")
    results = run_multi_image_evaluation(max_images=1)
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test zero-shot detection on multiple images')
    parser.add_argument('--quick', action='store_true', help='Run quick test with 1 image')
    parser.add_argument('--max-images', type=int, default=3, help='Maximum number of images to test')
    parser.add_argument('--local-only', action='store_true', help='Use only local images, no downloads')
    
    args = parser.parse_args()
    
    if args.quick:
        quick_test()
    else:
        if args.local_only:
            print("Using local images only (no downloads)")
        run_multi_image_evaluation(max_images=args.max_images)