import cv2
import torch
from ultralytics import YOLO
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class YOLODetector:
    def __init__(self, model_type='yolov8l.pt', conf_threshold=0.25):
        """
        Initialize YOLO detector
        
        Args:
            model_type: YOLO model type or path
            conf_threshold: Confidence threshold for detections
        """
        self.model = YOLO(model_type)
        self.conf_threshold = conf_threshold
        print(f"Loaded YOLO model: {model_type}")
    
    def detect(self, image_path):
        """
        Perform object detection on image
        
        Args:
            image_path: Path to input image
            
        Returns:
            results: Detection results
        """
        # Run inference
        results = self.model(image_path, conf=self.conf_threshold)
        
        return results[0]  # Return first image results
    
    def extract_detections(self, results):
        """
        Extract bounding boxes and confidence scores from results
        
        Args:
            results: YOLO results object
            
        Returns:
            Tuple of (boxes, scores, class_ids)
        """
        if results.boxes is None:
            return np.array([]), np.array([]), np.array([])
        
        # Extract boxes in xyxy format
        boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy().astype(int)
        
        return boxes, scores, class_ids
    
    def visualize_detections(self, image_path, results, save_path=None):
        """
        Visualize YOLO detections
        
        Args:
            image_path: Path to input image
            results: YOLO results object
            save_path: Path to save visualization
        """
        # Plot using YOLO's built-in plotting
        plotted_image = results.plot()
        plotted_image = cv2.cvtColor(plotted_image, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(12, 8))
        plt.imshow(plotted_image)
        plt.axis('off')
        plt.title('YOLO Detections')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"Detection visualization saved to: {save_path}")
        
        plt.show()
        
        return plotted_image

# Test function
def test_yolo_detector():
    """Test the YOLO detector with a sample image"""
    detector = YOLODetector('yolov8n.pt')  # Use nano model for quick testing
    
    # You can download a test image or use any local image
    test_image_path = "https://ultralytics.com/images/bus.jpg"
    
    print("Testing YOLO detector...")
    results = detector.detect(test_image_path)
    boxes, scores, class_ids = detector.extract_detections(results)
    
    print(f"Detected {len(boxes)} objects")
    print(f"Boxes shape: {boxes.shape}")
    print(f"Scores: {scores[:5]}")  # Print first 5 scores
    
    # Visualize results
    detector.visualize_detections(test_image_path, results)

if __name__ == "__main__":
    test_yolo_detector()