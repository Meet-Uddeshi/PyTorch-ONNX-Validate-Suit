"""
PyTorch YOLO Inference Module

This module implements YOLO object detection inference using PyTorch and Ultralytics.
It loads a pre-trained YOLO model, processes input images, runs inference,
and generates annotated output images with bounding boxes, class labels, and confidence scores.

YOLO Inference Flow:
1. Load the YOLO model from a .pt file in evaluation mode
2. Load and preprocess the input image
3. Run forward pass through the model
4. Extract predictions (bounding boxes, class IDs, confidence scores)
5. Apply non-maximum suppression to filter overlapping detections
6. Draw bounding boxes with labels on the image
7. Save the annotated result

Output Structure:
- Bounding boxes: [x1, y1, x2, y2] coordinates in pixels
- Class names: Human-readable labels (e.g., 'person', 'car', 'dog')
- Confidence scores: Probability values between 0 and 1
"""

import torch
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO


class PyTorchYOLOInference:
    """
    PyTorch-based YOLO inference class for object detection.
    
    This class handles loading YOLO models, running inference on images,
    and visualizing detection results with bounding boxes.
    """
    
    def __init__(self, model_path, device=None):
        """
        Initialize the PyTorch YOLO inference model.
        
        Args:
            model_path: Path to the YOLO .pt model file
            device: Device to run inference on ('cuda' or 'cpu'). Auto-detects if None.
        
        The model is loaded in evaluation mode to ensure consistent inference behavior
        with dropout and batch normalization layers disabled.
        """
        self.model_path = Path(model_path)
        
        # Auto-detect device if not specified
        if device:
            self.device = device
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Print GPU information if using CUDA
        if self.device == 'cuda':
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"GPU Detected: {gpu_name} ({gpu_memory:.2f} GB)")
        else:
            print("Running on CPU (GPU not available)")
        
        self.model = None
        self.class_names = None
        
    def load_model(self):
        """
        Load the YOLO model from the specified path and set it to evaluation mode.
        
        This function loads the pre-trained YOLO model weights and extracts
        the class names from the model's metadata. The model is set to eval mode
        to disable training-specific operations like dropout.
        
        If the model file does not exist, it will be automatically downloaded
        from the Ultralytics model repository and saved to the models folder.
        """
        # Check if model file exists
        if not self.model_path.exists():
            print(f"Model not found at: {self.model_path}")
            print("Downloading model automatically...")
            
            # Create models directory if it doesn't exist
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Get model name from path (e.g., 'yolo11n.pt')
            model_name = self.model_path.name
            
            # Download the model using YOLO
            # When YOLO is called with just a model name, it downloads it automatically
            self.model = YOLO(model_name)
            
            # Save the downloaded model to the specified path
            import shutil
            downloaded_model_path = Path.home() / '.ultralytics' / 'weights' / model_name
            if downloaded_model_path.exists():
                shutil.copy(str(downloaded_model_path), str(self.model_path))
                print(f"Model downloaded and saved to: {self.model_path}")
        else:
            print(f"Loading PyTorch YOLO model from: {self.model_path}")
            self.model = YOLO(str(self.model_path))
        
        # Move model to the specified device (GPU or CPU)
        self.model.to(self.device)
        
        # Extract class names from the model
        self.class_names = self.model.names
        
        # Confirm device placement
        if self.device == 'cuda':
            print(f"Model loaded successfully on GPU ({torch.cuda.get_device_name(0)})")
            print(f"GPU Memory Allocated: {torch.cuda.memory_allocated(0) / (1024**2):.2f} MB")
        else:
            print(f"Model loaded successfully on CPU")
        
        print(f"Number of classes: {len(self.class_names)}")
        
    def load_image(self, image_path):
        """
        Load an image from the specified path.
        
        Args:
            image_path: Path to the input image file
            
        Returns:
            numpy.ndarray: Loaded image in BGR format (OpenCV format)
            
        The image is loaded using OpenCV which returns images in BGR format.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
            
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
            
        print(f"Image loaded: {image_path} (Shape: {image.shape})")
        return image
        
    def run_inference(self, image):
        """
        Run YOLO inference on the input image.
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            results: YOLO results object containing predictions
            
        This function performs the forward pass through the YOLO model.
        The model automatically handles:
        - Image preprocessing (resizing, normalization)
        - Anchor box generation
        - Non-maximum suppression (NMS) to filter overlapping detections
        - Confidence thresholding
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
            
        print(f"Running PyTorch inference on {self.device.upper()}...")
        
        # Run inference with the model
        # The model handles preprocessing internally
        results = self.model(image, verbose=False, device=self.device)
        
        # Show GPU memory usage if using CUDA
        if self.device == 'cuda':
            gpu_memory_used = torch.cuda.memory_allocated(0) / (1024**2)
            print(f"GPU Memory Used: {gpu_memory_used:.2f} MB")
        
        print(f"Inference complete. Detections found: {len(results[0].boxes)}")
        return results
        
    def extract_predictions(self, results):
        """
        Extract bounding boxes, class names, and confidence scores from inference results.
        
        Args:
            results: YOLO results object from inference
            
        Returns:
            tuple: (boxes, class_names, confidences)
                - boxes: List of bounding boxes as [x1, y1, x2, y2]
                - class_names: List of detected class names
                - confidences: List of confidence scores (0-1)
                
        The predictions are extracted from the first result (batch size 1).
        Each detection contains:
        - Bounding box coordinates in xyxy format (top-left and bottom-right)
        - Class ID which is mapped to human-readable class name
        - Confidence score indicating detection certainty
        """
        boxes = []
        class_names = []
        confidences = []
        
        # Extract predictions from the first result
        result = results[0]
        
        for box in result.boxes:
            # Extract bounding box coordinates (x1, y1, x2, y2)
            xyxy = box.xyxy[0].cpu().numpy()
            boxes.append(xyxy)
            
            # Extract class ID and map to class name
            class_id = int(box.cls[0].cpu().numpy())
            class_names.append(self.class_names[class_id])
            
            # Extract confidence score
            confidence = float(box.conf[0].cpu().numpy())
            confidences.append(confidence)
            
        return boxes, class_names, confidences
        
    def draw_predictions(self, image, boxes, class_names, confidences):
        """
        Draw bounding boxes with labels and confidence scores on the image.
        
        Args:
            image: Input image as numpy array (BGR format)
            boxes: List of bounding boxes as [x1, y1, x2, y2]
            class_names: List of class names for each detection
            confidences: List of confidence scores for each detection
            
        Returns:
            numpy.ndarray: Annotated image with bounding boxes and labels
            
        Each bounding box is drawn with:
        - Rectangle outline in green color
        - Label text showing class name and confidence percentage
        - Background rectangle for better label visibility
        """
        annotated_image = image.copy()
        
        for box, class_name, confidence in zip(boxes, class_names, confidences):
            # Convert coordinates to integers
            x1, y1, x2, y2 = map(int, box)
            
            # Draw bounding box rectangle
            color = (0, 255, 0)  # Green color in BGR
            thickness = 2
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, thickness)
            
            # Prepare label text with class name and confidence
            label = f"{class_name}: {confidence:.2f}"
            
            # Calculate text size for background rectangle
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(
                label, font, font_scale, font_thickness
            )
            
            # Draw background rectangle for text
            cv2.rectangle(
                annotated_image,
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width, y1),
                color,
                -1  # Filled rectangle
            )
            
            # Draw label text
            text_color = (0, 0, 0)  # Black text
            cv2.putText(
                annotated_image,
                label,
                (x1, y1 - baseline - 5),
                font,
                font_scale,
                text_color,
                font_thickness
            )
            
        return annotated_image
        
    def save_result(self, image, output_path):
        """
        Save the annotated image to the specified output path.
        
        Args:
            image: Annotated image as numpy array
            output_path: Path where the output image will be saved
            
        This function creates the output directory if it doesn't exist
        and saves the image using OpenCV.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        cv2.imwrite(str(output_path), image)
        print(f"Result saved to: {output_path}")
        
    def process(self, image_path, output_path):
        """
        Complete pipeline: load image, run inference, draw results, and save output.
        
        Args:
            image_path: Path to input image
            output_path: Path to save annotated output image
            
        Returns:
            tuple: (boxes, class_names, confidences) containing all detections
            
        This is the main entry point that orchestrates the entire inference pipeline:
        1. Load the input image
        2. Run inference to detect objects
        3. Extract predictions (boxes, classes, scores)
        4. Draw bounding boxes on the image
        5. Save the annotated result
        """
        # Load the input image
        image = self.load_image(image_path)
        
        # Run inference
        results = self.run_inference(image)
        
        # Extract predictions
        boxes, class_names, confidences = self.extract_predictions(results)
        
        # Draw predictions on the image
        annotated_image = self.draw_predictions(image, boxes, class_names, confidences)
        
        # Save the result
        self.save_result(annotated_image, output_path)
        
        # Print detection summary
        print("\n=== Detection Summary ===")
        for i, (box, class_name, confidence) in enumerate(zip(boxes, class_names, confidences)):
            print(f"Detection {i+1}: {class_name} (confidence: {confidence:.3f}) at {box}")
        print("========================\n")
        
        return boxes, class_names, confidences


def main():
    """
    Main function to demonstrate PyTorch YOLO inference.
    
    This function creates an instance of PyTorchYOLOInference,
    loads the model, and processes the input image to generate
    an annotated output with object detections.
    """
    # Define paths
    model_path = "models/yolo11n.pt"
    image_path = "data/image.jpeg"
    output_path = "output/pytorch_result.png"
    
    # Create inference instance
    inference = PyTorchYOLOInference(model_path)
    
    # Load the model
    inference.load_model()
    
    # Process the image
    boxes, class_names, confidences = inference.process(image_path, output_path)
    
    print("PyTorch inference completed successfully!")


if __name__ == "__main__":
    main()

