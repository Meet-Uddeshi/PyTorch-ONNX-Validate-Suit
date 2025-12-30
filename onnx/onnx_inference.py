"""
ONNX Runtime YOLO Inference Module

This module implements YOLO object detection inference using ONNX Runtime.
It performs the same operations as PyTorch inference but using the exported ONNX model.

Features:
- ONNX Runtime inference with GPU/CPU support
- Preprocessing and postprocessing matching PyTorch implementation
- PyTorch vs ONNX comparison and validation
- Numerical difference analysis with IoU computation
- Comparison visualization using matplotlib

Usage:
    # Run ONNX inference only
    python onnx/onnx_inference.py
    
    # Run comparison between PyTorch and ONNX
    python onnx/onnx_inference.py --compare

CRITICAL IMPORTANCE OF MATCHING PREPROCESSING AND POSTPROCESSING:

Why Exact Matching is Essential:
1. Model Consistency:
   - The ONNX model was exported from PyTorch with specific input/output formats
   - The model expects inputs in the EXACT same format it was trained with
   - Any deviation in preprocessing will cause incorrect predictions

2. Numerical Accuracy:
   - Small differences in preprocessing (e.g., normalization, resizing method) 
     accumulate through network layers and can cause significant output differences
   - Different resize algorithms (bilinear vs nearest) produce different pixel values
   - Color space mismatches (BGR vs RGB) completely change the input data

3. Coordinate System Alignment:
   - Bounding box coordinates are relative to the preprocessed image dimensions
   - If preprocessing differs, box coordinates won't map correctly to original image
   - Letterbox padding must be accounted for identically in both implementations

4. Postprocessing Logic:
   - NMS (Non-Maximum Suppression) parameters must match exactly
   - Confidence thresholds must be identical
   - Box format conversions (xywh to xyxy) must use the same logic
   - Class score extraction must follow the same indexing

5. Validation and Comparison:
   - To validate ONNX export correctness, outputs must be numerically close
   - Any preprocessing/postprocessing mismatch makes comparison meaningless
   - Debugging becomes impossible if pipelines differ

YOLO Preprocessing Pipeline:
1. Letterbox resize: Maintain aspect ratio, pad to square (640x640)
2. Color conversion: BGR (OpenCV) to RGB (model expects)
3. Normalization: Scale pixel values from [0, 255] to [0, 1]
4. Channel transpose: HWC (height, width, channels) to CHW (channels, height, width)
5. Add batch dimension: [3, 640, 640] to [1, 3, 640, 640]

YOLO Postprocessing Pipeline:
1. Extract output tensor: [1, 84, 8400] or [1, num_classes+4, num_anchors]
2. Transpose: [1, 84, 8400] to [1, 8400, 84] for easier processing
3. Extract boxes: First 4 values [x_center, y_center, width, height]
4. Extract class scores: Remaining 80 values (COCO dataset classes)
5. Filter by confidence: Keep only detections above threshold
6. Convert box format: xywh (center format) to xyxy (corner format)
7. Scale coordinates: Map from model space (640x640) to original image size
8. Apply NMS: Remove overlapping boxes using Intersection over Union (IoU)
9. Map class IDs to names: Convert numeric class IDs to human-readable labels
"""

import onnxruntime as ort
import cv2
import numpy as np
from pathlib import Path


class ONNXYOLOInference:
    """
    ONNX Runtime-based YOLO inference class for object detection.
    
    This class loads ONNX models, runs inference with ONNX Runtime,
    and processes results identically to the PyTorch implementation.
    """
    
    def __init__(self, model_path, conf_threshold=0.25, iou_threshold=0.45):
        """
        Initialize the ONNX YOLO inference model.
        
        Args:
            model_path: Path to the YOLO .onnx model file
            conf_threshold: Confidence threshold for filtering detections (0-1)
            iou_threshold: IoU threshold for Non-Maximum Suppression (0-1)
        
        Thresholds explanation:
        - conf_threshold: Minimum confidence score to consider a detection valid
          Lower values = more detections but more false positives
        - iou_threshold: Maximum overlap allowed between boxes (IoU = Intersection over Union)
          Lower values = more aggressive NMS, fewer overlapping boxes
        """
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.session = None
        self.input_name = None
        self.output_names = None
        self.input_shape = None
        
        # COCO dataset class names (80 classes)
        # This must match the order used during model training
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
            'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        
    def load_model(self):
        """
        Load the ONNX model and create an inference session.
        
        This function initializes ONNX Runtime with the model and configures
        execution providers (CPU or GPU). It also extracts input/output metadata.
        
        ONNX Runtime Execution Providers:
        - CUDAExecutionProvider: GPU acceleration (if available)
        - CPUExecutionProvider: CPU fallback (always available)
        
        The session is optimized for inference and can run on various hardware.
        """
        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")
            
        print(f"Loading ONNX model from: {self.model_path}")
        
        # Create ONNX Runtime session
        # Session options can be configured for optimization
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Set execution providers (GPU if available, otherwise CPU)
        # CUDAExecutionProvider will be used if onnxruntime-gpu is installed and CUDA is available
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        # Check available providers
        available_providers = ort.get_available_providers()
        print(f"Available execution providers: {available_providers}")
        
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=session_options,
            providers=providers
        )
        
        # Get input and output names
        # ONNX models have named inputs/outputs that must be used during inference
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        # Get input shape
        self.input_shape = self.session.get_inputs()[0].shape
        
        # Check which provider is actually being used
        active_provider = self.session.get_providers()[0]
        
        print(f"Model loaded successfully")
        print(f"Active execution provider: {active_provider}")
        
        if active_provider == 'CUDAExecutionProvider':
            print("GPU acceleration is ENABLED")
        elif active_provider == 'CPUExecutionProvider':
            print("WARNING: Running on CPU (GPU not available)")
            print("For GPU support, install: pip uninstall onnxruntime && pip install onnxruntime-gpu")
        
        print(f"Input name: {self.input_name}")
        print(f"Input shape: {self.input_shape}")
        print(f"Output names: {self.output_names}")
        print(f"Number of classes: {len(self.class_names)}")
        
    def load_image(self, image_path):
        """
        Load an image from the specified path.
        
        Args:
            image_path: Path to the input image file
            
        Returns:
            numpy.ndarray: Loaded image in BGR format (OpenCV format)
        
        OpenCV loads images in BGR format by default, which we'll convert
        to RGB during preprocessing to match the model's training format.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
            
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
            
        print(f"Image loaded: {image_path} (Shape: {image.shape})")
        return image
        
    def letterbox_resize(self, image, new_shape=(640, 640), color=(114, 114, 114)):
        """
        Resize image with aspect ratio preservation using letterbox method.
        
        Args:
            image: Input image as numpy array
            new_shape: Target size (height, width)
            color: Padding color (BGR format)
            
        Returns:
            tuple: (resized_image, ratio, (pad_width, pad_height))
            
        Letterbox resizing:
        - Maintains original aspect ratio (no distortion)
        - Scales image to fit within target size
        - Adds padding (letterbox bars) to reach exact target size
        - Padding is added symmetrically on both sides
        
        This is critical because:
        - The model was trained with letterbox preprocessing
        - Distorted images (simple resize) would break the model
        - We need the ratio and padding to map predictions back to original image
        """
        shape = image.shape[:2]  # Current shape [height, width]
        
        # Calculate scaling ratio (maintain aspect ratio)
        ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        
        # Compute new unpadded size
        new_unpad = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))
        
        # Calculate padding
        pad_width = new_shape[1] - new_unpad[0]
        pad_height = new_shape[0] - new_unpad[1]
        
        # Divide padding into 2 sides
        pad_width /= 2
        pad_height /= 2
        
        # Resize image
        if shape[::-1] != new_unpad:
            image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
            
        # Add padding
        top, bottom = int(round(pad_height - 0.1)), int(round(pad_height + 0.1))
        left, right = int(round(pad_width - 0.1)), int(round(pad_width + 0.1))
        image = cv2.copyMakeBorder(
            image, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=color
        )
        
        return image, ratio, (pad_width, pad_height)
        
    def preprocess(self, image):
        """
        Preprocess image for YOLO inference - MUST match PyTorch preprocessing.
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            tuple: (preprocessed_image, original_image, ratio, padding)
            
        Preprocessing steps (order matters):
        1. Store original image for later visualization
        2. Letterbox resize to 640x640 (maintains aspect ratio)
        3. Convert BGR to RGB (model expects RGB)
        4. Normalize to [0, 1] by dividing by 255.0
        5. Transpose from HWC to CHW format (channels first)
        6. Add batch dimension [1, 3, 640, 640]
        7. Convert to float32 (required by model)
        
        Each step is critical and must match PyTorch exactly.
        """
        original_image = image.copy()
        
        # Letterbox resize
        image, ratio, padding = self.letterbox_resize(image, new_shape=(640, 640))
        
        # Convert BGR to RGB
        # OpenCV uses BGR, but the model was trained with RGB images
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        # Neural networks work better with normalized inputs
        image = image.astype(np.float32) / 255.0
        
        # Transpose from HWC to CHW
        # PyTorch/ONNX models expect channels-first format
        image = np.transpose(image, (2, 0, 1))
        
        # Add batch dimension
        # Models expect batched inputs even for single images
        image = np.expand_dims(image, axis=0)
        
        print(f"Preprocessed image shape: {image.shape}")
        
        return image, original_image, ratio, padding
        
    def run_inference(self, preprocessed_image):
        """
        Run ONNX inference on the preprocessed image.
        
        Args:
            preprocessed_image: Preprocessed image tensor [1, 3, 640, 640]
            
        Returns:
            numpy.ndarray: Raw model output
            
        This function performs the forward pass through the ONNX model.
        The output format depends on the YOLO version:
        - YOLOv8/YOLO11: [1, 84, 8400] (4 box coords + 80 class scores, 8400 anchors)
        
        The raw output needs postprocessing to extract meaningful predictions.
        """
        if self.session is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
            
        # Check which device is being used
        device_type = self.session.get_providers()[0]
        if device_type == 'CUDAExecutionProvider':
            print("Running ONNX inference on GPU...")
        else:
            print("Running ONNX inference on CPU...")
        
        # Run inference
        # ONNX Runtime expects a dictionary mapping input names to tensors
        outputs = self.session.run(
            self.output_names,
            {self.input_name: preprocessed_image}
        )
        
        print(f"Inference complete. Output shape: {outputs[0].shape}")
        
        return outputs[0]
        
    def postprocess(self, output, original_image, ratio, padding):
        """
        Postprocess ONNX output to extract bounding boxes, classes, and scores.
        
        Args:
            output: Raw ONNX model output [1, 84, 8400]
            original_image: Original input image (for coordinate scaling)
            ratio: Scaling ratio from letterbox resize
            padding: Padding values (pad_width, pad_height) from letterbox
            
        Returns:
            tuple: (boxes, class_names, confidences)
            
        Postprocessing pipeline:
        1. Transpose output from [1, 84, 8400] to [8400, 84]
        2. Extract box coordinates (first 4 values) and class scores (remaining 80)
        3. Find maximum class score for each detection
        4. Filter by confidence threshold
        5. Convert boxes from xywh (center) to xyxy (corners) format
        6. Scale boxes back to original image size
        7. Apply Non-Maximum Suppression to remove overlapping boxes
        8. Map class IDs to human-readable names
        """
        # Transpose output from [1, 84, 8400] to [8400, 84]
        # This makes it easier to iterate over detections
        output = output[0].transpose()  # Shape: [8400, 84]
        
        # Extract boxes and scores
        boxes_xywh = output[:, :4]  # Box coordinates: [x_center, y_center, width, height]
        class_scores = output[:, 4:]  # Class scores: [80 classes]
        
        # Get maximum class score and corresponding class ID for each detection
        max_scores = np.max(class_scores, axis=1)
        max_classes = np.argmax(class_scores, axis=1)
        
        # Filter by confidence threshold
        # Only keep detections with confidence above threshold
        valid_indices = max_scores >= self.conf_threshold
        
        boxes_xywh = boxes_xywh[valid_indices]
        max_scores = max_scores[valid_indices]
        max_classes = max_classes[valid_indices]
        
        print(f"Detections after confidence filtering: {len(boxes_xywh)}")
        
        if len(boxes_xywh) == 0:
            return [], [], []
            
        # Convert boxes from xywh to xyxy format
        # xywh: [x_center, y_center, width, height]
        # xyxy: [x1, y1, x2, y2] (top-left and bottom-right corners)
        boxes_xyxy = self.xywh2xyxy(boxes_xywh)
        
        # Scale boxes back to original image size
        # Account for letterbox padding and scaling ratio
        boxes_xyxy = self.scale_boxes(
            boxes_xyxy, ratio, padding, original_image.shape
        )
        
        # Apply Non-Maximum Suppression
        # This removes overlapping boxes for the same object
        indices = self.non_max_suppression(boxes_xyxy, max_scores)
        
        print(f"Detections after NMS: {len(indices)}")
        
        # Extract final predictions
        final_boxes = boxes_xyxy[indices]
        final_scores = max_scores[indices]
        final_classes = max_classes[indices]
        
        # Convert class IDs to names
        final_class_names = [self.class_names[class_id] for class_id in final_classes]
        
        return final_boxes, final_class_names, final_scores
        
    def xywh2xyxy(self, boxes_xywh):
        """
        Convert bounding boxes from xywh to xyxy format.
        
        Args:
            boxes_xywh: Boxes in [x_center, y_center, width, height] format
            
        Returns:
            numpy.ndarray: Boxes in [x1, y1, x2, y2] format
            
        Conversion formula:
        - x1 = x_center - width / 2 (left edge)
        - y1 = y_center - height / 2 (top edge)
        - x2 = x_center + width / 2 (right edge)
        - y2 = y_center + height / 2 (bottom edge)
        """
        boxes_xyxy = np.copy(boxes_xywh)
        boxes_xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2  # x1
        boxes_xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2  # y1
        boxes_xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2  # x2
        boxes_xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2  # y2
        return boxes_xyxy
        
    def scale_boxes(self, boxes, ratio, padding, original_shape):
        """
        Scale boxes from model input size back to original image size.
        
        Args:
            boxes: Bounding boxes in xyxy format (on 640x640 input)
            ratio: Scaling ratio from letterbox resize
            padding: Padding values (pad_width, pad_height)
            original_shape: Original image shape (height, width, channels)
            
        Returns:
            numpy.ndarray: Scaled boxes in original image coordinates
            
        Reverse letterbox transformation:
        1. Subtract padding (remove letterbox bars)
        2. Divide by scaling ratio (reverse resize)
        3. Clip to image boundaries (ensure boxes are within image)
        
        This is critical to display boxes at correct positions on original image.
        """
        # Remove padding
        boxes[:, [0, 2]] -= padding[0]  # Remove horizontal padding from x coordinates
        boxes[:, [1, 3]] -= padding[1]  # Remove vertical padding from y coordinates
        
        # Reverse scaling
        boxes /= ratio
        
        # Clip boxes to image boundaries
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, original_shape[1])  # Clip x to width
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, original_shape[0])  # Clip y to height
        
        return boxes
        
    def non_max_suppression(self, boxes, scores):
        """
        Apply Non-Maximum Suppression to remove overlapping boxes.
        
        Args:
            boxes: Bounding boxes in xyxy format
            scores: Confidence scores for each box
            
        Returns:
            list: Indices of boxes to keep
            
        NMS Algorithm:
        1. Sort boxes by confidence score (highest first)
        2. Take the box with highest score
        3. Remove all boxes that overlap with it significantly (IoU > threshold)
        4. Repeat until no boxes remain
        
        This prevents multiple detections of the same object.
        IoU (Intersection over Union) measures box overlap:
        - IoU = Area of Overlap / Area of Union
        - Higher IoU = more overlap
        """
        # Use OpenCV's NMS implementation (fast and reliable)
        indices = cv2.dnn.NMSBoxes(
            bboxes=boxes.tolist(),
            scores=scores.tolist(),
            score_threshold=self.conf_threshold,
            nms_threshold=self.iou_threshold
        )
        
        # OpenCV returns indices as a list of lists or array
        if len(indices) > 0:
            indices = indices.flatten()
        else:
            indices = []
            
        return indices
        
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
            
        Visual style must match PyTorch implementation for consistency:
        - Green bounding box rectangles
        - Class name and confidence percentage as labels
        - Background rectangle for text readability
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
        2. Preprocess (resize, normalize, format)
        3. Run ONNX inference
        4. Postprocess (extract boxes, apply NMS)
        5. Draw bounding boxes on the image
        6. Save the annotated result
        """
        # Load the input image
        image = self.load_image(image_path)
        
        # Preprocess the image
        preprocessed_image, original_image, ratio, padding = self.preprocess(image)
        
        # Run inference
        output = self.run_inference(preprocessed_image)
        
        # Postprocess to extract predictions
        boxes, class_names, confidences = self.postprocess(
            output, original_image, ratio, padding
        )
        
        # Draw predictions on the original image
        annotated_image = self.draw_predictions(
            original_image, boxes, class_names, confidences
        )
        
        # Save the result
        self.save_result(annotated_image, output_path)
        
        # Print detection summary
        print("\n=== Detection Summary ===")
        for i, (box, class_name, confidence) in enumerate(zip(boxes, class_names, confidences)):
            print(f"Detection {i+1}: {class_name} (confidence: {confidence:.3f}) at {box}")
        print("========================\n")
        
        return boxes, class_names, confidences


class ONNXPyTorchComparison:
    """
    Comparison class for analyzing differences between PyTorch and ONNX inference results.
    
    This class compares detection outputs from both implementations to validate
    that the ONNX export is correct and numerically equivalent to PyTorch.
    
    Why Outputs Should Be Similar:
    1. Same Model Weights:
       - ONNX model is exported directly from PyTorch weights
       - No training or weight updates occur during export
       - Weights are bit-identical between formats
    
    2. Same Computational Graph:
       - ONNX preserves the exact neural network architecture
       - Layer operations are mathematically equivalent
       - Forward pass logic is identical
    
    3. Same Preprocessing:
       - Both use letterbox resize with identical parameters
       - Same normalization (divide by 255)
       - Same color space conversion (BGR to RGB)
       - Same tensor format (CHW, batch dimension)
    
    4. Same Postprocessing:
       - Identical confidence thresholding
       - Same NMS parameters (IoU threshold)
       - Same box coordinate transformations
    
    Why Small Numerical Differences May Exist:
    1. Floating-Point Precision:
       - Different libraries may use slightly different floating-point operations
       - Rounding errors accumulate differently in PyTorch vs ONNX Runtime
       - Small differences in order of operations can affect final digits
       - Example: 0.8234567 vs 0.8234569 (difference in 7th decimal place)
    
    2. Operator Implementation:
       - PyTorch and ONNX Runtime may implement operators slightly differently
       - Optimizations (SIMD, GPU kernels) can introduce tiny variations
       - Different math libraries (MKL, cuBLAS) have implementation differences
    
    3. Non-Deterministic Operations:
       - Some GPU operations are non-deterministic for performance
       - Parallel reduction operations may sum in different orders
       - Atomic operations on GPU can have race conditions
    
    4. NMS Variations:
       - Non-Maximum Suppression can produce slightly different results
       - When two boxes have nearly identical scores, ordering can vary
       - This can cause one implementation to suppress a box that the other keeps
    
    Expected Tolerance:
    - Box coordinates: Within 1-2 pixels (negligible visual difference)
    - Confidence scores: Within 0.001-0.01 (0.1-1% relative difference)
    - Same detections: 95%+ of boxes should match
    - Class predictions: Should be identical (same argmax)
    
    If differences exceed these tolerances:
    - Preprocessing mismatch (most common cause)
    - Wrong model version exported
    - Bug in postprocessing logic
    - Incorrect coordinate transformations
    """
    
    def __init__(self):
        """
        Initialize the comparison analyzer.
        
        This class computes metrics to validate ONNX export correctness
        by comparing against PyTorch ground truth.
        """
        self.pytorch_boxes = None
        self.pytorch_classes = None
        self.pytorch_confidences = None
        
        self.onnx_boxes = None
        self.onnx_classes = None
        self.onnx_confidences = None
        
    def set_pytorch_results(self, boxes, class_names, confidences):
        """
        Store PyTorch inference results for comparison.
        
        Args:
            boxes: List of bounding boxes from PyTorch [x1, y1, x2, y2]
            class_names: List of class names
            confidences: List of confidence scores
        """
        self.pytorch_boxes = np.array(boxes) if len(boxes) > 0 else np.array([])
        self.pytorch_classes = class_names
        self.pytorch_confidences = np.array(confidences) if len(confidences) > 0 else np.array([])
        
    def set_onnx_results(self, boxes, class_names, confidences):
        """
        Store ONNX inference results for comparison.
        
        Args:
            boxes: List of bounding boxes from ONNX [x1, y1, x2, y2]
            class_names: List of class names
            confidences: List of confidence scores
        """
        self.onnx_boxes = np.array(boxes) if len(boxes) > 0 else np.array([])
        self.onnx_classes = class_names
        self.onnx_confidences = np.array(confidences) if len(confidences) > 0 else np.array([])
        
    def compute_iou(self, box1, box2):
        """
        Compute Intersection over Union (IoU) between two bounding boxes.
        
        Args:
            box1: First box [x1, y1, x2, y2]
            box2: Second box [x1, y1, x2, y2]
            
        Returns:
            float: IoU value between 0 and 1
            
        IoU measures how much two boxes overlap:
        - IoU = Area of Intersection / Area of Union
        - IoU = 1.0: Perfect overlap (boxes are identical)
        - IoU = 0.0: No overlap at all
        - IoU > 0.5: Generally considered a good match
        - IoU > 0.75: Excellent match
        
        IoU is the standard metric for comparing bounding boxes in object detection.
        """
        # Extract coordinates
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Compute intersection rectangle coordinates
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        # Compute intersection area
        intersection_width = max(0, x2_i - x1_i)
        intersection_height = max(0, y2_i - y1_i)
        intersection_area = intersection_width * intersection_height
        
        # Compute areas of both boxes
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # Compute union area
        union_area = box1_area + box2_area - intersection_area
        
        # Compute IoU
        if union_area == 0:
            return 0.0
        iou = intersection_area / union_area
        
        return iou
        
    def match_boxes(self, iou_threshold=0.5):
        """
        Match boxes between PyTorch and ONNX results using IoU.
        
        Args:
            iou_threshold: Minimum IoU to consider boxes as matching
            
        Returns:
            tuple: (matched_pairs, unmatched_pytorch, unmatched_onnx)
                - matched_pairs: List of (pytorch_idx, onnx_idx, iou) tuples
                - unmatched_pytorch: Indices of PyTorch boxes without matches
                - unmatched_onnx: Indices of ONNX boxes without matches
                
        Matching algorithm:
        1. Compute IoU between all PyTorch and ONNX box pairs
        2. Greedily match boxes with highest IoU first
        3. Each box can only match once (one-to-one matching)
        4. Only accept matches above IoU threshold
        
        This tells us which detections correspond to the same object
        in both implementations so we can compare their properties.
        """
        if len(self.pytorch_boxes) == 0 or len(self.onnx_boxes) == 0:
            return [], list(range(len(self.pytorch_boxes))), list(range(len(self.onnx_boxes)))
            
        matched_pairs = []
        matched_pytorch = set()
        matched_onnx = set()
        
        # Compute IoU matrix
        iou_matrix = np.zeros((len(self.pytorch_boxes), len(self.onnx_boxes)))
        for i, pt_box in enumerate(self.pytorch_boxes):
            for j, onnx_box in enumerate(self.onnx_boxes):
                iou_matrix[i, j] = self.compute_iou(pt_box, onnx_box)
        
        # Greedy matching: repeatedly find the highest IoU pair
        while True:
            # Find maximum IoU
            max_iou = np.max(iou_matrix)
            
            if max_iou < iou_threshold:
                break
                
            # Find indices of maximum IoU
            i, j = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            
            # Add to matched pairs
            matched_pairs.append((i, j, iou_matrix[i, j]))
            matched_pytorch.add(i)
            matched_onnx.add(j)
            
            # Remove matched boxes from consideration
            iou_matrix[i, :] = -1
            iou_matrix[:, j] = -1
        
        # Find unmatched boxes
        unmatched_pytorch = [i for i in range(len(self.pytorch_boxes)) if i not in matched_pytorch]
        unmatched_onnx = [j for j in range(len(self.onnx_boxes)) if j not in matched_onnx]
        
        return matched_pairs, unmatched_pytorch, unmatched_onnx
        
    def compute_metrics(self):
        """
        Compute comparison metrics between PyTorch and ONNX results.
        
        Returns:
            dict: Dictionary containing comparison metrics
            
        Metrics computed:
        - Number of detections (PyTorch vs ONNX)
        - Number of matched boxes
        - Average IoU for matched boxes
        - Confidence score differences
        - Class prediction agreement
        
        These metrics quantify how similar the outputs are.
        """
        metrics = {
            'pytorch_detections': len(self.pytorch_boxes),
            'onnx_detections': len(self.onnx_boxes),
            'matched_boxes': 0,
            'unmatched_pytorch': 0,
            'unmatched_onnx': 0,
            'avg_iou': 0.0,
            'max_iou': 0.0,
            'min_iou': 0.0,
            'avg_confidence_diff': 0.0,
            'max_confidence_diff': 0.0,
            'class_agreement': 0.0,
            'matched_pairs': []
        }
        
        if len(self.pytorch_boxes) == 0 and len(self.onnx_boxes) == 0:
            return metrics
            
        # Match boxes
        matched_pairs, unmatched_pytorch, unmatched_onnx = self.match_boxes(iou_threshold=0.5)
        
        metrics['matched_boxes'] = len(matched_pairs)
        metrics['unmatched_pytorch'] = len(unmatched_pytorch)
        metrics['unmatched_onnx'] = len(unmatched_onnx)
        metrics['matched_pairs'] = matched_pairs
        
        if len(matched_pairs) > 0:
            # IoU statistics
            ious = [iou for _, _, iou in matched_pairs]
            metrics['avg_iou'] = np.mean(ious)
            metrics['max_iou'] = np.max(ious)
            metrics['min_iou'] = np.min(ious)
            
            # Confidence differences
            conf_diffs = []
            class_agreements = []
            
            for pt_idx, onnx_idx, _ in matched_pairs:
                conf_diff = abs(self.pytorch_confidences[pt_idx] - self.onnx_confidences[onnx_idx])
                conf_diffs.append(conf_diff)
                
                class_match = 1.0 if self.pytorch_classes[pt_idx] == self.onnx_classes[onnx_idx] else 0.0
                class_agreements.append(class_match)
            
            metrics['avg_confidence_diff'] = np.mean(conf_diffs)
            metrics['max_confidence_diff'] = np.max(conf_diffs)
            metrics['class_agreement'] = np.mean(class_agreements)
        
        return metrics
        
    def print_comparison(self):
        """
        Print a detailed comparison report to the console.
        
        This provides a human-readable summary of how similar
        the PyTorch and ONNX outputs are.
        """
        print("\n" + "=" * 60)
        print("PyTorch vs ONNX Comparison Report")
        print("=" * 60)
        
        metrics = self.compute_metrics()
        
        print(f"\nDetection Counts:")
        print(f"  PyTorch detections: {metrics['pytorch_detections']}")
        print(f"  ONNX detections: {metrics['onnx_detections']}")
        print(f"  Difference: {abs(metrics['pytorch_detections'] - metrics['onnx_detections'])}")
        
        print(f"\nBox Matching (IoU > 0.5):")
        print(f"  Matched boxes: {metrics['matched_boxes']}")
        print(f"  Unmatched PyTorch boxes: {metrics['unmatched_pytorch']}")
        print(f"  Unmatched ONNX boxes: {metrics['unmatched_onnx']}")
        
        if metrics['matched_boxes'] > 0:
            print(f"\nIoU Statistics:")
            print(f"  Average IoU: {metrics['avg_iou']:.4f}")
            print(f"  Maximum IoU: {metrics['max_iou']:.4f}")
            print(f"  Minimum IoU: {metrics['min_iou']:.4f}")
            
            print(f"\nConfidence Score Differences:")
            print(f"  Average difference: {metrics['avg_confidence_diff']:.4f}")
            print(f"  Maximum difference: {metrics['max_confidence_diff']:.4f}")
            
            print(f"\nClass Prediction Agreement:")
            print(f"  Agreement rate: {metrics['class_agreement']*100:.1f}%")
            
            # Detailed matched pairs
            print(f"\nDetailed Matched Pairs:")
            for i, (pt_idx, onnx_idx, iou) in enumerate(metrics['matched_pairs'], 1):
                pt_class = self.pytorch_classes[pt_idx]
                onnx_class = self.onnx_classes[onnx_idx]
                pt_conf = self.pytorch_confidences[pt_idx]
                onnx_conf = self.onnx_confidences[onnx_idx]
                conf_diff = abs(pt_conf - onnx_conf)
                
                match_symbol = "✓" if pt_class == onnx_class else "✗"
                print(f"  {i}. IoU: {iou:.4f} | {match_symbol} {pt_class} | "
                      f"PyT conf: {pt_conf:.4f} | ONNX conf: {onnx_conf:.4f} | "
                      f"Diff: {conf_diff:.4f}")
        
        # Overall assessment
        print(f"\n" + "=" * 60)
        print("Assessment:")
        
        if metrics['matched_boxes'] == 0:
            print("  ⚠ WARNING: No matching boxes found!")
            print("  Check preprocessing and postprocessing alignment.")
        elif metrics['avg_iou'] > 0.9 and metrics['avg_confidence_diff'] < 0.01:
            print("  ✓ EXCELLENT: Outputs are nearly identical.")
            print("  ONNX export is correct and numerically accurate.")
        elif metrics['avg_iou'] > 0.75 and metrics['avg_confidence_diff'] < 0.05:
            print("  ✓ GOOD: Outputs are very similar.")
            print("  Small differences are within expected tolerances.")
        else:
            print("  ⚠ WARNING: Outputs differ significantly.")
            print("  Review preprocessing and postprocessing for mismatches.")
        
        print("=" * 60 + "\n")
        
    def generate_comparison_graph(self, output_path="output/comparison.png"):
        """
        Generate a comparison visualization using matplotlib.
        
        Args:
            output_path: Path to save the comparison graph
            
        This function creates a multi-panel figure showing:
        1. Detection count comparison (bar chart)
        2. Confidence score comparison (scatter plot)
        3. IoU distribution (histogram)
        4. Box coordinate differences (box plot)
        
        Visual comparison helps quickly assess ONNX export quality.
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        # Compute metrics
        metrics = self.compute_metrics()
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('PyTorch vs ONNX Inference Comparison', fontsize=16, fontweight='bold')
        
        # Subplot 1: Detection Counts
        ax1 = axes[0, 0]
        counts = [metrics['pytorch_detections'], metrics['onnx_detections']]
        bars = ax1.bar(['PyTorch', 'ONNX'], counts, color=['#4CAF50', '#2196F3'], alpha=0.7, edgecolor='black')
        ax1.set_ylabel('Number of Detections', fontsize=11)
        ax1.set_title('Detection Count Comparison', fontsize=12, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Subplot 2: Confidence Score Comparison
        ax2 = axes[0, 1]
        if metrics['matched_boxes'] > 0:
            pt_confs = [self.pytorch_confidences[pt_idx] for pt_idx, _, _ in metrics['matched_pairs']]
            onnx_confs = [self.onnx_confidences[onnx_idx] for _, onnx_idx, _ in metrics['matched_pairs']]
            
            ax2.scatter(pt_confs, onnx_confs, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
            
            # Plot diagonal line (perfect agreement)
            min_val = min(min(pt_confs), min(onnx_confs))
            max_val = max(max(pt_confs), max(onnx_confs))
            ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Agreement')
            
            ax2.set_xlabel('PyTorch Confidence', fontsize=11)
            ax2.set_ylabel('ONNX Confidence', fontsize=11)
            ax2.set_title('Confidence Score Comparison', fontsize=12, fontweight='bold')
            ax2.legend()
            ax2.grid(alpha=0.3)
            
            # Add correlation coefficient
            corr = np.corrcoef(pt_confs, onnx_confs)[0, 1]
            ax2.text(0.05, 0.95, f'Correlation: {corr:.4f}',
                    transform=ax2.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            ax2.text(0.5, 0.5, 'No matched boxes', ha='center', va='center', fontsize=12)
            ax2.set_title('Confidence Score Comparison', fontsize=12, fontweight='bold')
        
        # Subplot 3: IoU Distribution
        ax3 = axes[1, 0]
        if metrics['matched_boxes'] > 0:
            ious = [iou for _, _, iou in metrics['matched_pairs']]
            ax3.hist(ious, bins=20, color='#FF9800', alpha=0.7, edgecolor='black')
            ax3.axvline(metrics['avg_iou'], color='red', linestyle='--', linewidth=2, label=f'Mean: {metrics["avg_iou"]:.3f}')
            ax3.set_xlabel('IoU (Intersection over Union)', fontsize=11)
            ax3.set_ylabel('Frequency', fontsize=11)
            ax3.set_title('IoU Distribution for Matched Boxes', fontsize=12, fontweight='bold')
            ax3.legend()
            ax3.grid(axis='y', alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No matched boxes', ha='center', va='center', fontsize=12)
            ax3.set_title('IoU Distribution', fontsize=12, fontweight='bold')
        
        # Subplot 4: Summary Statistics
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Create summary text
        summary_text = "Summary Statistics\n" + "─" * 40 + "\n\n"
        summary_text += f"Detection Counts:\n"
        summary_text += f"  • PyTorch: {metrics['pytorch_detections']}\n"
        summary_text += f"  • ONNX: {metrics['onnx_detections']}\n\n"
        
        summary_text += f"Matching Results:\n"
        summary_text += f"  • Matched boxes: {metrics['matched_boxes']}\n"
        summary_text += f"  • Unmatched PyTorch: {metrics['unmatched_pytorch']}\n"
        summary_text += f"  • Unmatched ONNX: {metrics['unmatched_onnx']}\n\n"
        
        if metrics['matched_boxes'] > 0:
            summary_text += f"IoU Statistics:\n"
            summary_text += f"  • Average: {metrics['avg_iou']:.4f}\n"
            summary_text += f"  • Range: [{metrics['min_iou']:.4f}, {metrics['max_iou']:.4f}]\n\n"
            
            summary_text += f"Confidence Differences:\n"
            summary_text += f"  • Average: {metrics['avg_confidence_diff']:.4f}\n"
            summary_text += f"  • Maximum: {metrics['max_confidence_diff']:.4f}\n\n"
            
            summary_text += f"Class Agreement: {metrics['class_agreement']*100:.1f}%\n\n"
            
            # Overall verdict
            if metrics['avg_iou'] > 0.9 and metrics['avg_confidence_diff'] < 0.01:
                verdict = "✓ EXCELLENT\nOutputs are nearly identical"
                color = 'green'
            elif metrics['avg_iou'] > 0.75 and metrics['avg_confidence_diff'] < 0.05:
                verdict = "✓ GOOD\nOutputs are very similar"
                color = 'blue'
            else:
                verdict = "⚠ WARNING\nOutputs differ significantly"
                color = 'red'
        else:
            verdict = "⚠ ERROR\nNo matching boxes found"
            color = 'red'
        
        ax4.text(0.1, 0.95, summary_text, transform=ax4.transAxes,
                fontsize=10, verticalalignment='top', family='monospace')
        
        ax4.text(0.5, 0.1, verdict, transform=ax4.transAxes,
                fontsize=12, fontweight='bold', ha='center', va='center',
                color=color, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Adjust layout and save
        plt.tight_layout()
        
        # Create output directory if needed
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
        print(f"Comparison graph saved to: {output_path}")
        
        plt.close()


def main():
    """
    Main function to demonstrate ONNX YOLO inference.
    
    This function creates an instance of ONNXYOLOInference,
    loads the model, and processes the input image to generate
    an annotated output with object detections.
    """
    # Define paths
    model_path = "models/yolo11n.onnx"
    image_path = "data/image.jpeg"
    output_path = "output/onnx_result.png"
    
    # Create inference instance
    # Confidence threshold: 0.25 (only show detections with >25% confidence)
    # IoU threshold: 0.45 (remove boxes with >45% overlap)
    inference = ONNXYOLOInference(
        model_path,
        conf_threshold=0.25,
        iou_threshold=0.45
    )
    
    # Load the model
    inference.load_model()
    
    # Process the image
    boxes, class_names, confidences = inference.process(image_path, output_path)
    
    print("ONNX inference completed successfully!")


def compare_pytorch_onnx():
    """
    Compare PyTorch and ONNX inference results on the same image.
    
    This function runs both implementations, compares their outputs,
    and generates a detailed comparison report and visualization.
    
    This demonstrates:
    - How to validate ONNX export correctness
    - Expected numerical differences between implementations
    - Proper methodology for comparing object detection outputs
    """
    import sys
    from pathlib import Path
    
    # Add project root to Python path to enable imports
    # This allows importing from yolo/ directory when running from project root
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    from yolo.pytorch_inference import PyTorchYOLOInference
    
    print("=" * 60)
    print("Running PyTorch vs ONNX Comparison")
    print("=" * 60)
    
    # Define paths
    pytorch_model_path = "models/yolo11n.pt"
    onnx_model_path = "models/yolo11n.onnx"
    image_path = "data/image.jpeg"
    pytorch_output_path = "output/pytorch_result.png"
    onnx_output_path = "output/onnx_result.png"
    comparison_output_path = "output/comparison.png"
    
    # Run PyTorch inference
    print("\n" + "─" * 60)
    print("Step 1: Running PyTorch Inference")
    print("─" * 60)
    pytorch_inference = PyTorchYOLOInference(pytorch_model_path)
    pytorch_inference.load_model()
    pt_boxes, pt_classes, pt_confidences = pytorch_inference.process(
        image_path, pytorch_output_path
    )
    
    # Run ONNX inference
    print("\n" + "─" * 60)
    print("Step 2: Running ONNX Inference")
    print("─" * 60)
    onnx_inference = ONNXYOLOInference(
        onnx_model_path,
        conf_threshold=0.25,
        iou_threshold=0.45
    )
    onnx_inference.load_model()
    onnx_boxes, onnx_classes, onnx_confidences = onnx_inference.process(
        image_path, onnx_output_path
    )
    
    # Compare results
    print("\n" + "─" * 60)
    print("Step 3: Comparing Results")
    print("─" * 60)
    comparison = ONNXPyTorchComparison()
    comparison.set_pytorch_results(pt_boxes, pt_classes, pt_confidences)
    comparison.set_onnx_results(onnx_boxes, onnx_classes, onnx_confidences)
    
    # Print comparison report
    comparison.print_comparison()
    
    # Generate comparison graph
    print("Generating comparison visualization...")
    comparison.generate_comparison_graph(comparison_output_path)
    
    print("\n" + "=" * 60)
    print("Comparison Complete!")
    print("=" * 60)
    print(f"\nOutputs saved:")
    print(f"  - PyTorch result: {pytorch_output_path}")
    print(f"  - ONNX result: {onnx_output_path}")
    print(f"  - Comparison graph: {comparison_output_path}")
    print()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--compare":
        # Run comparison mode
        compare_pytorch_onnx()
    else:
        # Run standard ONNX inference only
        main()

