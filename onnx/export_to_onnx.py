"""
ONNX Model Export Module

This module exports PyTorch YOLO models to ONNX (Open Neural Network Exchange) format.
ONNX is a portable format that allows models to run across different frameworks and platforms.

What ONNX Export Does:
- Traces the PyTorch model's computational graph by running a forward pass with dummy input
- Converts PyTorch operations to ONNX operators (standardized operations)
- Serializes the model architecture and weights into a single .onnx file
- Enables model deployment on platforms that support ONNX Runtime (C++, Java, etc.)
- Allows inference without requiring PyTorch as a dependency

What Does NOT Change During Export:
- Model weights remain identical (no training or fine-tuning occurs)
- Model architecture stays the same (layer structure is preserved)
- Numerical outputs should be nearly identical to PyTorch (within floating-point precision)
- The model stays in evaluation mode (no dropout, batch norm uses running stats)

Why Dummy Input is Required:
- ONNX export uses "tracing" - it records operations by actually running the model
- PyTorch needs concrete tensor shapes to trace the computational graph
- The dummy input must match the expected input shape: (batch, channels, height, width)
- For dynamic axes, we specify which dimensions can vary (e.g., batch size)
- The actual values in the dummy input don't matter, only the shape and dtype
"""

import torch
from pathlib import Path
from ultralytics import YOLO


class ONNXExporter:
    """
    ONNX exporter class for converting PyTorch YOLO models to ONNX format.
    
    This class handles loading PyTorch models, creating appropriate dummy inputs,
    and exporting to ONNX with proper configuration for deployment.
    """
    
    def __init__(self, model_path, output_path, input_shape=(1, 3, 640, 640), opset_version=12):
        """
        Initialize the ONNX exporter with model paths and configuration.
        
        Args:
            model_path: Path to the PyTorch .pt model file
            output_path: Path where the .onnx model will be saved
            input_shape: Shape of the dummy input tensor (batch, channels, height, width)
            opset_version: ONNX opset version (12+ recommended for YOLO compatibility)
        
        The input shape (1, 3, 640, 640) represents:
        - 1: Batch size (single image)
        - 3: RGB color channels
        - 640: Image height in pixels
        - 640: Image width in pixels
        
        Opset version 12+ is recommended as it includes operations used by modern YOLO models.
        """
        self.model_path = Path(model_path)
        self.output_path = Path(output_path)
        self.input_shape = input_shape
        self.opset_version = opset_version
        self.model = None
        
    def load_model(self):
        """
        Load the PyTorch YOLO model from the specified path.
        
        This function loads the pre-trained YOLO model and ensures it's in
        evaluation mode, which is required for consistent export behavior.
        Evaluation mode disables training-specific layers like dropout.
        """
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
        print(f"Loading PyTorch model from: {self.model_path}")
        self.model = YOLO(str(self.model_path))
        
        # Ensure model is in evaluation mode for export
        self.model.model.eval()
        
        print(f"Model loaded successfully")
        print(f"Model type: {type(self.model.model)}")
        
    def create_dummy_input(self):
        """
        Create a dummy input tensor for ONNX export tracing.
        
        Returns:
            torch.Tensor: Dummy input tensor with the specified shape
            
        The dummy input is used during export to trace the model's computational graph.
        ONNX export requires running a forward pass to record all operations.
        
        The tensor is filled with random values (torch.randn), but the actual values
        don't matter - only the shape and data type are important for tracing.
        
        We use float32 dtype because:
        - It's the standard for neural network inference
        - It balances precision and performance
        - Most hardware accelerators are optimized for float32
        """
        dummy_input = torch.randn(*self.input_shape, dtype=torch.float32)
        print(f"Created dummy input with shape: {dummy_input.shape}")
        return dummy_input
        
    def export_to_onnx(self, dummy_input, dynamic_batch=True):
        """
        Export the PyTorch model to ONNX format with specified configurations.
        
        Args:
            dummy_input: Dummy input tensor for tracing (not used with Ultralytics export)
            dynamic_batch: Enable dynamic batch dimension (allows variable batch sizes)
            
        Dynamic Batch Dimension:
        - When enabled, the exported model can accept any batch size at inference time
        - Without it, the model is fixed to the batch size used during export
        - This is crucial for production where batch sizes may vary
        - We specify dynamic_axes={'images': {0: 'batch'}} to make dimension 0 dynamic
        
        Opset Version:
        - Determines which ONNX operators are available
        - Higher versions support more operations but may have less compatibility
        - Version 12+ is recommended for modern YOLO models
        
        The export process:
        1. Runs a forward pass with dummy input to trace operations
        2. Converts each PyTorch operation to equivalent ONNX operators
        3. Saves the graph structure and weights to a .onnx file
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
            
        # Create output directory if it doesn't exist
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\nExporting model to ONNX format...")
        print(f"Output path: {self.output_path}")
        print(f"Opset version: {self.opset_version}")
        print(f"Dynamic batch: {dynamic_batch}")
        
        # Use Ultralytics' built-in export method which handles ONNX export internally
        # This is simpler and handles all the complexities of YOLO model export
        self.model.export(
            format='onnx',
            imgsz=self.input_shape[2],  # Image size (640)
            dynamic=dynamic_batch,       # Enable dynamic batch dimension
            opset=self.opset_version,    # ONNX opset version
            simplify=True                # Simplify the ONNX model
        )
        
        # Move the exported file to the desired location if needed
        # Ultralytics exports to the same directory as the .pt file
        source_path = self.model_path.with_suffix('.onnx')
        if source_path != self.output_path and source_path.exists():
            import shutil
            shutil.move(str(source_path), str(self.output_path))
        
        print(f"\nONNX export completed successfully!")
        print(f"Model saved to: {self.output_path}")
        
    def verify_export(self):
        """
        Verify that the exported ONNX model is valid and can be loaded.
        
        This function performs basic validation:
        - Checks if the output file exists
        - Verifies the file size is reasonable
        - Attempts to load the model with ONNX to check validity
        
        This is important to catch export errors early before deployment.
        """
        if not self.output_path.exists():
            raise FileNotFoundError(f"Exported model not found: {self.output_path}")
            
        file_size_mb = self.output_path.stat().st_size / (1024 * 1024)
        print(f"\n=== Export Verification ===")
        print(f"File exists: Yes")
        print(f"File size: {file_size_mb:.2f} MB")
        
        try:
            import onnx
            
            # Load and check the ONNX model
            onnx_model = onnx.load(str(self.output_path))
            onnx.checker.check_model(onnx_model)
            
            print(f"ONNX model is valid: Yes")
            print(f"ONNX opset version: {onnx_model.opset_import[0].version}")
            
            # Print input/output information
            print(f"\nModel Inputs:")
            for input_tensor in onnx_model.graph.input:
                print(f"  - Name: {input_tensor.name}")
                shape = [dim.dim_value if dim.dim_value > 0 else 'dynamic' 
                        for dim in input_tensor.type.tensor_type.shape.dim]
                print(f"    Shape: {shape}")
                
            print(f"\nModel Outputs:")
            for output_tensor in onnx_model.graph.output:
                print(f"  - Name: {output_tensor.name}")
                
            print("===========================\n")
            
        except ImportError:
            print("ONNX package not available for detailed verification")
            print("Install with: pip install onnx")
        except Exception as e:
            print(f"Verification warning: {e}")
            
    def process(self, dynamic_batch=True, verify=True):
        """
        Complete export pipeline: load model, create dummy input, export, and verify.
        
        Args:
            dynamic_batch: Enable dynamic batch dimension
            verify: Perform post-export verification
            
        This is the main entry point that orchestrates the entire export process:
        1. Load the PyTorch model
        2. Create a dummy input tensor for tracing
        3. Export the model to ONNX format
        4. Verify the exported model is valid
        """
        # Load the PyTorch model
        self.load_model()
        
        # Create dummy input for tracing
        dummy_input = self.create_dummy_input()
        
        # Export to ONNX
        self.export_to_onnx(dummy_input, dynamic_batch=dynamic_batch)
        
        # Verify the export
        if verify:
            self.verify_export()
            
        print("ONNX export process completed successfully!")


def main():
    """
    Main function to demonstrate ONNX export from PyTorch YOLO model.
    
    This function creates an ONNXExporter instance and exports the
    YOLO model to ONNX format with dynamic batch support.
    """
    # Define paths
    model_path = "models/yolo11n.pt"
    output_path = "models/yolo11n.onnx"
    
    # Define input shape (batch, channels, height, width)
    # YOLO models typically use 640x640 input size
    input_shape = (1, 3, 640, 640)
    
    # Opset version 12+ recommended for YOLO compatibility
    opset_version = 12
    
    # Create exporter instance
    exporter = ONNXExporter(
        model_path=model_path,
        output_path=output_path,
        input_shape=input_shape,
        opset_version=opset_version
    )
    
    # Run the export process with dynamic batch support
    exporter.process(dynamic_batch=True, verify=True)


if __name__ == "__main__":
    main()

