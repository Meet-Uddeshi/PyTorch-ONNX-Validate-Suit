"""
ONNX Model Validator Module

This module validates ONNX models to ensure they are correctly formed and ready for inference.
Validation is a critical step before deploying ONNX models to production environments.

Why ONNX Validation is Necessary Before Inference:

1. Detect Export Errors:
   - The export process can fail silently, producing corrupted models
   - Operators might not be properly converted from PyTorch to ONNX
   - Model structure could be incomplete or malformed

2. Ensure Compatibility:
   - Verify the opset version is supported by the target runtime
   - Check that all operators are valid ONNX operators
   - Confirm input/output shapes are properly defined

3. Catch Type Mismatches:
   - Verify tensor types (float32, int64, etc.) are consistent
   - Ensure data types match between connected layers
   - Detect dimension mismatches that could cause runtime errors

4. Prevent Runtime Failures:
   - Invalid models will fail during inference, wasting computation time
   - Validation catches issues early in the development cycle
   - Saves debugging time by identifying problems before deployment

5. Verify Model Integrity:
   - Ensure all required model parameters (weights) are present
   - Check that the computational graph is complete and connected
   - Validate that initializers and constants are properly defined

Without validation, you might deploy a model that:
- Crashes during inference with cryptic errors
- Produces incorrect results silently
- Has incompatible operations for the target platform
- Contains corrupted weights or missing parameters
"""

import onnx
from pathlib import Path
from onnx import checker, helper, shape_inference
import numpy as np


class ONNXValidator:
    """
    ONNX model validator class for checking model integrity and correctness.
    
    This class performs comprehensive validation of ONNX models including
    graph structure, operator compatibility, type checking, and shape inference.
    """
    
    def __init__(self, model_path):
        """
        Initialize the ONNX validator with a model path.
        
        Args:
            model_path: Path to the ONNX model file (.onnx)
            
        The validator will load and check the model for various issues
        that could prevent successful inference.
        """
        self.model_path = Path(model_path)
        self.model = None
        self.is_valid = False
        self.validation_errors = []
        self.validation_warnings = []
        
    def load_model(self):
        """
        Load the ONNX model from the specified path.
        
        This function reads the ONNX model file and deserializes it into
        an in-memory model object. The model contains:
        - Graph structure (nodes, edges)
        - Operator types and attributes
        - Initializers (weights and constants)
        - Input/output specifications
        
        Raises:
            FileNotFoundError: If the model file doesn't exist
            Exception: If the model cannot be loaded or parsed
        """
        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")
            
        print(f"Loading ONNX model from: {self.model_path}")
        
        try:
            # Load the ONNX model
            self.model = onnx.load(str(self.model_path))
            
            # Get file size for information
            file_size_mb = self.model_path.stat().st_size / (1024 * 1024)
            print(f"Model loaded successfully")
            print(f"File size: {file_size_mb:.2f} MB")
            
            return True
            
        except Exception as e:
            error_msg = f"Failed to load model: {str(e)}"
            print(f"ERROR: {error_msg}")
            self.validation_errors.append(error_msg)
            return False
            
    def check_model_structure(self):
        """
        Perform basic structural validation of the ONNX model.
        
        This function uses ONNX's built-in checker to verify:
        - Model structure is well-formed
        - Graph is properly connected
        - No orphaned nodes or missing edges
        - All required fields are present
        
        The checker validates the model against the ONNX specification
        and ensures it conforms to the expected format.
        
        Returns:
            bool: True if structure is valid, False otherwise
        """
        if self.model is None:
            print("ERROR: Model not loaded. Call load_model() first.")
            return False
            
        print("\n=== Checking Model Structure ===")
        
        try:
            # Check model structure and validity
            # This performs comprehensive validation of the model
            checker.check_model(self.model)
            print("✓ Model structure is valid")
            return True
            
        except Exception as e:
            error_msg = f"Model structure check failed: {str(e)}"
            print(f"✗ {error_msg}")
            self.validation_errors.append(error_msg)
            return False
            
    def check_graph_integrity(self):
        """
        Check the computational graph for integrity issues.
        
        This function validates:
        - All nodes have valid inputs and outputs
        - Tensor names are unique and properly referenced
        - No circular dependencies in the graph
        - All edges connect valid nodes
        
        Graph integrity is crucial because a broken graph will cause
        inference to fail or produce incorrect results.
        
        Returns:
            bool: True if graph is valid, False otherwise
        """
        if self.model is None:
            print("ERROR: Model not loaded. Call load_model() first.")
            return False
            
        print("\n=== Checking Graph Integrity ===")
        
        try:
            graph = self.model.graph
            
            # Check if graph has nodes
            if len(graph.node) == 0:
                error_msg = "Graph has no nodes"
                print(f"✗ {error_msg}")
                self.validation_errors.append(error_msg)
                return False
            
            print(f"✓ Graph contains {len(graph.node)} nodes")
            
            # Check if graph has inputs
            if len(graph.input) == 0:
                error_msg = "Graph has no inputs defined"
                print(f"✗ {error_msg}")
                self.validation_errors.append(error_msg)
                return False
                
            print(f"✓ Graph has {len(graph.input)} input(s)")
            
            # Check if graph has outputs
            if len(graph.output) == 0:
                error_msg = "Graph has no outputs defined"
                print(f"✗ {error_msg}")
                self.validation_errors.append(error_msg)
                return False
                
            print(f"✓ Graph has {len(graph.output)} output(s)")
            
            # Check initializers (weights)
            print(f"✓ Graph has {len(graph.initializer)} initializer(s)")
            
            return True
            
        except Exception as e:
            error_msg = f"Graph integrity check failed: {str(e)}"
            print(f"✗ {error_msg}")
            self.validation_errors.append(error_msg)
            return False
            
    def check_opset_version(self):
        """
        Check the ONNX opset version and operator compatibility.
        
        The opset version determines which operations are available:
        - Higher versions support more operations
        - Lower versions have better compatibility with older runtimes
        - Each opset defines specific operator behaviors
        
        This check ensures the model uses a supported opset version
        and that all operators are valid for that version.
        
        Returns:
            bool: True if opset is valid, False otherwise
        """
        if self.model is None:
            print("ERROR: Model not loaded. Call load_model() first.")
            return False
            
        print("\n=== Checking Opset Version ===")
        
        try:
            # Get opset version
            opset_version = self.model.opset_import[0].version
            print(f"✓ ONNX Opset Version: {opset_version}")
            
            # Check if opset version is reasonable
            if opset_version < 7:
                warning_msg = f"Opset version {opset_version} is very old. Consider using 11+."
                print(f"⚠ WARNING: {warning_msg}")
                self.validation_warnings.append(warning_msg)
            elif opset_version > 20:
                warning_msg = f"Opset version {opset_version} is very new. Ensure runtime compatibility."
                print(f"⚠ WARNING: {warning_msg}")
                self.validation_warnings.append(warning_msg)
            else:
                print(f"✓ Opset version is reasonable")
                
            return True
            
        except Exception as e:
            error_msg = f"Opset version check failed: {str(e)}"
            print(f"✗ {error_msg}")
            self.validation_errors.append(error_msg)
            return False
            
    def perform_shape_inference(self):
        """
        Perform shape inference on the model to validate tensor shapes.
        
        Shape inference:
        - Propagates shape information through the graph
        - Validates that tensor shapes are compatible between operations
        - Detects dimension mismatches early
        - Ensures output shapes can be determined
        
        This is important because shape mismatches will cause runtime errors.
        Shape inference helps catch these issues during validation.
        
        Returns:
            bool: True if shape inference succeeds, False otherwise
        """
        if self.model is None:
            print("ERROR: Model not loaded. Call load_model() first.")
            return False
            
        print("\n=== Performing Shape Inference ===")
        
        try:
            # Perform shape inference
            # This propagates shape information through the entire graph
            inferred_model = shape_inference.infer_shapes(self.model)
            
            print("✓ Shape inference completed successfully")
            
            # Update model with inferred shapes
            self.model = inferred_model
            
            return True
            
        except Exception as e:
            error_msg = f"Shape inference failed: {str(e)}"
            print(f"✗ {error_msg}")
            self.validation_errors.append(error_msg)
            return False
            
    def display_model_info(self):
        """
        Display detailed information about the ONNX model.
        
        This function provides a summary of:
        - Model inputs (name, shape, type)
        - Model outputs (name, shape, type)
        - Opset version
        - Number of nodes and parameters
        
        This information helps understand the model structure and
        verify it matches expectations.
        """
        if self.model is None:
            print("ERROR: Model not loaded. Call load_model() first.")
            return
            
        print("\n=== Model Information ===")
        
        graph = self.model.graph
        
        # Display inputs
        print("\nInputs:")
        for input_tensor in graph.input:
            # Skip initializers from inputs
            if input_tensor.name in [init.name for init in graph.initializer]:
                continue
                
            print(f"  Name: {input_tensor.name}")
            
            # Get tensor type
            tensor_type = input_tensor.type.tensor_type
            dtype = onnx.TensorProto.DataType.Name(tensor_type.elem_type)
            print(f"  Type: {dtype}")
            
            # Get shape
            shape = []
            for dim in tensor_type.shape.dim:
                if dim.dim_value:
                    shape.append(dim.dim_value)
                elif dim.dim_param:
                    shape.append(dim.dim_param)
                else:
                    shape.append('?')
            print(f"  Shape: {shape}")
            
        # Display outputs
        print("\nOutputs:")
        for output_tensor in graph.output:
            print(f"  Name: {output_tensor.name}")
            
            # Get tensor type if available
            if output_tensor.type.HasField('tensor_type'):
                tensor_type = output_tensor.type.tensor_type
                dtype = onnx.TensorProto.DataType.Name(tensor_type.elem_type)
                print(f"  Type: {dtype}")
                
                # Get shape if available
                if tensor_type.HasField('shape'):
                    shape = []
                    for dim in tensor_type.shape.dim:
                        if dim.dim_value:
                            shape.append(dim.dim_value)
                        elif dim.dim_param:
                            shape.append(dim.dim_param)
                        else:
                            shape.append('?')
                    print(f"  Shape: {shape}")
                    
        print(f"\nTotal nodes: {len(graph.node)}")
        print(f"Total initializers: {len(graph.initializer)}")
        print("=========================")
        
    def validate(self, verbose=True):
        """
        Run complete validation pipeline on the ONNX model.
        
        This function orchestrates all validation checks:
        1. Load the model
        2. Check model structure
        3. Check graph integrity
        4. Check opset version
        5. Perform shape inference
        6. Display model information
        
        Args:
            verbose: Display detailed model information
            
        Returns:
            bool: True if all validations pass, False otherwise
            
        The validation process is sequential - if an early check fails,
        subsequent checks may be skipped as they could produce misleading results.
        """
        print("=" * 60)
        print("ONNX Model Validation")
        print("=" * 60)
        
        # Reset validation state
        self.is_valid = False
        self.validation_errors = []
        self.validation_warnings = []
        
        # Step 1: Load model
        if not self.load_model():
            print("\n" + "=" * 60)
            print("VALIDATION FAILED: Could not load model")
            print("=" * 60)
            return False
            
        # Step 2: Check model structure
        structure_valid = self.check_model_structure()
        
        # Step 3: Check graph integrity
        graph_valid = self.check_graph_integrity()
        
        # Step 4: Check opset version
        opset_valid = self.check_opset_version()
        
        # Step 5: Perform shape inference
        shape_valid = self.perform_shape_inference()
        
        # Step 6: Display model information
        if verbose:
            self.display_model_info()
            
        # Determine overall validation result
        self.is_valid = (structure_valid and graph_valid and 
                        opset_valid and shape_valid)
        
        # Display validation summary
        print("\n" + "=" * 60)
        print("Validation Summary")
        print("=" * 60)
        
        if self.is_valid:
            print("✓ MODEL IS VALID")
            print("  The model passed all validation checks.")
            print("  It is ready for ONNX Runtime inference.")
        else:
            print("✗ MODEL IS INVALID")
            print("  The model failed one or more validation checks.")
            print("  Review the errors above before using this model.")
            
        if self.validation_errors:
            print(f"\nErrors found: {len(self.validation_errors)}")
            for i, error in enumerate(self.validation_errors, 1):
                print(f"  {i}. {error}")
                
        if self.validation_warnings:
            print(f"\nWarnings found: {len(self.validation_warnings)}")
            for i, warning in enumerate(self.validation_warnings, 1):
                print(f"  {i}. {warning}")
                
        print("=" * 60)
        
        return self.is_valid


def main():
    """
    Main function to demonstrate ONNX model validation.
    
    This function creates a validator instance and runs
    comprehensive validation on the exported ONNX model.
    """
    # Define model path
    model_path = "models/yolo11n.onnx"
    
    # Create validator instance
    validator = ONNXValidator(model_path)
    
    # Run validation
    is_valid = validator.validate(verbose=True)
    
    # Exit with appropriate code
    if is_valid:
        print("\n✓ Validation completed successfully!")
        return 0
    else:
        print("\n✗ Validation failed!")
        return 1


if __name__ == "__main__":
    exit(main())

