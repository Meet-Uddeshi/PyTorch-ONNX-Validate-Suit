"""
Streamlit Frontend Application for PyTorch-ONNX Validation Suite

This module provides a web-based user interface for validating YOLO model exports
from PyTorch to ONNX format. It allows users to convert models, run inference,
and compare results through an intuitive visual interface.

Application Flow:
1. Display project overview and purpose
2. Check and load PyTorch model
3. Export PyTorch model to ONNX format
4. Run PyTorch inference on test image
5. Run ONNX inference on the same image
6. Display results and comparison visualizations
7. Explain metrics and comparisons

The interface is designed to be self-explanatory with clear explanations
at each step, making it accessible and easy to understand.
"""

import streamlit as st
import sys
import importlib.util
import tempfile
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

# Add project root to Python path for imports
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import local modules
# Note: We need to be careful with the 'onnx' directory name as it conflicts
# with the installed onnx library. We import after setting up the path.

def import_from_path(module_name, file_path):
    """
    Import a module from a specific file path.
    
    This helper function allows us to import modules from our local onnx/
    directory without conflicts with the installed onnx library.
    """
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Import from yolo directory (no conflicts)
from yolo.pytorch_inference import PyTorchYOLOInference

# Import from onnx directory using explicit paths to avoid conflicts
onnx_export_module = import_from_path(
    "export_to_onnx", 
    project_root / "onnx" / "export_to_onnx.py"
)
onnx_validator_module = import_from_path(
    "onnx_validator",
    project_root / "onnx" / "onnx_validator.py"
)
onnx_inference_module = import_from_path(
    "onnx_inference",
    project_root / "onnx" / "onnx_inference.py"
)

# Extract classes from imported modules
ONNXExporter = onnx_export_module.ONNXExporter
ONNXValidator = onnx_validator_module.ONNXValidator
ONNXYOLOInference = onnx_inference_module.ONNXYOLOInference
ONNXPyTorchComparison = onnx_inference_module.ONNXPyTorchComparison


class StreamlitApp:
    """
    Main Streamlit application class for PyTorch-ONNX validation.
    
    This class encapsulates all UI components and workflow logic
    for the validation suite, providing a clean interface for
    model conversion, inference, and comparison.
    """
    
    def __init__(self):
        """
        Initialize the Streamlit application with default paths and configuration.
        
        Sets up all file paths and initializes session state for maintaining
        application state across Streamlit reruns.
        """
        self.pytorch_model_path = Path("models/yolo11n.pt")
        self.onnx_model_path = Path("models/yolo11n.onnx")
        self.image_path = Path("data/image.jpeg")
        self.pytorch_output_path = Path("output/pytorch_result.png")
        self.onnx_output_path = Path("output/onnx_result.png")
        self.comparison_output_path = Path("output/comparison.png")
        
        # Initialize session state for tracking workflow progress
        self.initialize_session_state()
        
    def initialize_session_state(self):
        """
        Initialize Streamlit session state variables.
        
        Session state persists values across Streamlit reruns,
        allowing us to track the workflow progress and results.
        """
        if 'pytorch_model_loaded' not in st.session_state:
            st.session_state.pytorch_model_loaded = False
            
        if 'onnx_model_exported' not in st.session_state:
            st.session_state.onnx_model_exported = False
            
        if 'pytorch_inference_done' not in st.session_state:
            st.session_state.pytorch_inference_done = False
            
        if 'onnx_inference_done' not in st.session_state:
            st.session_state.onnx_inference_done = False
            
        if 'comparison_done' not in st.session_state:
            st.session_state.comparison_done = False
            
        if 'comparison_metrics' not in st.session_state:
            st.session_state.comparison_metrics = None
            
        if 'uploaded_image_path' not in st.session_state:
            st.session_state.uploaded_image_path = None
            
        if 'current_image_path' not in st.session_state:
            st.session_state.current_image_path = None
            
    def render_header(self):
        """
        Render the application header with title and description.
        
        Provides an overview of what the application does and why it's useful.
        """
        st.title("PyTorch-ONNX Validation Suite")
        
        st.markdown("""
        ### Welcome to the Model Validation Tool
        
        This application helps you validate that your AI models work correctly 
        when converted from PyTorch to ONNX format.
        
        **What does this tool do?**
        - Converts AI models from PyTorch (training format) to ONNX (deployment format)
        - Runs object detection on the same image using both formats
        - Compares results to ensure accuracy is maintained
        - Visualizes differences for easy understanding
        
        **Why is this important?**
        - ONNX models are faster and work on more devices
        - We need to verify the conversion doesn't change the results
        - Visual comparison helps catch any issues early
        
        ---
        """)
        
    def check_pytorch_model(self):
        """
        Check if PyTorch model exists and display its status.
        
        This function verifies the model file is present and shows
        model information to the user.
        """
        st.subheader("Step 1: PyTorch Model Status")
        
        if self.pytorch_model_path.exists():
            file_size_mb = self.pytorch_model_path.stat().st_size / (1024 * 1024)
            
            st.success(f"PyTorch model found: `{self.pytorch_model_path}`")
            st.info(f"Model size: {file_size_mb:.2f} MB")
            
            # Load model button
            if st.button("Load PyTorch Model", key="load_pytorch"):
                with st.spinner("Loading PyTorch model..."):
                    try:
                        inference = PyTorchYOLOInference(str(self.pytorch_model_path))
                        inference.load_model()
                        st.session_state.pytorch_model_loaded = True
                        st.success("PyTorch model loaded successfully!")
                        st.info(f"Model can detect {len(inference.class_names)} different object types")
                    except Exception as e:
                        st.error(f"Error loading model: {str(e)}")
                        
            if st.session_state.pytorch_model_loaded:
                st.success("Model is ready for inference")
                
        else:
            st.error(f"PyTorch model not found at `{self.pytorch_model_path}`")
            st.warning("Please ensure the model file exists before proceeding.")
            
        st.markdown("---")
        
    def export_to_onnx(self):
        """
        Handle ONNX export with user interface and validation.
        
        This function converts the PyTorch model to ONNX format,
        validates the export, and displays results to the user.
        """
        st.subheader("Step 2: Export to ONNX Format")
        
        st.markdown("""
        **What is ONNX?**
        
        ONNX (Open Neural Network Exchange) is a universal format for AI models.
        
        **Benefits:**
        - Runs faster than PyTorch (optimized for inference)
        - Works on more devices (phones, embedded systems, web browsers)
        - Doesn't require PyTorch installation (smaller deployment size)
        
        **Export Process:**
        - Converts model architecture to ONNX format
        - Preserves all weights (no accuracy loss expected)
        - Validates the exported model is correct
        """)
        
        # Check if already exported
        if self.onnx_model_path.exists():
            file_size_mb = self.onnx_model_path.stat().st_size / (1024 * 1024)
            st.info(f"ONNX model already exists: `{self.onnx_model_path}` ({file_size_mb:.2f} MB)")
            st.session_state.onnx_model_exported = True
        
        # Export button
        if st.button("Convert to ONNX", key="export_onnx", 
                     disabled=not st.session_state.pytorch_model_loaded):
            
            with st.spinner("Converting model to ONNX format... This may take a minute."):
                try:
                    # Export
                    progress_text = st.empty()
                    progress_text.text("Step 1/2: Exporting to ONNX...")
                    
                    exporter = ONNXExporter(
                        model_path=str(self.pytorch_model_path),
                        output_path=str(self.onnx_model_path),
                        input_shape=(1, 3, 640, 640),
                        opset_version=12
                    )
                    exporter.process(dynamic_batch=True, verify=False)
                    
                    # Validate
                    progress_text.text("Step 2/2: Validating exported model...")
                    
                    validator = ONNXValidator(str(self.onnx_model_path))
                    is_valid = validator.validate(verbose=False)
                    
                    progress_text.empty()
                    
                    if is_valid:
                        st.success("Model successfully converted and validated!")
                        st.session_state.onnx_model_exported = True
                        
                        file_size_mb = self.onnx_model_path.stat().st_size / (1024 * 1024)
                        st.info(f"ONNX model saved: {file_size_mb:.2f} MB")
                    else:
                        st.error("ONNX model validation failed. Please check the model.")
                        
                except Exception as e:
                    st.error(f"Error during export: {str(e)}")
                    
        if not st.session_state.pytorch_model_loaded:
            st.warning("Please load the PyTorch model first (Step 1)")
            
        st.markdown("---")
        
    def run_pytorch_inference(self):
        """
        Run PyTorch inference and display results.
        
        This function processes the test image using PyTorch,
        detects objects, and shows the annotated result.
        """
        st.subheader("Step 3: Run PyTorch Inference")
        
        st.markdown("""
        **What is inference?**
        
        Inference means using the trained model to make predictions on new data.
        In this case, we're detecting objects in an image.
        
        **Process:**
        1. Load and preprocess the image
        2. Feed it through the neural network
        3. Extract detected objects (bounding boxes, labels, confidence scores)
        4. Draw boxes around detected objects
        """)
        
        # Image upload section
        st.markdown("**Choose Image Source:**")
        uploaded_file = st.file_uploader(
            "Upload an image from your device (JPG, JPEG, PNG)",
            type=["jpg", "jpeg", "png"],
            key="image_uploader"
        )
        
        # Determine which image to use
        current_image_path = self.image_path
        if uploaded_file is not None:
            # Save uploaded image to temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                current_image_path = Path(tmp_file.name)
                st.session_state.uploaded_image_path = current_image_path
            st.success("Uploaded image will be used for inference")
        elif 'uploaded_image_path' in st.session_state and st.session_state.uploaded_image_path:
            current_image_path = st.session_state.uploaded_image_path
        
        # Show original image
        if current_image_path.exists():
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Original Image:**")
                original_image = Image.open(current_image_path)
                st.image(original_image, use_container_width=True)
        else:
            st.error(f"Image not found: `{current_image_path}`")
            return
            
        # Run inference button
        if st.button("Run PyTorch Inference", key="run_pytorch",
                     disabled=not st.session_state.pytorch_model_loaded):
            
            with st.spinner("Running PyTorch inference..."):
                try:
                    inference = PyTorchYOLOInference(str(self.pytorch_model_path))
                    inference.load_model()
                    
                    boxes, class_names, confidences = inference.process(
                        str(current_image_path),
                        str(self.pytorch_output_path)
                    )
                    
                    st.session_state.pytorch_inference_done = True
                    st.session_state.pytorch_results = (boxes, class_names, confidences)
                    st.session_state.current_image_path = str(current_image_path)
                    
                    st.success(f"PyTorch detected {len(boxes)} objects!")
                    
                except Exception as e:
                    st.error(f"Error during inference: {str(e)}")
                    
        # Display results
        if st.session_state.pytorch_inference_done and self.pytorch_output_path.exists():
            with col2:
                st.write("**PyTorch Detection Result:**")
                result_image = Image.open(self.pytorch_output_path)
                st.image(result_image, use_container_width=True)
                
            # Show detection details
            if 'pytorch_results' in st.session_state:
                boxes, class_names, confidences = st.session_state.pytorch_results
                
                st.write("**Detected Objects:**")
                for i, (cls, conf) in enumerate(zip(class_names, confidences), 1):
                    st.write(f"{i}. **{cls}** - Confidence: {conf:.1%}")
                    
        if not st.session_state.pytorch_model_loaded:
            st.warning("Please load the PyTorch model first (Step 1)")
            
        st.markdown("---")
        
    def run_onnx_inference(self):
        """
        Run ONNX inference and display results.
        
        This function processes the same test image using ONNX Runtime,
        detects objects, and shows the annotated result for comparison.
        """
        st.subheader("Step 4: Run ONNX Inference")
        
        st.markdown("""
        **Running the ONNX Model:**
        
        Now we run the same detection using the ONNX version of the model.
        This should produce nearly identical results to PyTorch.
        
        **What to expect:**
        - Same objects detected (in most cases)
        - Very similar confidence scores (within 1-2%)
        - Bounding boxes in almost the same positions
        - Small differences are normal due to numerical precision
        """)
        
        # Run inference button
        if st.button("Run ONNX Inference", key="run_onnx",
                     disabled=not st.session_state.onnx_model_exported):
            
            with st.spinner("Running ONNX inference..."):
                try:
                    # Use the same image that was used for PyTorch inference
                    image_to_use = st.session_state.get('current_image_path', str(self.image_path))
                    
                    inference = ONNXYOLOInference(
                        str(self.onnx_model_path),
                        conf_threshold=0.25,
                        iou_threshold=0.45
                    )
                    inference.load_model()
                    
                    boxes, class_names, confidences = inference.process(
                        image_to_use,
                        str(self.onnx_output_path)
                    )
                    
                    st.session_state.onnx_inference_done = True
                    st.session_state.onnx_results = (boxes, class_names, confidences)
                    
                    st.success(f"ONNX detected {len(boxes)} objects!")
                    
                except Exception as e:
                    st.error(f"Error during inference: {str(e)}")
                    
        # Display results
        if st.session_state.onnx_inference_done and self.onnx_output_path.exists():
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**PyTorch Result:**")
                if self.pytorch_output_path.exists():
                    pytorch_img = Image.open(self.pytorch_output_path)
                    st.image(pytorch_img, use_container_width=True)
                    
            with col2:
                st.write("**ONNX Result:**")
                onnx_img = Image.open(self.onnx_output_path)
                st.image(onnx_img, use_container_width=True)
                
            # Show detection details
            if 'onnx_results' in st.session_state:
                boxes, class_names, confidences = st.session_state.onnx_results
                
                st.write("**ONNX Detected Objects:**")
                for i, (cls, conf) in enumerate(zip(class_names, confidences), 1):
                    st.write(f"{i}. **{cls}** - Confidence: {conf:.1%}")
                    
        if not st.session_state.onnx_model_exported:
            st.warning("Please export the ONNX model first (Step 2)")
            
        st.markdown("---")
        
    def display_comparison(self):
        """
        Display detailed comparison between PyTorch and ONNX results.
        
        This function generates comparison metrics and visualizations
        with clear explanations of what they mean.
        """
        st.subheader("Step 5: Comparison Analysis")
        
        st.markdown("""
        **Understanding the Comparison:**
        
        We compare the two models to verify they produce similar results.
        This validates that the ONNX conversion was successful.
        """)
        
        # Check if both inferences are done
        if not (st.session_state.pytorch_inference_done and st.session_state.onnx_inference_done):
            st.warning("Please run both PyTorch and ONNX inference first (Steps 3 & 4)")
            return
            
        # Generate comparison button
        if st.button("Generate Comparison Report", key="compare"):
            with st.spinner("Analyzing differences..."):
                try:
                    # Get results
                    pt_boxes, pt_classes, pt_confidences = st.session_state.pytorch_results
                    onnx_boxes, onnx_classes, onnx_confidences = st.session_state.onnx_results
                    
                    # Create comparison
                    comparison = ONNXPyTorchComparison()
                    comparison.set_pytorch_results(pt_boxes, pt_classes, pt_confidences)
                    comparison.set_onnx_results(onnx_boxes, onnx_classes, onnx_confidences)
                    
                    # Generate graph
                    comparison.generate_comparison_graph(str(self.comparison_output_path))
                    
                    # Get metrics
                    metrics = comparison.compute_metrics()
                    st.session_state.comparison_metrics = metrics
                    st.session_state.comparison_done = True
                    
                    st.success("Comparison analysis complete!")
                    
                except Exception as e:
                    st.error(f"Error during comparison: {str(e)}")
                    
        # Display comparison results
        if st.session_state.comparison_done and st.session_state.comparison_metrics:
            self.render_comparison_results()
            
        st.markdown("---")
        
    def render_comparison_results(self):
        """
        Render detailed comparison results with explanations.
        
        This function displays metrics, visualizations, and interpretations
        in a clear and understandable way.
        """
        metrics = st.session_state.comparison_metrics
        
        # Overall verdict
        st.markdown("### Overall Assessment")
        
        if metrics['matched_boxes'] == 0:
            st.error("WARNING: No matching boxes found!")
            st.markdown("""
            **What this means:**
            - The two models detected completely different objects
            - This indicates a problem with the conversion
            - Check preprocessing and postprocessing settings
            """)
        elif metrics['avg_iou'] > 0.9 and metrics['avg_confidence_diff'] < 0.01:
            st.success("EXCELLENT: Models produce nearly identical results!")
            st.markdown("""
            **What this means:**
            - The ONNX conversion was successful
            - Outputs are within expected precision tolerances
            - Safe to deploy the ONNX model in production
            """)
        elif metrics['avg_iou'] > 0.75 and metrics['avg_confidence_diff'] < 0.05:
            st.success("GOOD: Models produce very similar results!")
            st.markdown("""
            **What this means:**
            - Small differences exist but are within acceptable range
            - The ONNX model is working correctly
            - Differences are due to numerical precision variations
            """)
        else:
            st.warning("WARNING: Models show significant differences!")
            st.markdown("""
            **What this means:**
            - Outputs differ more than expected
            - May indicate issues in preprocessing or postprocessing
            - Review conversion settings and validation
            """)
            
        st.markdown("---")
        
        # Key metrics
        st.markdown("### Key Metrics Explained")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="PyTorch Detections",
                value=metrics['pytorch_detections'],
                help="Number of objects detected by PyTorch model"
            )
            
        with col2:
            st.metric(
                label="ONNX Detections",
                value=metrics['onnx_detections'],
                help="Number of objects detected by ONNX model"
            )
            
        with col3:
            diff = metrics['onnx_detections'] - metrics['pytorch_detections']
            st.metric(
                label="Difference",
                value=diff,
                delta=f"{diff:+d}",
                help="Difference in detection counts (ideally 0)"
            )
            
        st.markdown("---")
        
        if metrics['matched_boxes'] > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    label="Matched Boxes",
                    value=metrics['matched_boxes'],
                    help="Number of detections that match between models"
                )
                
                st.metric(
                    label="Average IoU",
                    value=f"{metrics['avg_iou']:.3f}",
                    help="How well the bounding boxes align (1.0 = perfect)"
                )
                
            with col2:
                st.metric(
                    label="Avg Confidence Difference",
                    value=f"{metrics['avg_confidence_diff']:.4f}",
                    help="Average difference in confidence scores (lower is better)"
                )
                
                st.metric(
                    label="Class Agreement",
                    value=f"{metrics['class_agreement']*100:.1f}%",
                    help="Percentage of detections with matching class labels"
                )
                
        st.markdown("---")
        
        # Visualization explanation
        st.markdown("### Comparison Visualization")
        
        st.markdown("""
        **Understanding the Graphs:**
        
        The comparison visualization contains 4 panels that help you understand
        how similar the two models are:
        """)
        
        if self.comparison_output_path.exists():
            comparison_img = Image.open(self.comparison_output_path)
            st.image(comparison_img, use_container_width=True)
            
        st.markdown("""
        #### Panel 1: Detection Count Comparison (Top Left)
        **What it shows:** Bar chart comparing number of detections
        - Shows if both models found the same number of objects
        - Validates detection consistency across frameworks
        - **Good result:** Bars are the same height (equal detections)
        
        #### Panel 2: Confidence Score Comparison (Top Right)
        **What it shows:** Scatter plot comparing confidence scores
        - Each dot represents one detected object
        - Dots near the red line mean both models are equally confident
        - Correlation value near 1.0 indicates numerical equivalence
        - **Good result:** Points cluster tightly around the diagonal line
        
        #### Panel 3: IoU Distribution (Bottom Left)
        **What it shows:** Histogram of box overlap quality
        - Shows how precisely the boxes align
        - Bars on the right (near 1.0) indicate better alignment
        - IoU (Intersection over Union) measures spatial agreement between boxes
        - **Good result:** Most bars clustered above 0.8-0.9
        
        #### Panel 4: Summary Statistics (Bottom Right)
        **What it shows:** Text summary with overall verdict
        - Quick assessment of whether the conversion was successful
        - Shows detection counts, matching results, and average differences
        - **Good result:** "EXCELLENT" or "GOOD" verdict
        
        ---
        
        **What are acceptable differences?**
        - **Box positions:** Within 1-2 pixels (invisible to human eye)
        - **Confidence scores:** Within 0.01 (1% difference)
        - **Detection count:** Same or differ by 1-2 objects maximum
        - **Class labels:** Should always match
        
        **Why do small differences occur?**
        1. **Floating-point math:** Different libraries round numbers slightly differently
        2. **Operation order:** PyTorch and ONNX may compute in different orders
        3. **Hardware variations:** CPU vs GPU implementations vary slightly
        4. **Non-deterministic operations:** Some GPU operations prioritize speed over exact reproducibility
        
        These are normal and expected - they don't indicate a problem!
        """)
        
    def render_footer(self):
        """
        Render application footer with additional information.
        
        Provides additional technical details and context.
        """
        st.markdown("---")
        st.markdown("""
        ### Additional Information
        
        **Files Generated:**
        - `output/pytorch_result.png` - PyTorch detection visualization
        - `output/onnx_result.png` - ONNX detection visualization  
        - `output/comparison.png` - Detailed comparison graphs
        
        **Technical Details:**
        - Model: YOLO11n (nano) - 80 object classes
        - Input size: 640x640 pixels
        - Framework: PyTorch → ONNX Runtime
        - Confidence threshold: 25%
        - NMS IoU threshold: 45%
        
        ---
        
        *PyTorch-ONNX Validation Suite - Ensuring Model Consistency Across Frameworks*
        """)
        
    def run(self):
        """
        Main application entry point.
        
        This function orchestrates the entire UI flow, calling each section
        in sequence to create the complete user experience.
        """
        # Set page configuration
        st.set_page_config(
            page_title="PyTorch-ONNX Validation Suite",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="collapsed"
        )
        
        # Render sections
        self.render_header()
        self.check_pytorch_model()
        self.export_to_onnx()
        self.run_pytorch_inference()
        self.run_onnx_inference()
        self.display_comparison()
        self.render_footer()


def main():
    """
    Main function to launch the Streamlit application.
    
    Creates an instance of StreamlitApp and runs the interface.
    """
    app = StreamlitApp()
    app.run()


if __name__ == "__main__":
    main()

