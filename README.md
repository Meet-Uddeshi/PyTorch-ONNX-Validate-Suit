# PyTorch-ONNX Validation Suite

A comprehensive validation tool for verifying YOLO model conversions from PyTorch to ONNX format. This suite helps ensure your AI models work correctly after conversion with automated testing, comparison, and visualization.

## What Does This Tool Do?

This tool helps you:
- ✅ Convert PyTorch YOLO models to ONNX format
- ✅ Run inference on both PyTorch and ONNX models
- ✅ Compare results to verify conversion accuracy
- ✅ Visualize differences with detailed metrics
- ✅ Ensure your model works correctly before deployment

## Why Use This Tool?

When deploying AI models to production, you often need to convert from PyTorch (training format) to ONNX (deployment format). This conversion needs to be validated to ensure:
- Same objects are detected
- Confidence scores are similar
- Bounding boxes are in the same positions
- No accuracy is lost during conversion

---

## Installation & Setup

### Prerequisites

- Python 3.8 or higher
- 4GB+ RAM (8GB recommended)
- 2GB disk space
- NVIDIA GPU (optional, for faster inference)

### Quick Start

```bash
# 1. Clone the repository
git clone <repository-url>
cd PyTorch-ONNX-Validation-Suite

# 2. Create virtual environment (recommended)
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on Linux/macOS
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify installation
python -c "import torch; import onnxruntime; print('Installation successful!')"
```

### GPU Setup (Optional - 10x Faster)

If you have an NVIDIA GPU (like RTX 4050):

```bash
# 1. Check if GPU is detected
python check_gpu.py

# 2. Install GPU packages (if needed)
pip uninstall torch torchvision torchaudio onnxruntime -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install onnxruntime-gpu

# 3. Verify GPU setup
python check_gpu.py
```

**GPU Requirements:**
- NVIDIA GPU drivers (download from [nvidia.com](https://www.nvidia.com/Download/index.aspx))
- CUDA Toolkit 11.8 or 12.x (download from [NVIDIA CUDA](https://developer.nvidia.com/cuda-downloads))
- cuDNN library (optional, for ONNX GPU)

**Note:** The project automatically falls back to CPU if GPU is not available.

---

## Project Structure

```
PyTorch-ONNX-Validation-Suite/
│
├── frontend/
│   └── app.py                      # Streamlit web interface
│
├── yolo/
│   └── pytorch_inference.py        # PyTorch YOLO inference
│
├── onnx/
│   ├── export_to_onnx.py          # Convert PyTorch to ONNX
│   ├── onnx_inference.py          # ONNX inference & comparison
│   └── onnx_validator.py          # ONNX model validation
│
├── models/
│   ├── yolo11n.pt                 # PyTorch model (auto-downloaded)
│   └── yolo11n.onnx               # ONNX model (exported)
│
├── data/
│   └── image.jpeg                 # Test images
│
├── output/
│   ├── pytorch_result.png         # PyTorch detection results
│   ├── onnx_result.png            # ONNX detection results
│   └── comparison.png             # Comparison graphs
│
├── check_gpu.py                   # GPU verification utility
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── .gitignore                     # Git ignore rules
```

### File Descriptions

| File/Folder | Purpose |
|-------------|---------|
| `frontend/app.py` | Web-based UI for the validation workflow |
| `yolo/pytorch_inference.py` | Run object detection using PyTorch |
| `onnx/export_to_onnx.py` | Convert PyTorch models to ONNX |
| `onnx/onnx_inference.py` | Run object detection using ONNX |
| `onnx/onnx_validator.py` | Validate ONNX model structure |
| `check_gpu.py` | Check GPU availability and configuration |
| `models/` | Stores model files (PyTorch and ONNX) |
| `data/` | Input images for testing |
| `output/` | Generated results and visualizations |

---

## How to Use

### Option 1: Web Interface (Recommended)

Launch the interactive web interface:

```bash
streamlit run frontend/app.py
```

The web app will open in your browser with 5 easy steps:

1. **Load PyTorch Model** - Load the YOLO model
2. **Export to ONNX** - Convert to ONNX format
3. **Run PyTorch Inference** - Detect objects using PyTorch
4. **Run ONNX Inference** - Detect objects using ONNX
5. **Compare Results** - View detailed comparison

### Option 2: Command Line

Run individual components:

```bash
# Run PyTorch inference
python yolo/pytorch_inference.py

# Export PyTorch to ONNX
python onnx/export_to_onnx.py

# Run ONNX inference
python onnx/onnx_inference.py

# Run comparison between PyTorch and ONNX
python onnx/onnx_inference.py --compare

# Validate ONNX model
python onnx/onnx_validator.py
```

---

## Detailed Workflow

### Step-by-Step Process

#### 1. Model Loading
- Load YOLO11n PyTorch model (auto-downloads if not present)
- Model size: ~6MB
- 80 object classes (COCO dataset)

#### 2. ONNX Export
- Convert PyTorch model to ONNX format
- Preserves all weights and architecture
- Validates exported model structure
- Output: `models/yolo11n.onnx`

#### 3. PyTorch Inference
- Load input image
- Preprocess: Resize to 640x640, normalize
- Run detection
- Draw bounding boxes
- Save result: `output/pytorch_result.png`

#### 4. ONNX Inference
- Use same input image
- Apply identical preprocessing
- Run detection using ONNX Runtime
- Draw bounding boxes
- Save result: `output/onnx_result.png`

#### 5. Comparison
- Match detected objects between models
- Calculate metrics:
  - **IoU (Intersection over Union)**: Box overlap accuracy
  - **Confidence Difference**: Score comparison
  - **Detection Count**: Number of objects found
  - **Class Agreement**: Label matching rate
- Generate comparison graphs: `output/comparison.png`

---

## Understanding Results

### Comparison Metrics

| Metric | Excellent | Good | Needs Review |
|--------|-----------|------|--------------|
| **IoU** | > 0.95 | > 0.80 | < 0.80 |
| **Confidence Diff** | < 0.5% | < 2% | > 2% |
| **Detection Count Diff** | 0 | ≤ 2 | > 2 |
| **Class Agreement** | 100% | ≥ 95% | < 95% |

### What the Metrics Mean

**IoU (Intersection over Union)**
- Measures how well bounding boxes align
- 1.0 = Perfect match
- 0.0 = No overlap

**Confidence Difference**
- Difference in detection confidence scores
- Lower is better
- Small differences (< 1%) are normal

**Detection Count**
- Number of objects detected by each model
- Should be the same or very close

**Class Agreement**
- Percentage of detections with matching labels
- Should be 100% (both models identify same objects)

### Example Output

```
Detection Counts:
  PyTorch: 5 objects
  ONNX: 5 objects

Matched Boxes: 5
Average IoU: 0.964
Confidence Difference: 0.003 (0.3%)
Class Agreement: 100%

Assessment: ✓ EXCELLENT
Models produce nearly identical results!
```

---

## Features

### Automatic GPU/CPU Detection
- Uses GPU automatically if available
- Falls back to CPU if GPU not found
- No code changes needed

### Visual Comparison
- Side-by-side result images
- Detailed comparison graphs
- Easy-to-understand metrics

### Web Interface
- No coding required
- Step-by-step workflow
- Upload custom images
- Interactive visualizations

### Command Line Tools
- Scriptable for automation
- CI/CD integration ready
- Batch processing support

---

## Performance

### Inference Speed

| Device | PyTorch | ONNX | Speedup |
|--------|---------|------|---------|
| **RTX 4050 GPU** | 15-30ms | 10-20ms | 1.5-2x |
| **CPU (modern)** | 150-300ms | 100-200ms | 1.5x |

**ONNX Benefits:**
- Faster inference (1.5-3x)
- Lower memory usage
- Smaller deployment size
- Cross-platform compatibility

---

## Troubleshooting

### Common Issues

**Issue: Model not found**
```bash
# Solution: Model auto-downloads on first run
python yolo/pytorch_inference.py
```

**Issue: GPU not detected**
```bash
# Solution: Check GPU setup
python check_gpu.py

# Install GPU packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install onnxruntime-gpu
```

**Issue: Different detection results**
```
Cause: Preprocessing mismatch
Solution: Ensure both models use same image preprocessing
```

**Issue: Out of memory**
```
Cause: Insufficient GPU memory
Solution: Use CPU mode or close other applications
```

### Debug Mode

Enable verbose output:
```bash
# Add verbose flag to any script
python yolo/pytorch_inference.py --verbose
```

---

## Requirements

### Python Packages

```
torch>=2.0.0              # PyTorch framework
torchvision>=0.15.0       # PyTorch vision utilities
torchaudio>=2.0.0         # PyTorch audio (for CUDA support)
ultralytics>=8.0.0        # YOLO implementation
onnx>=1.14.0              # ONNX model format
onnxscript>=0.1.0         # ONNX export utilities
onnxruntime-gpu>=1.16.0   # ONNX Runtime (GPU)
opencv-python>=4.8.0      # Image processing
Pillow>=10.0.0            # Image handling
numpy>=1.24.0             # Numerical computing
matplotlib>=3.7.0         # Visualization
streamlit>=1.28.0         # Web interface
pyyaml>=6.0               # Configuration files
```

### System Requirements

**Minimum:**
- Python 3.8+
- 4GB RAM
- 2GB disk space

**Recommended:**
- Python 3.10+
- 8GB+ RAM
- NVIDIA GPU with 4GB+ VRAM
- SSD storage

---

## Advanced Usage

### Custom Images

```python
# Using Python
from yolo.pytorch_inference import PyTorchYOLOInference

inference = PyTorchYOLOInference("models/yolo11n.pt")
inference.load_model()
boxes, classes, scores = inference.process(
    "path/to/your/image.jpg",
    "output/result.png"
)
```

### Batch Processing

```python
# Process multiple images
import os
from pathlib import Path

image_dir = Path("data/images/")
for image_path in image_dir.glob("*.jpg"):
    boxes, classes, scores = inference.process(
        str(image_path),
        f"output/{image_path.stem}_result.png"
    )
```

### Custom Thresholds

```python
from onnx.onnx_inference import ONNXYOLOInference

# Adjust confidence and NMS thresholds
inference = ONNXYOLOInference(
    "models/yolo11n.onnx",
    conf_threshold=0.5,  # Higher = fewer detections
    iou_threshold=0.3    # Lower = less suppression
)
```

---

## Quick Reference

### Essential Commands

```bash
# Check GPU
python check_gpu.py

# Launch web interface
streamlit run frontend/app.py

# Run PyTorch inference
python yolo/pytorch_inference.py

# Run ONNX inference
python onnx/onnx_inference.py

# Compare models
python onnx/onnx_inference.py --compare

# Export to ONNX
python onnx/export_to_onnx.py

# Validate ONNX
python onnx/onnx_validator.py
```

### File Locations

- **Models**: `models/` directory
- **Input images**: `data/` directory
- **Results**: `output/` directory
- **Dependencies**: `requirements.txt`
