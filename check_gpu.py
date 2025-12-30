"""
GPU Detection and Verification Utility

This utility checks if GPU acceleration is available and properly configured
for both PyTorch and ONNX Runtime. It provides detailed information about
the detected hardware and suggests fixes if GPU is not detected.
"""

import sys


def check_pytorch_gpu():
    """
    Check if PyTorch can detect and use the GPU.
    
    Returns:
        bool: True if GPU is available, False otherwise
    """
    print("\n" + "=" * 60)
    print("Checking PyTorch GPU Support")
    print("=" * 60)
    
    try:
        import torch
        
        # Check CUDA availability
        cuda_available = torch.cuda.is_available()
        print(f"CUDA Available: {cuda_available}")
        
        if cuda_available:
            # Get CUDA version
            print(f"CUDA Version: {torch.version.cuda}")
            
            # Get number of GPUs
            gpu_count = torch.cuda.device_count()
            print(f"Number of GPUs: {gpu_count}")
            
            # Get GPU details
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                print(f"\nGPU {i}: {gpu_name}")
                print(f"  Memory: {gpu_memory:.2f} GB")
                
            # Test GPU with a simple tensor operation
            print("\nTesting GPU with tensor operation...")
            test_tensor = torch.randn(1000, 1000).cuda()
            result = test_tensor @ test_tensor
            print("GPU tensor operation successful!")
            
            return True
        else:
            print("\nWARNING: CUDA is not available!")
            print("\nPossible reasons:")
            print("1. PyTorch CPU-only version is installed")
            print("2. NVIDIA GPU drivers are not installed")
            print("3. CUDA toolkit is not installed")
            print("\nSolution:")
            print("Install PyTorch with CUDA support:")
            print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
            return False
            
    except ImportError:
        print("ERROR: PyTorch is not installed!")
        print("Install with: pip install torch torchvision torchaudio")
        return False
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return False


def check_onnxruntime_gpu():
    """
    Check if ONNX Runtime can use GPU acceleration.
    
    Returns:
        bool: True if GPU provider is available, False otherwise
    """
    print("\n" + "=" * 60)
    print("Checking ONNX Runtime GPU Support")
    print("=" * 60)
    
    try:
        import onnxruntime as ort
        
        # Get available providers
        available_providers = ort.get_available_providers()
        print(f"Available Execution Providers:")
        for provider in available_providers:
            print(f"  - {provider}")
        
        # Check if CUDA provider is available
        cuda_available = 'CUDAExecutionProvider' in available_providers
        
        if cuda_available:
            print("\nCUDA Execution Provider: Available")
            
            # Get CUDA provider options
            try:
                cuda_provider_options = ort.get_device()
                print(f"Default Device: {cuda_provider_options}")
            except:
                pass
            
            print("\nONNX Runtime is ready to use GPU!")
            return True
        else:
            print("\nWARNING: CUDA Execution Provider is not available!")
            print("\nPossible reasons:")
            print("1. onnxruntime (CPU-only) is installed instead of onnxruntime-gpu")
            print("2. CUDA libraries are not compatible")
            print("3. cuDNN is not installed")
            print("\nSolution:")
            print("Uninstall CPU version and install GPU version:")
            print("  pip uninstall onnxruntime")
            print("  pip install onnxruntime-gpu")
            return False
            
    except ImportError:
        print("ERROR: ONNX Runtime is not installed!")
        print("Install with: pip install onnxruntime-gpu")
        return False
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return False


def check_cuda_environment():
    """
    Check CUDA environment variables and system configuration.
    """
    print("\n" + "=" * 60)
    print("Checking CUDA Environment")
    print("=" * 60)
    
    import os
    
    # Check CUDA_PATH
    cuda_path = os.environ.get('CUDA_PATH')
    if cuda_path:
        print(f"CUDA_PATH: {cuda_path}")
    else:
        print("CUDA_PATH: Not set")
    
    # Check PATH for CUDA
    path = os.environ.get('PATH', '')
    cuda_in_path = 'cuda' in path.lower()
    print(f"CUDA in PATH: {cuda_in_path}")
    
    # Check cuDNN
    cudnn_path = os.environ.get('CUDNN_PATH')
    if cudnn_path:
        print(f"CUDNN_PATH: {cudnn_path}")
    else:
        print("CUDNN_PATH: Not set (may not be required)")


def print_summary(pytorch_ok, onnxruntime_ok):
    """
    Print overall summary and recommendations.
    
    Args:
        pytorch_ok: Whether PyTorch GPU is working
        onnxruntime_ok: Whether ONNX Runtime GPU is working
    """
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    if pytorch_ok and onnxruntime_ok:
        print("\nStatus: ALL GOOD!")
        print("Both PyTorch and ONNX Runtime can use your GPU.")
        print("Your RTX 4050 is ready for inference!")
        
        print("\nNext steps:")
        print("1. Run inference with PyTorch: python yolo/pytorch_inference.py")
        print("2. Run inference with ONNX: python onnx/onnx_inference.py")
        print("3. Launch the web interface: streamlit run frontend/app.py")
        
    elif pytorch_ok and not onnxruntime_ok:
        print("\nStatus: PARTIAL")
        print("PyTorch GPU works, but ONNX Runtime needs GPU support.")
        print("\nFix ONNX Runtime:")
        print("  pip uninstall onnxruntime")
        print("  pip install onnxruntime-gpu")
        
    elif not pytorch_ok and onnxruntime_ok:
        print("\nStatus: PARTIAL")
        print("ONNX Runtime GPU works, but PyTorch needs GPU support.")
        print("\nFix PyTorch:")
        print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        
    else:
        print("\nStatus: GPU NOT DETECTED")
        print("Neither PyTorch nor ONNX Runtime can use GPU.")
        
        print("\nSystem Requirements:")
        print("1. NVIDIA GPU (you have RTX 4050 - 6GB)")
        print("2. NVIDIA GPU drivers (latest recommended)")
        print("3. CUDA Toolkit 11.8 or 12.x")
        print("4. cuDNN library")
        
        print("\nInstallation steps:")
        print("1. Download and install NVIDIA GPU drivers:")
        print("   https://www.nvidia.com/Download/index.aspx")
        print("2. Download and install CUDA Toolkit:")
        print("   https://developer.nvidia.com/cuda-downloads")
        print("3. Install GPU-enabled packages:")
        print("   pip install -r requirements.txt")


def main():
    """
    Main function to run all GPU checks.
    """
    print("=" * 60)
    print("GPU Detection and Verification Utility")
    print("RTX 4050 6GB - GPU Configuration Check")
    print("=" * 60)
    
    # Check PyTorch
    pytorch_ok = check_pytorch_gpu()
    
    # Check ONNX Runtime
    onnxruntime_ok = check_onnxruntime_gpu()
    
    # Check CUDA environment
    check_cuda_environment()
    
    # Print summary
    print_summary(pytorch_ok, onnxruntime_ok)
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()

