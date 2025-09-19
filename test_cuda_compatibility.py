#!/usr/bin/env python3
"""
Test script for CUDA compatibility checking
"""

import sys
from pathlib import Path

# Add smart_segments to path
sys.path.insert(0, str(Path(__file__).parent / "smart_segments"))

from smart_segments.utils.system_utils import CUDACompatibilityChecker

def main():
    print("Testing CUDA compatibility...")
    
    # Check CUDA compatibility
    gpu_info = CUDACompatibilityChecker.check_cuda_compatibility()
    
    print(f"\nGPU Information:")
    print(f"  Has GPU: {gpu_info.has_gpu}")
    print(f"  GPU Name: {gpu_info.gpu_name}")
    print(f"  GPU Memory: {gpu_info.gpu_memory} MB")
    print(f"  CUDA Available: {gpu_info.cuda_available}")
    print(f"  CUDA Version: {gpu_info.cuda_version}")
    print(f"  PyTorch CUDA Version: {gpu_info.pytorch_cuda_version}")
    print(f"  Is Compatible: {gpu_info.is_compatible}")
    
    if gpu_info.compatibility_error:
        print(f"  Compatibility Error: {gpu_info.compatibility_error}")
    
    # Get recommended device
    device, reason = CUDACompatibilityChecker.get_recommended_device()
    print(f"\nRecommended Device: {device}")
    print(f"Reason: {reason}")

if __name__ == "__main__":
    main()