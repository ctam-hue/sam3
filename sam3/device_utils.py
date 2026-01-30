# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""
Device utility functions for cross-platform device selection (CPU, CUDA, MPS).
This module provides helpers to automatically detect and use the best available device.
"""

import torch


def get_default_device() -> torch.device:
    """
    Get the default device based on availability.
    
    Priority order:
    1. CUDA (if available)
    2. MPS (if available on Apple Silicon)
    3. CPU (fallback)
    
    Returns:
        torch.device: The best available device
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_device_str() -> str:
    """
    Get the default device as a string.
    
    Returns:
        str: Device string ('cuda', 'mps', or 'cpu')
    """
    return str(get_default_device().type)


def to_device(tensor_or_module, device=None, **kwargs):
    """
    Move a tensor or module to the specified device.
    
    Args:
        tensor_or_module: Tensor or nn.Module to move
        device: Target device (if None, uses default device)
        **kwargs: Additional arguments to pass to .to()
        
    Returns:
        Tensor or module on the target device
    """
    if device is None:
        device = get_default_device()
    return tensor_or_module.to(device, **kwargs)


def is_cuda_available() -> bool:
    """Check if CUDA is available."""
    return torch.cuda.is_available()


def is_mps_available() -> bool:
    """Check if MPS (Apple Silicon) is available."""
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def get_device_type(device) -> str:
    """
    Get the device type string from a device object.
    
    Args:
        device: torch.device or string
        
    Returns:
        str: Device type ('cuda', 'mps', or 'cpu')
    """
    if isinstance(device, str):
        return device.split(":")[0]
    elif isinstance(device, torch.device):
        return device.type
    else:
        raise ValueError(f"Unknown device type: {device}")


def supports_autocast(device) -> bool:
    """
    Check if the device supports torch.autocast.
    
    Args:
        device: torch.device or string
        
    Returns:
        bool: True if autocast is supported
    """
    device_type = get_device_type(device)
    # CUDA and CPU support autocast; MPS support varies by PyTorch version
    return device_type in ("cuda", "cpu")


def get_autocast_device_type(device=None):
    """
    Return device_type safe for torch.amp.autocast().
    Use this instead of hardcoding "cuda" to avoid the warning:
    "User provided device_type of 'cuda', but CUDA is not available. Disabling"

    Returns "cuda" only when CUDA is available; otherwise "cpu".
    """
    if device is not None:
        device_type = get_device_type(device)
        if device_type == "cuda" and not is_cuda_available():
            return "cpu"
        return device_type if device_type in ("cuda", "cpu") else "cpu"
    return "cuda" if is_cuda_available() else "cpu"


def empty_cache(device=None):
    """
    Empty the device cache if supported.
    
    Args:
        device: Target device (if None, uses default device)
    """
    if device is None:
        device = get_default_device()
    
    device_type = get_device_type(device)
    if device_type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device_type == "mps" and is_mps_available():
        # MPS also has empty_cache in recent PyTorch versions
        if hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()


def synchronize(device=None):
    """
    Synchronize the device if supported.
    
    Args:
        device: Target device (if None, uses default device)
    """
    if device is None:
        device = get_default_device()
    
    device_type = get_device_type(device)
    if device_type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device_type == "mps" and is_mps_available():
        # MPS synchronization
        if hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()


def get_device_memory_info(device=None):
    """
    Get memory information for the device if supported.
    
    Args:
        device: Target device (if None, uses default device)
        
    Returns:
        dict: Memory information (allocated, reserved, etc.) or None if not supported
    """
    if device is None:
        device = get_default_device()
    
    device_type = get_device_type(device)
    if device_type == "cuda" and torch.cuda.is_available():
        return {
            "allocated": torch.cuda.memory_allocated(device) / 1024**3,  # GB
            "reserved": torch.cuda.memory_reserved(device) / 1024**3,  # GB
        }
    elif device_type == "mps" and is_mps_available():
        # MPS memory tracking (if available)
        if hasattr(torch.mps, "current_allocated_memory"):
            return {
                "allocated": torch.mps.current_allocated_memory() / 1024**3,  # GB
            }
    return None


def set_device(device):
    """
    Set the current device (for CUDA).
    
    Args:
        device: Device to set as current
    """
    device_type = get_device_type(device)
    if device_type == "cuda" and torch.cuda.is_available():
        if isinstance(device, torch.device) and device.index is not None:
            torch.cuda.set_device(device.index)
        elif isinstance(device, int):
            torch.cuda.set_device(device)
        elif isinstance(device, str) and ":" in device:
            device_idx = int(device.split(":")[1])
            torch.cuda.set_device(device_idx)
