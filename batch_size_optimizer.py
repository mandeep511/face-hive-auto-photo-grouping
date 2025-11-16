"""
Utility to automatically determine optimal batch size based on available system resources.
"""
import platform
from typing import Optional

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def get_available_ram_gb() -> float:
    """
    Get available RAM in GB.
    Returns 0 if psutil is not available.
    """
    if not PSUTIL_AVAILABLE:
        return 0.0
    
    # Get available memory (not total, but what's actually free)
    mem = psutil.virtual_memory()
    # Use available memory (which accounts for buffers/cache that can be freed)
    available_gb = mem.available / (1024 ** 3)
    return available_gb


def get_gpu_memory_gb() -> Optional[float]:
    """
    Get available GPU memory in GB if CUDA is available.
    Returns None if GPU is not available.
    """
    if not TORCH_AVAILABLE:
        return None
    
    if not torch.cuda.is_available():
        return None
    
    try:
        # Get total GPU memory
        # Note: This is called before models are loaded, so we use total memory
        # and account for model loading in the memory estimation
        total_memory = torch.cuda.get_device_properties(0).total_memory
        # Get currently allocated memory (might be from other processes)
        allocated_memory = torch.cuda.memory_allocated(0)
        
        # Available memory = total - allocated
        # We'll reserve some for model weights (~500MB for YOLO)
        model_reserve_gb = 0.5
        free_memory = total_memory - allocated_memory
        available_memory = free_memory - (model_reserve_gb * 1024 ** 3)
        available_gb = max(0, available_memory) / (1024 ** 3)
        return available_gb
    except Exception:
        # If anything fails, return None to fall back to RAM
        return None


def estimate_memory_per_image() -> float:
    """
    Estimate memory usage per image in GB.
    This is a heuristic based on typical image processing needs:
    - Image loading: ~20MB (average for 2-5MP photos)
    - YOLO processing overhead: ~100MB (model activations, intermediate tensors)
    - Face crops storage: ~5MB (average 1-2 faces per image)
    - ArcFace processing: ~10MB
    Total: ~135MB per image (conservative estimate)
    """
    return 0.135  # GB per image


def calculate_optimal_batch_size(
    ram_buffer_percent: float = 25.0,
    min_batch_size: int = 1,
    max_batch_size: int = 64,
    prefer_gpu: bool = True,
) -> int:
    """
    Calculate optimal batch size based on available system resources.
    
    Args:
        ram_buffer_percent: Percentage of RAM to keep free (default 25%)
        min_batch_size: Minimum batch size to return (default 1)
        max_batch_size: Maximum batch size to return (default 64)
        prefer_gpu: If True, prefer GPU memory over RAM (default True)
    
    Returns:
        Optimal batch size as integer
    """
    # Try GPU first if available and preferred
    if prefer_gpu and TORCH_AVAILABLE:
        gpu_memory = get_gpu_memory_gb()
        if gpu_memory and gpu_memory > 1.0:  # At least 1GB GPU memory
            # GPU memory is typically more constrained, use smaller buffer
            usable_gpu_memory = gpu_memory * (1 - ram_buffer_percent / 100)
            memory_per_image = estimate_memory_per_image()
            # GPU processing is more efficient, so we can use slightly less memory per image
            gpu_memory_per_image = memory_per_image * 0.7  # 30% more efficient on GPU
            batch_size = int(usable_gpu_memory / gpu_memory_per_image)
            batch_size = max(min_batch_size, min(batch_size, max_batch_size))
            print(f"[batch_optimizer] GPU detected: {gpu_memory:.2f}GB available")
            print(f"[batch_optimizer] Calculated GPU batch size: {batch_size}")
            return batch_size
    
    # Fall back to RAM
    if not PSUTIL_AVAILABLE:
        print("[batch_optimizer] psutil not available, using default batch size of 8")
        return 8
    
    available_ram = get_available_ram_gb()
    if available_ram < 1.0:  # Less than 1GB available
        print(f"[batch_optimizer] Low RAM detected ({available_ram:.2f}GB), using minimum batch size")
        return min_batch_size
    
    # Calculate usable RAM (with buffer)
    usable_ram = available_ram * (1 - ram_buffer_percent / 100)
    memory_per_image = estimate_memory_per_image()
    
    # Calculate batch size
    batch_size = int(usable_ram / memory_per_image)
    batch_size = max(min_batch_size, min(batch_size, max_batch_size))
    
    print(f"[batch_optimizer] Available RAM: {available_ram:.2f}GB")
    print(f"[batch_optimizer] Usable RAM (with {ram_buffer_percent}% buffer): {usable_ram:.2f}GB")
    print(f"[batch_optimizer] Estimated memory per image: {memory_per_image:.3f}GB")
    print(f"[batch_optimizer] Calculated batch size: {batch_size}")
    
    return batch_size

