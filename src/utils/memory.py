"""Memory management utilities"""
import torch
import gc

def clear_memory():
    """Clear GPU/CPU memory"""
    torch.cuda.empty_cache()
    gc.collect()
    
def print_memory_stats():
    """Print memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")