"""Logging utilities"""

def print_section(title):
    """Print section header"""
    print("\n" + "="*60)
    print(title)
    print("="*60)

def print_step(step, total, message):
    """Print progress step"""
    print(f"\n[{step}/{total}] {message}")