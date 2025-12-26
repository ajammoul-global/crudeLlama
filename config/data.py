"""Data configuration"""

class DataConfig:
    """Dataset parameters"""
    
    # Sampling
    SAMPLE_SIZE = 200  # Per class
    TEST_SIZE = 0.2
    RANDOM_SEED = 42
    
    # Preprocessing
    MAX_TITLE_LENGTH = 80
    
    # Label mapping
    LABEL_MAP = {0: "Fake", 1: "Real"}