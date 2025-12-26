"""Model configuration"""
import torch

class ModelConfig:
    """Model hyperparameters"""
    
    # Model selection
    MODEL_NAME = "meta-llama/Llama-3.2-1B"
    
    # Quantization
    LOAD_IN_4BIT = True
    QUANT_TYPE = "nf4"
    COMPUTE_DTYPE = torch.float16
    USE_DOUBLE_QUANT = True
    
    # LoRA
    LORA_R = 8
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.05
    LORA_TARGET_MODULES = ["q_proj", "v_proj"]
    
    # Tokenization
    MAX_LENGTH = 256
    PADDING = "max_length"
    TRUNCATION = True