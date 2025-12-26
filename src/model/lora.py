"""LoRA management"""
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from config import ModelConfig

class LoRAManager:
    """Manage LoRA fine-tuning"""
    
    def __init__(self, config=None):
        self.config = config or ModelConfig()
    
    def apply_lora(self, model,config=None):
        """Apply LoRA to model"""
        print("Applying LoRA...")
        
        # Prepare model
        model = prepare_model_for_kbit_training(model)
        
        # LoRA config
        lora_config = LoraConfig(
            r=self.config.LORA_R,
            lora_alpha=self.config.LORA_ALPHA,
            target_modules=self.config.LORA_TARGET_MODULES,
            lora_dropout=self.config.LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        # Apply LoRA
        model = get_peft_model(model, lora_config)
        model.config.use_cache = False
        
        # Print params
        model.print_trainable_parameters()
        
        print("✓ LoRA applied!")
        return model