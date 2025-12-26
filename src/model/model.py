"""Model loading for training and inference - REUSABLE!"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, PreTrainedTokenizerFast
from peft import AutoPeftModelForCausalLM
from config import ModelConfig

class ModelLoader:
    """
    Unified model loader for both training and inference
    Handles base models and fine-tuned models
    """
    
    def __init__(self, config=None):
        """
        Initialize ModelLoader
        
        Args:
            config: ModelConfig instance (optional)
        """
        self.config = config or ModelConfig()
    
    def load_base_model(self):
        """
        Load base model for training
        
        Returns:
            model: Base model with quantization
        """
        print("Loading base model for training...")
        
        # Quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.config.LOAD_IN_4BIT,
            bnb_4bit_quant_type=self.config.QUANT_TYPE,
            bnb_4bit_compute_dtype=self.config.COMPUTE_DTYPE,
            bnb_4bit_use_double_quant=self.config.USE_DOUBLE_QUANT,
        )
        
        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            self.config.MODEL_NAME,
            quantization_config=bnb_config,
            device_map='auto',
            torch_dtype=torch.float16,
        )
        
        print("✓ Base model loaded!")
        return model
    
    def load_finetuned_model(self, model_path):
        """
        Load fine-tuned model for inference
        
        Args:
            model_path (str): Path to fine-tuned model checkpoint
        
        Returns:
            model: Fine-tuned model with LoRA adapters
        """
        print(f"Loading fine-tuned model from {model_path}...")
        
        
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_path,
            device_map='auto',
            torch_dtype=torch.float16,
        )
        model.eval()  
        
        print("✓ Fine-tuned model loaded!")
        return model
    
    def load_tokenizer(self, path=None):  
        """
        Load tokenizer (works for both training and inference)
        
        Args:
            path (str): Path to tokenizer (default: from config)
        
        Returns:
            tokenizer: Configured tokenizer
        """
        path = path or self.config.MODEL_NAME  
        
        print(f"Loading tokenizer from {path}...")
        
        
        tokenizer = AutoTokenizer.from_pretrained(path)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        print("✓ Tokenizer loaded via AutoTokenizer!")
        return tokenizer
        