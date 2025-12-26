"""Model loading for training and inference - REUSABLE!"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, PreTrainedTokenizerFast
from peft import AutoPeftModelForCausalLM, PeftModel
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
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=self.config.USE_DOUBLE_QUANT,
        )
        
        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            self.config.MODEL_NAME,
            quantization_config=bnb_config,
            device_map={"": torch.cuda.current_device()},
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
    
    def merge_and_save_model(self, model, tokenizer, adapter_path, output_path):
        """
        Merge LoRA adapters with base model and save as complete model
        
        Args:
            model: Model with LoRA adapters applied
            tokenizer: Tokenizer object
            adapter_path (str): Path where LoRA adapters were saved
            output_path (str): Path to save merged model
        
        Returns:
            merged_model: Model with LoRA weights merged into base model
        """
        print(f"\nMerging LoRA adapters with base model...")
        
        try:
            # Load the base model (without quantization for merging)
            print("Loading base model for merging...")
            base_model = AutoModelForCausalLM.from_pretrained(
                self.config.MODEL_NAME,
                device_map='auto',
                torch_dtype=torch.float16,
            )
            
            # Load LoRA adapters
            print(f"Loading LoRA adapters from {adapter_path}...")
            peft_model = PeftModel.from_pretrained(base_model, adapter_path)
            
            # Merge LoRA weights into base model
            print("Merging weights...")
            merged_model = peft_model.merge_and_unload()
            
            # Save merged model
            print(f"Saving merged model to {output_path}...")
            merged_model.save_pretrained(output_path)
            tokenizer.save_pretrained(output_path)
            
            print(f"✅ Merged model saved to: {output_path}")
            print(f"   Files: {list(__import__('pathlib').Path(output_path).glob('*'))[:5]}...")
            
            return merged_model
            
        except Exception as e:
            print(f"❌ Error merging model: {e}")
            raise
        
