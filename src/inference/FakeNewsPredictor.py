"""Fake news predictor for inference - REUSES ModelLoader!"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.model.model import ModelLoader  # ← REUSE!
from config import ModelConfig, PathConfig

class FakeNewsPredictor:
    """
    Make predictions on new articles
    Uses ModelLoader for consistency with training
    Loads merged model by default (LoRA + base model combined)
    Can load from local disk or Hugging Face Hub
    """
    
    def __init__(self, model_path=None, use_merged=True, from_hub=False):
        """
        Initialize predictor
        
        Args:
            model_path: Path to trained model (default: merged model)
                       Can be local path or HF repo name (e.g., "username/repo-name")
            use_merged: If True, load merged model; if False, load LoRA adapters
            from_hub: If True, treat model_path as HF repo name; if False, treat as local path
        """
        if model_path is None:
            # Use merged model by default (faster, single file)
            self.model_path = str(PathConfig.MERGED_MODEL_DIR)
            from_hub = False
            use_merged = True
        else:
            self.model_path = model_path
        
        print(f"Loading predictor from: {self.model_path}")
        print(f"Mode: {'Merged model' if use_merged else 'LoRA adapters'}")
        print(f"Source: {'Hugging Face Hub' if from_hub else 'Local disk'}")
        
        self.from_hub = from_hub
        
        if use_merged:
            # Load merged model directly (no LoRA needed)
            self._load_merged_model()
        else:
            # Load base model + LoRA adapters
            self._load_lora_model()
        
        print("✓ Predictor ready!")
    
    def _load_merged_model(self):
        """Load fully merged model (no LoRA)"""
        print(f"Loading tokenizer from {self.model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"Loading merged model from {self.model_path}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map='auto',
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        self.model.eval()
    
    def _load_lora_model(self):
        """Load base model with LoRA adapters"""
        model_loader = ModelLoader()
        self.tokenizer = model_loader.load_tokenizer(self.model_path)
        self.model = model_loader.load_finetuned_model(self.model_path)
    
    def predict(self, title, text=""):
        """
        Predict if article is fake or real
        
        Args:
            title (str): Article title
            text (str): Article text (optional)
        
        Returns:
            dict: Prediction results
        """
      
        if text:
            prompt = f"Classify: {title[:80]}\nText: {text[:500]}\nAnswer:"
        else:
            prompt = f"Classify: {title[:80]}\nAnswer:"
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=ModelConfig.MAX_LENGTH
        ).to(self.model.device)
       
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
       
        if "Real" in response and "Fake" not in response:
            prediction = 1
            label = "Real"
            confidence = 0.9
        elif "Fake" in response and "Real" not in response:
            prediction = 0
            label = "Fake"
            confidence = 0.9
        else:
            prediction = None
            label = "Unknown"
            confidence = 0.5
        
        return {
            'prediction': prediction,
            'label': label,
            'confidence': confidence,
            'title': title,
            'raw_output': response
        }
    