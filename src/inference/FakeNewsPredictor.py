"""Fake news predictor for inference - REUSES ModelLoader!"""
import torch
from src.model.model import ModelLoader  # ← REUSE!
from config import ModelConfig, PathConfig

class FakeNewsPredictor:
    """
    Make predictions on new articles
    Uses ModelLoader for consistency with training
    """
    
    def __init__(self, model_path=None):
        """
        Initialize predictor
        
        Args:
            model_path: Path to trained model (default: from config)
        """
        self.model_path = model_path or PathConfig.OUTPUT_DIR
        
        
        model_loader = ModelLoader()
        
     
        self.tokenizer = model_loader.load_tokenizer(self.model_path)
        
     
        self.model = model_loader.load_finetuned_model(self.model_path)
        
        print("✓ Predictor ready!")
    
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
    