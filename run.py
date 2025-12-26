"""Prediction script - uses merged model for fast inference"""
import sys
from pathlib import Path
from src.inference.FakeNewsPredictor import FakeNewsPredictor
from src.utils.logger import print_section
from config import PathConfig

def main():
    """Predict single article"""
    
    # Check arguments
    if len(sys.argv) < 2:
        print("Usage: python run.py 'Article title' ['Article text']")
        print("\nExample:")
        print("  python run.py 'Scientists discover aliens on Mars'")
        return
    
    title = sys.argv[1]
    text = sys.argv[2] if len(sys.argv) > 2 else ""
    
    # Load predictor (uses merged model by default)
    print("Loading model...")
    try:
        predictor = FakeNewsPredictor(use_merged=True)
    except Exception as e:
        print(f"Merged model not found: {e}")
        print("Falling back to LoRA adapters...")
        try:
            predictor = FakeNewsPredictor(
                model_path=str(PathConfig.MODEL_OUTPUT_DIR),
                use_merged=False
            )
        except Exception as e2:
            print(f"Error loading model: {e2}")
            return
    
    # Predict
    result = predictor.predict(title, text)
    
    # Display result
    print_section("PREDICTION RESULT")
    print(f"\nTitle: {title}")
    if text:
        print(f"Text: {text[:100]}...")
    print(f"\n→ Prediction: {result['label']}")
    print(f"→ Confidence: {result['confidence']:.0%}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
