"""Quick accuracy test on a few examples"""
import sys
from pathlib import Path

# When running this script directly (python testing/test.py) the top-level
# package path (project root) isn't on sys.path which causes imports like
# `from src.inference.predictor import ...` to fail. Insert the project
# root so absolute imports work both when running as a module and as a script.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.FakeNewsPredictor import FakeNewsPredictor

def quick_test():
    predictor = FakeNewsPredictor("./models/fine-tuned/fake_news_detector")
    
    # Test cases
    test_cases = [
        {"title": "Scientists cure cancer with lemon juice", "label": 0},  # Fake
        {"title": "Stock market closes at record high", "label": 1},  # Real
        {"title": "Aliens land in New York City", "label": 0},  # Fake
        {"title": "New study shows benefits of exercise", "label": 1},  # Real
    ]
    
    correct = 0
    for i, case in enumerate(test_cases):
        result = predictor.predict(case)
        is_correct = result['prediction'] == case['label']
        correct += is_correct
        
        print(f"\nTest {i+1}:")
        print(f"  Title: {case['title']}")
        print(f"  True: {'Fake' if case['label']==0 else 'Real'}")
        print(f"  Predicted: {result['label']}")
        print(f"  ✓" if is_correct else "  ✗")
    
    accuracy = correct / len(test_cases)
    print(f"\n{'='*50}")
    print(f"Quick Test Accuracy: {accuracy:.2%} ({correct}/{len(test_cases)})")
    print(f"{'='*50}")

if __name__ == "__main__":
    quick_test()