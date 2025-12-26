"""Prediction script"""
from src.inference.FakeNewsPredictor import FakeNewsPredictor

def main():
    # Initialize predictor
    predictor = FakeNewsPredictor(model_path="./models/fine-tuned/fake_news_detector")
    
    # Example prediction
    news = {
        'title': "Scientists discover aliens on Mars",
        'text': "A team of researchers claims to have found evidence of alien life..."
    }
    
    result = predictor.predict(news)
    
    print(f"Prediction: {result['label']}")
    print(f"Confidence: {result['confidence']:.2f}")

if __name__ == "__main__":
    main()