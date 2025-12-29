# Fake News Detector - Docker Setup Guide

A production-ready fake news detection system using Meta's Llama-3.2-1B model with LoRA fine-tuning.

## 📋 Prerequisites

- Docker & Docker Compose installed
- 8GB+ RAM available
- GPU support (recommended for faster training)
- Git installed
- Kaggle API key (for Kaggle notebook setup)

## 🚀 Quick Start - Local Docker

### 1. Clone from GitHub

```bash
# Clone the repository
git clone https://github.com/yourusername/crudeLlama.git
cd crudeLlama

# Create and activate virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
# Install required packages
pip install -r Requirements.txt
```

### 3. Build and Run Docker Container

```bash
# Build the image
docker build -t fake-news-detector .

# Run the container
docker compose up --build
```

Your application will be available at `http://localhost:8000`.

## 🏋️ Training the Model Locally

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/crudeLlama.git
cd crudeLlama

# Install dependencies
pip install -r Requirements.txt
```

### Training Script

```bash
# Run the main training pipeline
python Main.py
```

The training pipeline automatically:
- ✅ Loads the base Llama-3.2-1B model
- ✅ Applies LoRA adapters for memory efficiency
- ✅ Preprocesses the fake news dataset
- ✅ Fine-tunes the model
- ✅ Merges weights and uploads to Hugging Face Hub

## 📓 Kaggle Notebook Setup

### Step 1: Create a New Kaggle Notebook

1. Go to [Kaggle.com](https://www.kaggle.com)
2. Click "Code" → "New Notebook"
3. Select "Notebook" as the environment

### Step 2: Set Up the Environment

Copy and paste this code in your first cell:

```python
# Install required packages
!pip install transformers peft bitsandbytes torch accelerate datasets -q

# Clone the repository
!git clone https://github.com/yourusername/crudeLlama.git
%cd crudeLlama
```

### Step 3: Load and Explore Data

```python
import os
import pandas as pd
from src.data.loader import DataLoader

# Load the dataset
data_loader = DataLoader()
dataset = data_loader.load_data()

# Preview the data
print(f"Dataset shape: {dataset['train'].shape}")
print(dataset['train'].head())
```

### Step 4: Train the Model on Kaggle

```python
import torch
from src.data.loader import DataLoader
from src.data.preprocess import DataPreprocessor
from src.model.model import ModelLoader
from src.model.lora import LoRAManager
from src.tunning.tune import ModelTrainer
from src.utils.memory import clear_memory, print_memory_stats
from src.utils.logger import print_section, print_step

print_section("FAKE NEWS DETECTION - TRAINING ON KAGGLE")

# Step 1: Load model
print_step(1, 5, "Loading model...")
model_loader = ModelLoader()
model = model_loader.load_base_model()
tokenizer = model_loader.load_tokenizer()
print_memory_stats()

# Step 2: Apply LoRA
print_step(2, 5, "Applying LoRA...")
lora_manager = LoRAManager()
model = lora_manager.apply_lora(model)

# Step 3: Load data
print_step(3, 5, "Loading data...")
data_loader = DataLoader()
dataset = data_loader.load_data()

# Step 4: Preprocess
print_step(4, 5, "Preprocessing data...")
preprocessor = DataPreprocessor(tokenizer)
tokenized_data = preprocessor.tokenize_dataset(dataset)

clear_memory()

# Step 5: Train
print_step(5, 5, "Training model...")
trainer_manager = ModelTrainer(model, tokenizer)
trainer = trainer_manager.train(tokenized_data)

print("✅ Training completed successfully!")
```

### Step 5: Run Inference

```python
from src.inference.FakeNewsPredictor import FakeNewsPredictor

# Load the trained model
predictor = FakeNewsPredictor()

# Make predictions
test_articles = [
    "Breaking news: Scientists discover new species",
    "Celebrity claims water cures all diseases"
]

for article in test_articles:
    result = predictor.predict(article)
    print(f"Article: {article[:50]}...")
    print(f"Prediction: {result['label']} (Confidence: {result['confidence']:.2f}%)\n")
```

### Step 6: Save Model to Kaggle Output

```python
import shutil

# Copy the fine-tuned model to output
output_path = "/kaggle/working/fake-news-detector-model"
source_path = "models/fine-tunned/fake_news_detector"

if os.path.exists(source_path):
    shutil.copytree(source_path, output_path, dirs_exist_ok=True)
    print(f"✅ Model saved to: {output_path}")
else:
    print("⚠️ Model not found")
```

## 🐳 Docker Deployment

### Building for Production

```bash
# Build for your architecture
docker build -t fake-news-detector:latest .

# For different CPU architectures
docker build --platform=linux/amd64 -t fake-news-detector:latest .  # Intel/AMD
docker build --platform=linux/arm64 -t fake-news-detector:latest .  # ARM (Apple Silicon)
```

### Pushing to Registry

```bash
# Tag your image
docker tag fake-news-detector:latest myregistry.com/fake-news-detector:latest

# Push to Docker Hub or your private registry
docker push myregistry.com/fake-news-detector:latest
```

## 📁 Project Structure

```
crudeLlama/
├── data/                    # Datasets (raw & processed)
├── models/                  # Base and fine-tuned models
├── src/                     # Source code
│   ├── data/               # Data loading & preprocessing
│   ├── model/              # Model loading & LoRA setup
│   ├── tunning/            # Training pipeline
│   ├── inference/          # Prediction pipeline
│   └── utils/              # Logger & memory utilities
├── config/                  # Configuration files
├── testing/                 # Test scripts
├── Main.py                 # Training entry point
├── Requirements.txt        # Python dependencies
├── Dockerfile              # Docker configuration
└── compose.yaml            # Docker Compose setup
```

## 🔧 Configuration

Modify settings in `config/` directory:

- `config/model.py` - Model hyperparameters
- `config/data.py` - Data loading settings
- `config/training.py` - Training configuration
- `config/path.py` - Path configurations

## 📊 Model Details

- **Base Model**: Meta-Llama-3.2-1B
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Quantization**: 4-bit BitsAndBytes
- **Framework**: PyTorch + Transformers
- **Dataset**: 400 labeled news articles (200 fake, 200 real)

## 🧠 Memory Efficiency

The system uses advanced techniques for memory optimization:

- 4-bit quantization reduces model size
- LoRA adapters require only 0.8% additional parameters
- Gradient checkpointing reduces memory footprint
- Automatic batch size optimization

## 📝 Logging and Monitoring

All training and inference activities are logged:

```bash
# Logs are saved in working directory
# Check logs with:
tail -f training.log
```

## 🐛 Troubleshooting

### Out of Memory Errors on Kaggle

Reduce batch size in `config/training.py`:

```python
TRAINING_BATCH_SIZE = 2  # Decrease from 4
EVAL_BATCH_SIZE = 2
```

### Model Not Found

Ensure the model path is correctly configured in `config/path.py`.

### GPU Not Detected

```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"
```

## 📚 References

- [Docker's Python guide](https://docs.docker.com/language/python/)
- [Kaggle Notebooks Documentation](https://www.kaggle.com/notebooks)
- [Transformers Library](https://huggingface.co/transformers/)
- [PEFT (LoRA) Documentation](https://huggingface.co/docs/peft/)

## 📄 License

[Specify your license here]

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.