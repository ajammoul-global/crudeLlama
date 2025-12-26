# Fake News Detector: Fine-tuned Llama-3.2 with LoRA

## 📋 Project Overview

**Fake News Detector** is a production-ready fine-tuned LLaMA 3.2 (3B) model for detecting fake news articles, trained with LoRA (Low-Rank Adaptation) and optimized for deployment on **Kaggle** with automatic upload to **Hugging Face Hub**.

### Key Features
- ✅ **Train on Kaggle**: Automatic environment detection + secret management
- ✅ **Auto-upload to Hub**: Model automatically pushes to HF after training
- ✅ **Model Merging**: Creates both LoRA (~5MB) and merged (~6GB) versions
- ✅ **Two Deployment Options**: Load merged model or base + LoRA adapters
- ✅ **Tokenizer Fallback**: Handles edge cases with TokenizersBackend
- ✅ **Production Ready**: Works on Kaggle, Colab, local, or cloud

### Core Technologies
- **Base Model**: `meta-llama/Llama-3.2-1B` (lightweight, 1B parameters)
- **Fine-tuning**: PEFT LoRA (Low-Rank Adaptation)
- **Quantization**: 4-bit (BitsAndBytes) for GPU memory optimization
- **Framework**: PyTorch + Transformers + PEFT
- **Deployment**: Hugging Face Hub (automatic upload)
- **Dataset**: Fake News Dataset (200 fake + 200 real articles)
- **Training**: Optimized for Kaggle 32GB GPU with gradient accumulation

---

## 📁 Project Structure

```
crudeLlama/
├── config/                    # Configuration classes (paths, hyperparams, model config)
│   ├── __init__.py           # Exports ModelConfig, DataConfig, PathConfig, TrainingConfig
│   ├── model.py              # Model configuration (quantization, LoRA, tokenizer)
│   ├── data.py               # Dataset sampling and preprocessing config
│   ├── training.py           # Training hyperparameters
│   ├── path.py               # File paths (CSVs, output dirs)
│   └── login.py              # [Likely for HF token or credentials]
│
├── src/                       # Main source code (modular components)
│   ├── __init__.py
│   ├── data/                 # Data loading and preprocessing
│   │   ├── __init__.py
│   │   ├── loader.py         # DataLoader class (loads CSV, samples, splits)
│   │   └── preprocess.py     # DataPreprocessor (tokenization, formatting)
│   │
│   ├── model/                # Model initialization and LoRA
│   │   ├── __init__.py
│   │   ├── model.py          # ModelLoader (load base, fine-tuned, tokenizer)
│   │   └── lora.py           # LoRAManager (apply LoRA config to model)
│   │
│   ├── inference/            # Prediction on new articles
│   │   ├── __init__.py
│   │   └── FakeNewsPredictor.py   # FakeNewsPredictor class (reuses ModelLoader)
│   │
│   ├── tunning/              # Training loop and callbacks
│   │   ├── __init__.py
│   │   └── tune.py           # ModelTrainer (HF Trainer wrapper)
│   │
│   ├── utils/                # Utilities
│   │   ├── __init__.py
│   │   ├── logger.py         # Logging helpers (print_section, print_step)
│   │   └── memory.py         # GPU memory utilities (clear_memory, print_memory_stats)
│   │
│   └── export.py             # [Export model to ONNX or other formats]
│
├── data/                      # Datasets
│   ├── raw/                  # Raw CSV files
│   │   ├── Fake.csv          # ~40K fake news articles
│   │   └── True.csv          # ~20K real news articles
│   └── processed/            # [Processed datasets, intermediate files]
│
├── models/                    # Model checkpoints and fine-tuned models
│   ├── base/
│   │   └── llama-3.2-1b/    # Base model weights (downloaded from HF)
│   └── fine-tunned/          # Fine-tuned model directory
│       ├── fake_news_detector/  # Main fine-tuned model
│       │   ├── adapter_config.json
│       │   ├── adapter_model.safetensors
│       │   ├── tokenizer.json
│       │   ├── tokenizer_config.json
│       │   ├── README.md
│       │   └── checkpoint-160/   # Last checkpoint
│       └── checkpoints/      # Training checkpoints
│
├── testing/                   # Test and validation scripts
│   ├── __init__.py
│   ├── test.py              # Quick accuracy test on sample articles
│   ├── test_fine_tunned.py  # [Test fine-tuned model]
│   └── test_raw.py          # [Test raw/base model]
│
├── checkpoint-160/          # Last training checkpoint (temporary)
├── fake-news-detector-1b/   # [Exported model checkpoint]
│
├── Main.py                  # Main training script (orchestrates pipeline)
├── run.py                   # Inference script (single article prediction)
├── hello.py                 # [Test/demo script]
├── login.py                 # [HF login or credentials]
│
├── Requirements.txt         # Python dependencies
├── Dockerfile              # Docker containerization
├── compose.yaml            # Docker Compose config
├── README.md              # High-level project README
├── README.Docker.md       # Docker setup instructions
└── env                    # [Environment variables or venv]
```

---

## 🔧 Configuration System

### `config/__init__.py` — Central Import Point
Exports all configuration classes:
```python
from config.model import ModelConfig
from config.data import DataConfig
from config.training import TrainingConfig
from config.path import PathConfig
```

### `config/model.py` — Model & Tokenizer Configuration
**Purpose**: Define model architecture, quantization, LoRA, and tokenizer settings.

```python
class ModelConfig:
    MODEL_NAME = "meta-llama/Llama-3.2-1B"  # Lightweight 1B model
    LOAD_IN_4BIT = True
    QUANT_TYPE = "nf4"
    COMPUTE_DTYPE = "float16"
    USE_DOUBLE_QUANT = True
    
    LORA_R = 8
    LORA_ALPHA = 16
    LORA_TARGET_MODULES = ["q_proj", "v_proj"]
    LORA_DROPOUT = 0.05
    
    PAD_TOKEN = "eos"
    PADDING_SIDE = "right"
    MAX_SEQ_LENGTH = 256
```

**Key Settings**:
- `LOAD_IN_4BIT`: Quantize model to 4-bit (fits in ~1GB GPU memory vs ~6GB)
- `LORA_R=8`: Rank of LoRA adapters (low-rank approximation)
- `MAX_SEQ_LENGTH=256`: Truncate inputs to 256 tokens
- **Model Size**: 1B parameters (vs 3B previously) for faster training and inference

### `config/data.py` — Dataset Configuration
**Purpose**: Control data sampling, preprocessing, and labels.

```python
class DataConfig:
    SAMPLE_SIZE = 200          # 200 fake + 200 real
    TEST_SIZE = 0.2            # 80% train, 20% test
    RANDOM_SEED = 42
    MAX_TITLE_LENGTH = 80
    LABEL_MAP = {0: "Fake", 1: "Real"}
```

### `config/training.py` — Training Hyperparameters
**Purpose**: Control training loop (epochs, batch size, learning rate, etc.).

```python
class TrainingConfig:
    NUM_EPOCHS = 2
    BATCH_SIZE_TRAIN = 1
    GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch size = 1 * 4 = 4
    LEARNING_RATE = 2e-4
    OPTIMIZER = "paged_adamw_8bit"   # 8-bit optimizer for memory efficiency
    SAVE_STEPS = 100
    EVAL_STEPS = 100
```

**Important Notes**:
- Batch size = 1 + gradient accumulation steps = 4 → effective batch size of 4
- `paged_adamw_8bit` offloads optimizer state to CPU to save GPU memory

### `config/path.py` — File Paths & Kaggle/Hub Configuration
**Purpose**: Centralize all file paths (data, models, outputs) + configure HF Hub integration + detect Kaggle environment.

```python
import os

# Auto-detect Kaggle environment
IS_KAGGLE = os.path.exists('/kaggle/working')

# Paths (auto-adjust for Kaggle)
if IS_KAGGLE:
    BASE_DIR = '/kaggle/working'
    DATA_DIR = '/kaggle/input/fake-news-dataset'  # Dataset uploaded to Kaggle
else:
    BASE_DIR = r"C:\Users\lenovo\Desktop\crudeLlama"
    DATA_DIR = os.path.join(BASE_DIR, "data/raw")

FAKE_CSV = os.path.join(DATA_DIR, "Fake.csv")
TRUE_CSV = os.path.join(DATA_DIR, "True.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "models/fine-tunned/fake_news_detector")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "models/fine-tunned/checkpoints")

# Hugging Face Hub Configuration
HF_TOKEN = os.getenv("HF_TOKEN", "")  # Set in Kaggle secrets
HF_REPO_ID = "your-username/fake-news-detector"  # Update with your username
PUSH_TO_HUB = bool(HF_TOKEN)  # Auto-enable if token exists

# Kaggle Auto-Setup
if IS_KAGGLE and HF_TOKEN:
    from huggingface_hub import login
    login(token=HF_TOKEN)
```

**Key Features**:
- 🟠 **IS_KAGGLE**: Auto-detects Kaggle environment (checks `/kaggle/working`)
- 📁 **Path auto-adjustment**: Uses `/kaggle/working` on Kaggle, local paths otherwise
- 🔑 **HF_TOKEN**: Reads from Kaggle secrets (won't print in output)
- 🚀 **PUSH_TO_HUB**: Auto-enables when token exists (no manual config needed)
- 📤 **Auto-login**: Logs into HF Hub on Kaggle if token present

**Setup on Kaggle**:
1. Create a Kaggle secret named `HF_TOKEN` with your HF token value
2. Code automatically detects it and logs in
3. Model uploads to Hub after training completes

---

## 🚀 Core Components

### 1. Data Loading (`src/data/loader.py`)

**Class**: `DataLoader`

**Purpose**: Load CSV files, sample data, and create train/test split.

**Key Methods**:
- `load_data()` → Returns HF `datasets.DatasetDict` with 'train' and 'test' splits

**Workflow**:
1. Read `Fake.csv` and `True.csv` from raw data
2. Sample 200 articles from each (controlled by `DataConfig.SAMPLE_SIZE`)
3. Label fake=0, real=1
4. Shuffle and concatenate
5. Split: 80% train, 20% test

**Example Usage**:
```python
loader = DataLoader()
dataset = loader.load_data()  # {'train': Dataset, 'test': Dataset}
print(f"Train: {len(dataset['train'])}, Test: {len(dataset['test'])}")
```

---

### 2. Data Preprocessing (`src/data/preprocess.py`)

**Class**: `DataPreprocessor`

**Purpose**: Tokenize dataset, format prompts, and prepare for training.

**Key Methods**:
- `tokenize_dataset(dataset)` → Returns tokenized dataset with input_ids, attention_mask

**Tokenization Strategy**:
- Truncate to `MAX_SEQ_LENGTH` (256 tokens)
- Pad to fixed length
- Create labels (same as input_ids for causal LM)

**Example Usage**:
```python
preprocessor = DataPreprocessor(tokenizer)
tokenized_data = preprocessor.tokenize_dataset(dataset)  # HF DatasetDict
# Returns: {'train': Dataset(...), 'test': Dataset(...)}
```

---

### 3. Model Loading (`src/model/model.py`)

**Class**: `ModelLoader`

**Purpose**: Unified loader for base model, fine-tuned model, and tokenizer. **Key design: reusable across training and inference.**

**Key Methods**:

#### `load_base_model()`
- Loads `meta-llama/Llama-3.2-3B` with 4-bit quantization
- Used during training
- Returns quantized model

#### `load_finetuned_model(model_path)`
- Loads fine-tuned model with LoRA adapters from local directory
- Used during inference with LoRA adapters
- Expects: `adapter_model.safetensors` + tokenizer files at `model_path`
- Returns: Merged model with LoRA weights

#### `load_tokenizer(path=None)` ⭐ **WITH FALLBACK**
- Primary: Tries `AutoTokenizer.from_pretrained(path)`
- **Fallback**: If "TokenizersBackend" error, loads raw tokenizer from `tokenizer.json` using `tokenizers` library
- Sets special tokens (pad, eos) from `tokenizer_config.json`
- Returns: `PreTrainedTokenizer` or `PreTrainedTokenizerFast`

#### `merge_and_save_model(model, tokenizer, adapter_path, output_path)` ⭐ **NEW**
- Merges LoRA adapters with base model into a single complete model
- Used after training to create deployment-ready single-file model
- Steps:
  1. Load base model (unquantized)
  2. Load LoRA adapters from `adapter_path`
  3. Call `merge_and_unload()` to merge weights
  4. Save complete merged model to `output_path`
- Returns: Merged model (ready for inference)

**Tokenizer Loading Flow**:
```
AutoTokenizer.from_pretrained(path)
    ├─ Success → Return tokenizer
    └─ TokenizersBackend error
       └─ Fallback: tokenizers.Tokenizer.from_file() + wrap with PreTrainedTokenizerFast
```

**Example Usage**:
```python
loader = ModelLoader()

# For training
model = loader.load_base_model()      # Quantized 4-bit
tokenizer = loader.load_tokenizer()   # Default: base model

# After training: merge LoRA with base
merged_model = loader.merge_and_save_model(
    model=model,
    tokenizer=tokenizer,
    adapter_path="./models/fine-tunned/fake_news_detector",
    output_path="./models/fine-tunned/fake_news_detector_merged"
)

# For inference with merged model
tokenizer = loader.load_tokenizer(merged_model_path)
model = AutoModelForCausalLM.from_pretrained(merged_model_path)
```

---

### 4. LoRA Setup (`src/model/lora.py`)

**Class**: `LoRAManager`

**Purpose**: Apply LoRA (Low-Rank Adaptation) adapters to frozen base model.

**Key Methods**:
- `apply_lora(model)` → Returns model with LoRA adapters attached
- `print_trainable_params(model)` → Shows % of trainable params

**LoRA Configuration**:
- **Rank (r)**: 8 (low-rank matrices decomposition)
- **Alpha**: 16 (scaling factor)
- **Target Modules**: `["q_proj", "v_proj"]` (query and value projections in attention)
- **Dropout**: 0.05

**Why LoRA?**
- Only 0.5-5% of parameters are trainable
- Reduces memory usage by ~80%
- Faster training than full fine-tuning

---

### 5. Model Training (`src/tunning/tune.py`)

**Class**: `ModelTrainer`

**Purpose**: Wrapper around HuggingFace `Trainer` for the training loop.

**Key Methods**:
- `train(train_dataset, eval_dataset)` → Returns `Trainer` object after training

**Training Configuration** (from `TrainingConfig`):
- 2 epochs
- Batch size: 1 (gradient accumulation: 4 → effective: 4)
- Learning rate: 2e-4
- Optimizer: `paged_adamw_8bit` (8-bit, CPU-offloaded)
- FP16 precision: Enabled (mixed precision training)
- Gradient checkpointing: Enabled (save memory)
- Eval every 100 steps
- Save every 100 steps

---

### 6. Inference — Single Prediction (`src/inference/FakeNewsPredictor.py`)

**Class**: `FakeNewsPredictor`

**Purpose**: Make predictions on new articles. **Reuses `ModelLoader` for consistency.** **Supports both merged model and LoRA adapters.**

**Key Methods**:
- `__init__(model_path=None, use_merged=True)` → Loads merged model by default, falls back to LoRA
- `predict(title, text="")` → Returns dict with prediction, label, confidence
- `predict_batch(articles)` → Predict multiple articles
- `predict_csv(input_csv, output_csv)` → Batch predict from CSV file
- `_load_merged_model()` → Load fully merged model (fast, single file)
- `_load_lora_model()` → Load base + LoRA adapters (flexible, small size)

**Prediction Pipeline**:
1. Format input: `"Classify: {title}\nAnswer:"`
2. Tokenize with max_length=256
3. Generate with `max_new_tokens=10, temperature=0.1`
4. Decode response
5. Extract label by checking if "Real" or "Fake" appears in output
6. Return dict with prediction, label, confidence

**Output Format**:
```python
{
    'prediction': 1,           # 1=Real, 0=Fake, None=Unknown
    'label': 'Real',
    'confidence': 0.9,
    'title': 'Article title',
    'raw_output': 'Classify: Article...\nAnswer: Real'
}
```

**Example Usage**:
```python
# Load merged model (default, fast)
predictor = FakeNewsPredictor()
result = predictor.predict("Scientists discover aliens", "No text provided")
print(f"Prediction: {result['label']} ({result['confidence']:.0%})")

# Load with LoRA adapters (if merged not available)
predictor = FakeNewsPredictor(use_merged=False)

# Batch predict
articles = [
    {'title': 'Article 1', 'text': 'Some text'},
    {'title': 'Article 2', 'text': ''}
]
results = predictor.predict_batch(articles)

# Batch predict from CSV
predictor.predict_csv("input.csv", "output.csv")
```

---

## 📊 Workflow: Training & Inference

### Training Pipeline (`Main.py`)

```
1. ✅ Detect Environment (Kaggle vs Local)
   ├─ Display: 🟠 KAGGLE or 💻 LOCAL
   └─ Auto-adjust paths, enable GPU memory optimizations for Kaggle

2. Load base model (quantized 4-bit)
   └─ ModelLoader.load_base_model()

3. Apply LoRA adapters
   └─ LoRAManager.apply_lora(model)

4. Load dataset
   └─ DataLoader.load_data()  → train & test splits

5. Preprocess & tokenize
   └─ DataPreprocessor.tokenize_dataset()

6. Train with HF Trainer
   └─ ModelTrainer.train(train_data, test_data)
       - 2 epochs
       - Save checkpoints every 100 steps
       - Eval every 100 steps
       - 🟠 Kaggle: Optimized batch size & gradient accumulation

7. Save LoRA adapters locally
   └─ model.save_pretrained(OUTPUT_DIR)

8. ⭐ Merge LoRA with base model
   └─ ModelLoader.merge_and_save_model()
       - Load base model
       - Load LoRA adapters
       - Merge weights into single model
       - Save as complete model

9. ⭐ Auto-upload to Hugging Face Hub (if token configured)
   └─ Merged model: username/fake-news-detector
   └─ LoRA adapters: username/fake-news-detector-lora
   └─ All tokenizer files + README
```

**Kaggle Automatic Features**:
- 🔍 Auto-detects Kaggle environment from `/kaggle/working`
- 🔑 Auto-logs into HF Hub using Kaggle secret `HF_TOKEN`
- 📤 Auto-pushes model after training completes
- ✅ Shows environment & Hub config at startup
- 📊 Displays training progress with GPU stats

**Output**:
```
============================================================
Environment Detected: 🟠 KAGGLE
Hub Configuration: ENABLED
  - Repo: username/fake-news-detector
  - Token: ••••••••••••••••••
============================================================
...training...
✅ Training complete!
✅ Model merged successfully
✅ Models uploaded to Hub:
   - Merged: https://huggingface.co/username/fake-news-detector
   - LoRA: https://huggingface.co/username/fake-news-detector-lora
============================================================
```

**Files Created During Training**:

*LoRA Adapters Only (lightweight, ~5MB)*:
- `models/fine-tunned/fake_news_detector/adapter_model.safetensors`
- `models/fine-tunned/fake_news_detector/tokenizer.json`
- `models/fine-tunned/fake_news_detector/tokenizer_config.json`

*Merged Model (complete, ~6GB)*:
- `models/fine-tunned/fake_news_detector_merged/pytorch_model.bin` (or .safetensors)
- `models/fine-tunned/fake_news_detector_merged/config.json`
- `models/fine-tunned/fake_news_detector_merged/tokenizer.json`
- All other config files

*Checkpoints (periodic saves during training)*:
- `models/fine-tunned/checkpoints/checkpoint-XXX/`

### Inference Pipeline (`run.py` or `FakeNewsPredictor`)

```
1. Initialize predictor
   ├─ Try: Load merged model (fast, single file)
   │  └─ FakeNewsPredictor(use_merged=True)
   └─ Fallback: Load base + LoRA adapters (if merged unavailable)
      └─ FakeNewsPredictor(use_merged=False)

2. Format & tokenize input
   └─ "Classify: {title}\nAnswer:"

3. Generate prediction
   └─ model.generate(max_new_tokens=10)

4. Extract label from output
   └─ Check for "Real" or "Fake"

5. Return result
   └─ {'prediction': 1, 'label': 'Real', 'confidence': 0.9, ...}
```

**Two Deployment Options**:

| Aspect | Merged Model | LoRA Adapters |
|--------|--------------|---------------|
| **File size** | ~6GB (complete) | ~5MB (adapters only) |
| **Inference speed** | Fast (single load) | Slower (load base + adapters) |
| **Flexibility** | Fixed to one config | Can swap adapters |
| **Setup** | Copy one directory | Need base model + adapters |
| **Use case** | Production deployment | Research, multi-model serving |

---

## 📝 Entry Points & Usage

### 🟠 **NEW**: Training on Kaggle with Automatic Hub Upload

**Why Kaggle?**
- ✅ Free 32GB GPU (T4 or P100)
- ✅ No setup required (libraries pre-installed)
- ✅ Auto-logout on finish (no hanging processes)
- ✅ Built-in notebook environment

**Quick Setup**:
1. Go to https://www.kaggle.com/settings/account
2. Create a new notebook
3. Copy code from `KAGGLE_QUICK_START.md`
4. Add your HF token as a Kaggle secret (name it `HF_TOKEN`)
5. Run the notebook!

**What Happens Automatically**:
- Detects Kaggle environment
- Reads your HF token from Kaggle secrets
- Trains the model
- Merges LoRA with base model
- **Uploads everything to your HF Hub account**
- Shows Hub URLs at finish

**See These Guides**:
- 📄 `KAGGLE_QUICK_START.md` - Start here (5 minutes)
- 📄 `KAGGLE_SETUP.md` - Detailed walkthrough
- 📄 `KAGGLE_CHECKLIST.md` - Pre-training checklist
- 📄 `KAGGLE_RESOURCES.md` - Navigation guide

---

### 1. Training: `Main.py` (Local or Kaggle)

**Local Training**:
```bash
python Main.py
```

**Kaggle Training** (Recommended):
- Use `KAGGLE_QUICK_START.md` for copy-paste notebook code
- Add `HF_TOKEN` as Kaggle secret
- Run and auto-upload to Hub!

**What it does**: 
- 🔍 Auto-detects Kaggle vs local environment
- 📥 Loads base model + applies LoRA
- 📊 Loads data + preprocesses
- 🔄 Trains for 2 epochs
- 💾 Saves LoRA adapters locally
- 🔗 **Merges LoRA with base model**
- 📤 **Auto-uploads to Hugging Face Hub** (if token configured)
- 🟠 **Kaggle-specific**: Auto-detects secrets, optimized batch sizing

**Output**:
- Local: `models/fine-tunned/fake_news_detector/` (LoRA) + `fake_news_detector_merged/` (merged)
- Hub (automatic on Kaggle):
  - `username/fake-news-detector` (merged model)
  - `username/fake-news-detector-lora` (LoRA adapters)
- Both are accessible from any notebook: Kaggle, Colab, local

---

### 2. Single Prediction: `run.py`
```bash
python run.py "Article title" "Optional article text"
```
**Example**:
```bash
python run.py "Scientists discover aliens on Mars"
```
**What it does**:
- Loads merged model from local disk (or Hub if configured)
- Predicts on input article
- Prints prediction + confidence

**Output**:
```
Loading model...
✓ Predictor ready!
============================================================
PREDICTION RESULT
============================================================

Title: Scientists discover aliens on Mars
→ Prediction: Fake
→ Confidence: 90%
```

**Load from Hub**:
```bash
python -c "
from src.inference.FakeNewsPredictor import FakeNewsPredictor
p = FakeNewsPredictor('your-username/fake-news-detector', from_hub=True)
result = p.predict('Article title')
print(result)
"
```

---

### 3. Quick Test: `testing/test.py`
```bash
python .\testing\test.py
```
**What it does**:
- Auto-detects local fine-tuned model
- Tests on 4 sample articles
- Prints accuracy

**Output**:
```
Loading model from C:\...\fake_news_detector...
✓ Model loaded successfully!

Test 1:
  Title: Scientists cure cancer with lemon juice
  True: Fake
  Predicted: Fake
  ✓

Quick Test Accuracy: 100% (4/4)
```

---

## 🌐 Using Models from Hugging Face Hub

### Setup (One-time)

**On Local Machine**:
1. Get HF token: https://huggingface.co/settings/tokens
2. Authenticate: `huggingface-cli login`
3. Edit `config/path.py`:
   ```python
   HF_REPO_ID = "your-username/fake-news-detector"
   PUSH_TO_HUB = True
   ```
4. Run `Main.py` - models upload automatically after training

**On Kaggle** (Recommended):
1. Add HF token as Kaggle secret (name: `HF_TOKEN`)
2. Code auto-detects it and uploads (no config needed!)
3. See `KAGGLE_QUICK_START.md` for copy-paste notebook code

### After Training
Models automatically uploaded to:
- **Merged**: `https://huggingface.co/your-username/fake-news-detector` (~6GB, complete model)
- **LoRA**: `https://huggingface.co/your-username/fake-news-detector-lora` (~5MB, adapters only)

### Load from Hub (Any Environment)
```python
from src.inference.FakeNewsPredictor import FakeNewsPredictor

# Load merged model from Hub
predictor = FakeNewsPredictor(
    model_path="your-username/fake-news-detector",
    from_hub=True,
    use_merged=True
)

result = predictor.predict("Article title")
print(f"Prediction: {result['label']}")
```

### Deploy in Production
```python
# Load once, reuse for multiple predictions
predictor = FakeNewsPredictor("your-username/fake-news-detector", from_hub=True)

# Serve predictions via API or app
results = predictor.predict_batch([
    {"title": "Article 1"},
    {"title": "Article 2"}
])
```

### Full HF Hub Guide
See `HF_HUB_GUIDE.md` for detailed setup, troubleshooting, and advanced usage.

### Full Kaggle Integration Guide
See `KAGGLE_RESOURCES.md` for navigation to all Kaggle-specific guides (KAGGLE_QUICK_START.md, KAGGLE_SETUP.md, etc.)

---

## ⚙️ Key Technical Details

### Memory Optimization Strategies

1. **4-bit Quantization** (`ModelConfig.LOAD_IN_4BIT`)
   - Reduces model size from 3B params → ~800MB (instead of 6GB+)
   - Minimal accuracy loss

2. **LoRA Adapters** (`LoRAManager`)
   - Only train 0.5% of parameters
   - Freeze base model weights
   - Save only adapter_model.safetensors (~1.5MB)

3. **Gradient Checkpointing** (`TrainingConfig.GRADIENT_CHECKPOINTING`)
   - Recompute activations during backward pass instead of storing
   - Trade compute for memory

4. **8-bit Optimizer** (`paged_adamw_8bit`)
   - Offload optimizer states to CPU
   - Further reduce GPU memory usage

5. **Small Batch Size** (`BATCH_SIZE_TRAIN=1` + `GRADIENT_ACCUMULATION_STEPS=4`)
   - Effective batch=4 with tiny per-step memory footprint

### Tokenizer Fallback Mechanism

**Problem**: HF's `AutoTokenizer.from_pretrained()` fails if tokenizer_config.json references "TokenizersBackend" class that isn't registered.

**Solution** (in `ModelLoader.load_tokenizer()`):
```
Try AutoTokenizer → Fail with TokenizersBackend error
    ↓
Catch ValueError
    ↓
Load raw tokenizer: tokenizers.Tokenizer.from_file("tokenizer.json")
    ↓
Wrap: PreTrainedTokenizerFast(tokenizer_object=tk)
    ↓
Set special tokens from tokenizer_config.json
    ↓
Success ✓
```

---

## 📦 Dependencies

### Core Libraries
- **torch**: Deep learning framework
- **transformers**: HF Transformers (AutoTokenizer, AutoModel, etc.)
- **peft**: Parameter-Efficient Fine-Tuning (LoRA)
- **bitsandbytes**: 4-bit quantization
- **datasets**: HF datasets library (for dataset management)
- **pandas**: Data manipulation

### Optional
- **tensorboard**: Logging during training
- **accelerate**: Distributed training support
- **tokenizers**: Fast tokenizers (fallback loader)

### Installation
```bash
pip install -r Requirements.txt
```

---

## 🐛 Troubleshooting

### 1. TokenizersBackend Error
**Error**: `ValueError: Tokenizer class TokenizersBackend does not exist or is not currently imported.`

**Solution**: Already handled! The fallback mechanism in `ModelLoader.load_tokenizer()` catches this and loads from `tokenizer.json` directly.

**If it still fails**:
- Ensure `models/fine-tunned/fake_news_detector/tokenizer.json` exists
- Verify `tokenizers` package is installed: `pip install tokenizers`

---

### 2. Out of Memory (OOM)
**Error**: `CUDA out of memory. Tried to allocate X.XXGiB`

**Solutions**:
- Reduce `BATCH_SIZE_TRAIN` in `config/training.py`
- Increase `GRADIENT_ACCUMULATION_STEPS`
- Enable gradient checkpointing (already enabled by default)
- Reduce `MAX_SEQ_LENGTH` in `config/model.py`

---

### 3. Model Not Found
**Error**: `No local fine-tuned model directory found.`

**Solution**: Ensure trained model exists at `models/fine-tunned/fake_news_detector/` with:
- `adapter_model.safetensors`
- `tokenizer.json`
- `tokenizer_config.json`

Run `python Main.py` to train if missing.

---

### 4. Inference Accuracy Issues
If predictions are consistently wrong:
- Verify model was trained (check checkpoint age)
- Try different temperature: `self.model.generate(..., temperature=0.0)` for deterministic
- Check prompt format matches training prompt
- Manually inspect raw_output in prediction result

---

## 🔍 Code Reusability & Design Patterns

### Pattern 1: Configuration Centralization
All settings in `config/` module. Easy to modify without touching source code.

### Pattern 2: ModelLoader Reuse
Same `ModelLoader` class used in both training (`Main.py`) and inference (`FakeNewsPredictor`). Ensures consistency.

### Pattern 3: Modular Pipelines
- Data: `DataLoader` → `DataPreprocessor`
- Model: `ModelLoader` → `LoRAManager` → `ModelTrainer`
- Inference: `FakeNewsPredictor` (reuses ModelLoader)

### Pattern 4: Graceful Fallbacks
Tokenizer loading has a safe fallback mechanism for edge cases.

---

## 📚 Next Steps for Development

### Short-term Enhancements
1. Add model evaluation metrics (precision, recall, F1)
2. Implement confidence threshold filtering
3. Add batch CSV prediction with progress bar
4. Create model quantization export (ONNX, TFLite)

### Medium-term Features
1. Multi-class classification (not just Fake/Real)
2. Explainability: show which tokens contributed to prediction
3. A/B testing different LoRA configurations
4. Real-time API endpoint (FastAPI/Flask)

### Long-term
1. Larger base models (7B, 13B LLaMA versions)
2. Ensemble predictions
3. Domain-specific fine-tuning (e.g., medical vs political fake news)
4. Continuous retraining pipeline

---

## 📞 Support & Contact

For questions or issues:
1. Check logs in `models/fine-tunned/fake_news_detector/trainer_state.json`
2. Review training loss curves in checkpoints
3. Inspect `raw_output` from predictions for debugging
4. Verify path configuration in `config/path.py`

---

## 🎯 Quick Links

| Task | File/Link |
|------|-----------|
| **Train on Kaggle** | `KAGGLE_QUICK_START.md` 🟠 START HERE |
| **Detailed Kaggle Setup** | `KAGGLE_SETUP.md` |
| **Pre-training Checklist** | `KAGGLE_CHECKLIST.md` |
| **All Kaggle Guides** | `KAGGLE_RESOURCES.md` |
| **Hub Integration Details** | `HF_HUB_GUIDE.md` |
| **Model Merging Details** | `MERGE_IMPLEMENTATION.md` |
| **Train Locally** | `python Main.py` |
| **Make Predictions** | `python run.py "Article title"` |
| **Quick Test** | `python testing/test.py` |

---

**Last Updated**: December 26, 2025
**Base Model**: meta-llama/Llama-3.2-1B (1B parameters, lightweight)
**Framework**: PyTorch + HF Transformers + PEFT + BitsAndBytes
**Key Feature**: 🟠 Automatic Kaggle + Hub integration (no manual config needed!)
