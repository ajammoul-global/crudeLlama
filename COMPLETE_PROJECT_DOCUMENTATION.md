# Complete Project Documentation: Fake News Detector

## 📚 Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Configuration System](#configuration-system)
4. [Core Classes & Functions](#core-classes--functions)
5. [Data Pipeline](#data-pipeline)
6. [Model Training Pipeline](#model-training-pipeline)
7. [Inference Pipeline](#inference-pipeline)
8. [Utility Functions](#utility-functions)
9. [Main Entry Point](#main-entry-point)
10. [Usage Examples](#usage-examples)

---

## Project Overview

**Fake News Detector** is a production-ready machine learning system that fine-tunes Meta's Llama-3.2-1B language model using Low-Rank Adaptation (LoRA) to classify news articles as fake or real.

### Key Characteristics
- ✅ **Memory Efficient**: 4-bit quantization + LoRA adapters
- ✅ **Production Ready**: Automatic model merging + Hub upload
- ✅ **Cloud Optimized**: Works on Kaggle, Colab, local machines
- ✅ **Modular Design**: Reusable components for training and inference
- ✅ **Auto-Deployment**: Automatic upload to Hugging Face Hub

### Technology Stack
- **Base Model**: `meta-llama/Llama-3.2-1B` (1 billion parameters)
- **Fine-tuning Method**: PEFT (LoRA)
- **Quantization**: 4-bit BitsAndBytes
- **Framework**: PyTorch + Transformers + PEFT
- **Deployment**: Hugging Face Hub
- **Dataset**: Fake News Dataset (400 articles: 200 fake + 200 real)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   TRAINING PIPELINE                         │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  1. DATA LOADING                 2. MODEL PREPARATION       │
│     └─ DataLoader               └─ ModelLoader              │
│        └─ load_data()               └─ load_base_model()    │
│                                   └─ LoRAManager             │
│  3. DATA PREPROCESSING              └─ apply_lora()         │
│     └─ DataPreprocessor                                      │
│        └─ tokenize_dataset()    4. TRAINING                 │
│                                    └─ ModelTrainer           │
│  5. MODEL MERGING                    └─ train()             │
│     └─ ModelLoader                                           │
│        └─ merge_and_save_model()  6. HUB UPLOAD            │
│                                    └─ push_to_hub()         │
│                                                               │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                   INFERENCE PIPELINE                        │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  1. LOAD PREDICTOR              2. MAKE PREDICTION          │
│     └─ FakeNewsPredictor        └─ predict()               │
│        └─ _load_merged_model()                              │
│     OR └─ _load_lora_model()    3. RETURN RESULTS           │
│                                   └─ dict with prediction   │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## Configuration System

All configuration is centralized in the `config/` directory. This allows easy modification without touching source code.

### Files

#### `config/path.py` - File Paths & Hub Integration
**Purpose**: Centralize all file paths and Hugging Face configuration

```python
class PathConfig:
    IS_KAGGLE: bool              # Auto-detect Kaggle environment
    BASE_DIR: Path               # Project root directory
    DATA_DIR: Path               # Raw data directory
    OUTPUT_DIR: Path             # Model output directory
    
    HF_REPO_ID: str             # Hugging Face repo (e.g., "user/repo")
    HF_TOKEN: str               # HF token from environment
    PUSH_TO_HUB: bool           # Auto-upload to Hub if token exists
    PRIVATE_REPO: bool          # Make HF repo private
    
    FAKE_CSV: Path              # Path to Fake.csv
    TRUE_CSV: Path              # Path to True.csv
    MODEL_OUTPUT_DIR: Path      # LoRA adapters output
    MERGED_MODEL_DIR: Path      # Merged model output
    CHECKPOINT_DIR: Path        # Training checkpoints
    
    @classmethod
    def ensure_dirs(cls)        # Create all directories
```

**Key Behaviors**:
- Automatically detects Kaggle environment
- Adjusts paths based on environment
- Auto-enables `PUSH_TO_HUB` when `HF_TOKEN` is set
- Creates directories on demand

---

#### `config/model.py` - Model Configuration
**Purpose**: Define model architecture, quantization, LoRA settings

```python
class ModelConfig:
    MODEL_NAME: str             # "meta-llama/Llama-3.2-1B"
    LOAD_IN_4BIT: bool          # Enable 4-bit quantization
    QUANT_TYPE: str             # "nf4" (normal float 4-bit)
    COMPUTE_DTYPE: str          # "float16"
    USE_DOUBLE_QUANT: bool      # Double quantization
    
    LORA_R: int                 # 8 (rank of LoRA matrices)
    LORA_ALPHA: int             # 16 (scaling factor)
    LORA_TARGET_MODULES: list   # ["q_proj", "v_proj"]
    LORA_DROPOUT: float         # 0.05
    
    PAD_TOKEN: str              # "eos"
    PADDING_SIDE: str           # "right"
    MAX_SEQ_LENGTH: int         # 256 (max input tokens)
    MAX_LENGTH: int             # For inference
```

**Impact on Performance**:
- `LOAD_IN_4BIT=True`: Reduces memory from ~6GB to ~1GB
- `LORA_R=8`: Only 0.5% of parameters trainable
- `MAX_SEQ_LENGTH=256`: Limits input to 256 tokens

---

#### `config/data.py` - Dataset Configuration
**Purpose**: Control data sampling and preprocessing

```python
class DataConfig:
    SAMPLE_SIZE: int            # 200 (fake + real articles each)
    TEST_SIZE: float            # 0.2 (20% test, 80% train)
    RANDOM_SEED: int            # 42 (for reproducibility)
    MAX_TITLE_LENGTH: int       # 80 (truncate titles)
    LABEL_MAP: dict             # {0: "Fake", 1: "Real"}
```

---

#### `config/training.py` - Training Hyperparameters
**Purpose**: Control training loop and optimization

```python
class TrainingConfig:
    NUM_EPOCHS: int                     # 2
    BATCH_SIZE_TRAIN: int               # 1
    BATCH_SIZE_EVAL: int                # 1
    GRADIENT_ACCUMULATION_STEPS: int    # 4 (effective batch=4)
    
    LEARNING_RATE: float                # 2e-4
    WARMUP_STEPS: int                   # 10
    
    FP16: bool                          # True (mixed precision)
    GRADIENT_CHECKPOINTING: bool        # False (disabled for GPU compat)
    OPTIMIZER: str                      # "paged_adamw_8bit"
    
    LOGGING_STEPS: int                  # 10
    SAVE_STEPS: int                     # 100
    SAVE_TOTAL_LIMIT: int               # 1 (keep last checkpoint only)
    EVAL_STRATEGY: str                  # "steps"
    EVAL_STEPS: int                     # 100
```

---

## Core Classes & Functions

### 1. Data Loading (`src/data/loader.py`)

#### `DataLoader` Class

**Purpose**: Load and prepare dataset for training

**Methods**:

##### `__init__(config=None)`
- Initializes with optional `DataConfig`
- Uses default config if none provided

##### `load_data() -> Dict[str, Dataset]`
- Loads fake and real news articles from CSV files
- Samples `SAMPLE_SIZE` from each class
- Labels: 0 = Fake, 1 = Real
- Shuffles and splits: 80% train, 20% test
- Returns: `{"train": Dataset, "test": Dataset}`

**Example**:
```python
loader = DataLoader()
dataset = loader.load_data()
print(len(dataset['train']))  # 320 (400 * 0.8)
print(len(dataset['test']))   # 80  (400 * 0.2)
```

---

### 2. Data Preprocessing (`src/data/preprocess.py`)

#### `DataPreprocessor` Class

**Purpose**: Format and tokenize dataset

**Methods**:

##### `__init__(tokenizer, config=None)`
- Takes tokenizer instance
- Uses optional `DataConfig`

##### `format_example(example) -> dict`
- Converts raw example to instruction format
- Format: `"Classify: {title}\nAnswer: {label}"`
- Truncates title to `MAX_TITLE_LENGTH`

##### `tokenize_dataset(dataset) -> DatasetDict`
- Calls `format_example()` on all examples
- Tokenizes text to input_ids, attention_mask
- Truncates to `MAX_SEQ_LENGTH` (256 tokens)
- Pads all sequences to same length
- Returns tokenized dataset

**Example**:
```python
preprocessor = DataPreprocessor(tokenizer)
tokenized = preprocessor.tokenize_dataset(dataset)
# Returns: {"train": Dataset, "test": Dataset}
# Each example has: input_ids, attention_mask, token_type_ids
```

---

### 3. Model Loading (`src/model/model.py`)

#### `ModelLoader` Class

**Purpose**: Unified loader for base model, fine-tuned model, and tokenizer (reusable for training and inference)

**Key Design**: Single class used in both `Main.py` and `FakeNewsPredictor`

**Methods**:

##### `__init__(config=None)`
- Initializes with optional `ModelConfig`
- Stores configuration for later use

##### `load_base_model() -> PreTrainedModel`
- Loads `meta-llama/Llama-3.2-1B` with 4-bit quantization
- Creates `BitsAndBytesConfig` with settings from `ModelConfig`
- Maps model to current CUDA device
- Uses float16 compute dtype
- Returns: Quantized model ready for LoRA

**CUDA Configuration**:
```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
```

**Example**:
```python
loader = ModelLoader()
model = loader.load_base_model()
# Result: ~1GB memory usage (vs ~6GB unquantized)
```

---

##### `load_tokenizer(path=None) -> PreTrainedTokenizer`
- Loads tokenizer using AutoTokenizer
- Default path: `ModelConfig.MODEL_NAME`
- Sets `pad_token = eos_token`
- Sets `padding_side = "right"`
- Returns: Configured tokenizer

**Example**:
```python
tokenizer = loader.load_tokenizer()
# Returns: PreTrainedTokenizer configured and ready to use
```

---

##### `load_finetuned_model(model_path) -> PreTrainedModel`
- Loads fine-tuned model with LoRA adapters
- Uses `AutoPeftModelForCausalLM` (automatically loads base + LoRA)
- Sets to evaluation mode
- Returns: Model ready for inference

**Example**:
```python
model = loader.load_finetuned_model("models/fine-tunned/fake_news_detector")
# Returns: Model with LoRA adapters loaded and merged
```

---

##### `merge_and_save_model(model, tokenizer, adapter_path, output_path) -> PreTrainedModel`
- Merges LoRA adapters with base model into single file
- Steps:
  1. Load base model (unquantized for merging)
  2. Load LoRA adapters from `adapter_path`
  3. Call `merge_and_unload()` to merge weights
  4. Save complete merged model to `output_path`
  5. Save tokenizer to same location
- Returns: Merged model

**Example**:
```python
merged = loader.merge_and_save_model(
    model=model,
    tokenizer=tokenizer,
    adapter_path="models/fine-tunned/fake_news_detector",
    output_path="models/fine-tunned/fake_news_detector_merged"
)
# Result: Complete model ~6GB in output_path
```

**Output Files**:
- `pytorch_model.bin` - Model weights (~6GB)
- `config.json` - Model configuration
- `tokenizer.json` - Tokenizer vocabulary
- `tokenizer_config.json` - Tokenizer configuration

---

##### `push_to_hub(model_path, repo_name, token=None, private=False, model_type="merged") -> str`
- Uploads model to Hugging Face Hub
- Authenticates with token from parameter or `PathConfig.HF_TOKEN`
- Loads model and tokenizer from `model_path`
- Calls `model.push_to_hub()` and `tokenizer.push_to_hub()`
- Returns: Hub URL of uploaded model

**Example**:
```python
url = loader.push_to_hub(
    model_path="models/fine-tunned/fake_news_detector_merged",
    repo_name="Ali-jammoul/fake-news-detector",
    token="hf_...",
    model_type="merged"
)
# Returns: "https://huggingface.co/Ali-jammoul/fake-news-detector"
```

**Error Handling**:
- Checks for HF_TOKEN
- Provides helpful error messages
- Suggests troubleshooting steps

---

### 4. LoRA Management (`src/model/lora.py`)

#### `LoRAManager` Class

**Purpose**: Apply Low-Rank Adaptation to frozen base model

**Methods**:

##### `__init__(config=None)`
- Initializes with optional `ModelConfig`

##### `apply_lora(model) -> PreTrainedModel`
- Prepares model for 4-bit training: `prepare_model_for_kbit_training()`
- Creates `LoraConfig` with settings from `ModelConfig`:
  - `r=8`: Rank of LoRA matrices
  - `lora_alpha=16`: Scaling factor
  - `target_modules=["q_proj", "v_proj"]`: Only fine-tune attention
  - `lora_dropout=0.05`: Dropout in LoRA layers
- Applies LoRA with `get_peft_model()`
- Disables cache: `model.config.use_cache = False`
- Calls `model.print_trainable_parameters()` to show stats
- Returns: Model with LoRA adapters

**Example**:
```python
lora_manager = LoRAManager()
model = lora_manager.apply_lora(model)
# Output:
# trainable params: 1,835,008 || all params: 1,243,672,576 || trainable%: 0.15
```

**Why LoRA?**
- Only 0.15% of parameters trainable (~1.8M out of 1.2B)
- Reduces memory: ~80% less VRAM needed
- Faster training: Fewer parameters to update
- Smaller output: Adapters are ~5MB (vs 6GB full model)

---

### 5. Model Training (`src/tunning/tune.py`)

#### `ModelTrainer` Class

**Purpose**: Wrapper around Hugging Face Trainer for training loop

**Methods**:

##### `__init__(model, tokenizer, config=None)`
- Stores model, tokenizer, and `TrainingConfig`

##### `create_trainer(train_dataset, eval_dataset) -> Trainer`
- Creates `TrainingArguments` from `TrainingConfig`
- Creates `DataCollatorForLanguageModeling`
- Returns: Configured `Trainer` instance

**Training Arguments**:
```python
TrainingArguments(
    output_dir=PathConfig.OUTPUT_DIR,
    num_train_epochs=2,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,  # Effective batch=4
    learning_rate=2e-4,
    fp16=True,  # Mixed precision
    logging_steps=10,
    save_steps=100,
    eval_steps=100,
    gradient_checkpointing=False,
    optim="paged_adamw_8bit",
)
```

##### `train(train_dataset, eval_dataset) -> Trainer`
- Calls `create_trainer()` to build Trainer
- Calls `trainer.train()` to run training loop
- Returns: Trained Trainer instance

**Example**:
```python
trainer_manager = ModelTrainer(model, tokenizer)
trainer = trainer_manager.train(train_data, eval_data)
# Trains for 2 epochs on 320 samples
# Saves checkpoint every 100 steps
# Evaluates every 100 steps
```

**Training Process**:
1. Forward pass through model
2. Compute loss (language modeling)
3. Backward pass (gradient computation)
4. Update LoRA parameters (only ~1.8M params)
5. Save checkpoint every 100 steps
6. Evaluate on test set every 100 steps

---

### 6. Inference (`src/inference/FakeNewsPredictor.py`)

#### `FakeNewsPredictor` Class

**Purpose**: Make predictions on new articles (reuses `ModelLoader`)

**Key Design**: Uses same `ModelLoader` class as training for consistency

**Methods**:

##### `__init__(model_path=None, use_merged=True, from_hub=False)`
- Loads model for inference
- Parameters:
  - `model_path`: Path to model (default: merged model)
  - `use_merged`: Load merged model or LoRA adapters
  - `from_hub`: Load from HF Hub or local disk
- Calls either `_load_merged_model()` or `_load_lora_model()`

**Example**:
```python
# Load merged model from local disk (default)
predictor = FakeNewsPredictor()

# Load merged model from HF Hub
predictor = FakeNewsPredictor(
    model_path="Ali-jammoul/fake-news-detector",
    from_hub=True
)

# Load LoRA adapters
predictor = FakeNewsPredictor(use_merged=False)
```

---

##### `_load_merged_model()`
- Loads complete merged model (base + LoRA combined)
- Loads tokenizer
- Moves to GPU with `device_map='auto'`
- Sets to evaluation mode

**Performance**: Fast inference, single file load

---

##### `_load_lora_model()`
- Uses `ModelLoader` to load base + LoRA
- Loads base model and LoRA adapters separately

**Performance**: Slower (two file loads) but flexible

---

##### `predict(title, text="") -> dict`
- Makes prediction on article
- Parameters:
  - `title` (str): Article headline
  - `text` (str): Optional article body
- Returns prediction dictionary

**Prediction Format**:
```python
{
    'prediction': 0,           # 0=Fake, 1=Real, None=Unknown
    'label': 'Fake',
    'confidence': 0.9,
    'title': 'Article title',
    'raw_output': 'Classify: ...\nAnswer: Fake'
}
```

**Process**:
1. Format prompt: `"Classify: {title}\nAnswer:"`
2. Tokenize (max 256 tokens)
3. Generate with temperature=0.1 (deterministic)
4. Decode output
5. Extract label by checking if "Real" or "Fake" in output
6. Return result dict

**Example**:
```python
result = predictor.predict("Scientists discover aliens")
print(f"Prediction: {result['label']}")  # "Fake"
print(f"Confidence: {result['confidence']}")  # 0.9
```

---

### 7. Utility Functions (`src/utils/`)

#### `logger.py`

##### `print_section(title: str) -> None`
- Prints formatted section header
- Usage:
```python
print_section("Loading Data")
# Output:
# ============================================================
# Loading Data
# ============================================================
```

##### `print_step(step: int, total: int, message: str) -> None`
- Prints progress step
- Usage:
```python
print_step(1, 5, "Loading base model")
# Output:
# [1/5] Loading base model
```

---

## Data Pipeline

### Flow

```
Fake.csv (40K articles)          True.csv (20K articles)
    ↓                                 ↓
DataLoader.load_data()
    ├─ Sample 200 from Fake
    ├─ Sample 200 from Real
    ├─ Label: 0=Fake, 1=Real
    ├─ Shuffle
    └─ Split 80/20
         ↓
  {"train": 320, "test": 80}
         ↓
DataPreprocessor.tokenize_dataset()
    ├─ Format: "Classify: {title}\nAnswer: {label}"
    ├─ Tokenize
    └─ Pad to 256 tokens
         ↓
Tokenized Dataset (ready for training)
```

### Data Shapes

| Stage | Shape | Example |
|-------|-------|---------|
| Raw CSV | 40K rows | [title, text, label] |
| Sampled | 400 rows | [title, text, label] |
| Split | 320/80 | [title, text, label] |
| Formatted | 320/80 | ["Classify: title\nAnswer: Real"] |
| Tokenized | 320/80 | [input_ids, attention_mask] |

---

## Model Training Pipeline

### Training Process

```
1. INITIALIZATION
   ├─ Load base model (4-bit quantized)
   ├─ Apply LoRA adapters
   └─ Create trainer

2. DATASET PREPARATION
   ├─ Format examples
   ├─ Tokenize
   └─ Create data loader

3. TRAINING LOOP (2 epochs × 320 samples = 640 iterations)
   ├─ Forward pass
   ├─ Compute loss
   ├─ Backward pass (update LoRA weights only)
   ├─ Save checkpoint (every 100 steps)
   └─ Evaluate on test set (every 100 steps)

4. POST-TRAINING
   ├─ Merge LoRA with base model
   ├─ Save merged model to disk
   └─ Upload to Hugging Face Hub

5. RESULT
   ├─ Local: LoRA adapters (~5MB)
   ├─ Local: Merged model (~6GB)
   └─ Hub: Both versions available
```

### Memory Usage

| Component | Memory |
|-----------|--------|
| Base model (unquantized) | 6.0 GB |
| Base model (4-bit) | 1.0 GB |
| LoRA adapters | 0.2 GB |
| Optimizer state (8-bit) | 0.5 GB |
| Activation cache | 1.0 GB |
| **Total** | **~2.7 GB** |

---

## Inference Pipeline

### Single Prediction

```
User Input: "Article title"
    ↓
FakeNewsPredictor.predict()
    ├─ Format prompt
    ├─ Tokenize
    ├─ model.generate()
    ├─ Decode output
    ├─ Extract label
    └─ Return result dict
         ↓
Output: {"label": "Fake", "confidence": 0.9, ...}
```

### Batch Prediction

For multiple articles, call `predict()` in a loop:

```python
articles = ["Article 1", "Article 2", ...]
results = [predictor.predict(title) for title in articles]
```

---

## Utility Functions

All utility functions in `src/utils/` are simple logging helpers.

---

## Main Entry Point

### `Main.py` - Training Script

**Purpose**: Orchestrates entire training pipeline

**Flow**:

```python
1. Print environment detection (Kaggle or local)
2. Load configuration
3. DataLoader.load_data() → dataset
4. DataPreprocessor.tokenize_dataset() → tokenized_data
5. ModelLoader.load_base_model() → model
6. LoRAManager.apply_lora() → model with LoRA
7. ModelTrainer.train() → trainer
8. ModelLoader.merge_and_save_model() → merged_model
9. ModelLoader.push_to_hub() → Upload to HF (if token set)
10. Print final summary with Hub URLs
```

**Environment Detection**:
```python
IS_KAGGLE = os.environ.get('KAGGLE_KERNEL_RUN_TYPE') is not None

if IS_KAGGLE:
    print("🟠 KAGGLE ENVIRONMENT DETECTED")
else:
    print("💻 LOCAL ENVIRONMENT DETECTED")
```

**Hub Upload** (if `PUSH_TO_HUB=True`):
- Uploads merged model to `HF_REPO_ID`
- Uploads LoRA adapters to `HF_REPO_ID-lora`
- Prints Hub URLs for both

---

## Usage Examples

### Example 1: Train on Local Machine

```python
# 1. Setup
export HF_TOKEN="hf_..."
export HF_REPO_ID="username/fake-news-detector"

# 2. Run training
python Main.py

# Result:
# - Trained model: models/fine-tunned/fake_news_detector/
# - Merged model: models/fine-tunned/fake_news_detector_merged/
# - Uploaded to Hub: huggingface.co/username/fake-news-detector
```

### Example 2: Train on Kaggle

```python
# 1. Create Kaggle secret: HF_TOKEN = "hf_..."
# 2. Copy notebook code from KAGGLE_PATH_CONFIG_CELL.py
# 3. Run notebook
# 4. Model auto-uploads to Hub after training
```

### Example 3: Make Predictions

```python
from src.inference.FakeNewsPredictor import FakeNewsPredictor

# Load from local disk
predictor = FakeNewsPredictor()

# Single prediction
result = predictor.predict("Scientists find cure for cancer")
print(f"Label: {result['label']}")  # "Fake"

# Load from Hub
predictor = FakeNewsPredictor(
    model_path="username/fake-news-detector",
    from_hub=True
)
result = predictor.predict("COVID-19 vaccine causes 5G signals")
print(f"Label: {result['label']}")  # "Fake"
```

### Example 4: Merge and Export

```python
from src.model.model import ModelLoader
from src.export import export_model

# Merge LoRA with base
loader = ModelLoader()
merged = loader.merge_and_save_model(
    model=model,
    tokenizer=tokenizer,
    adapter_path="models/fine-tunned/fake_news_detector",
    output_path="models/fine-tunned/fake_news_detector_merged"
)

# Export as ZIP
export_model()  # Creates fake-news-detector-merged.zip
```

### Example 5: Upload to Hub

```python
from src.model.model import ModelLoader

loader = ModelLoader()

# Upload merged model
url1 = loader.push_to_hub(
    model_path="models/fine-tunned/fake_news_detector_merged",
    repo_name="username/fake-news-detector",
    token="hf_...",
    model_type="merged"
)
print(f"Merged: {url1}")

# Upload LoRA adapters
url2 = loader.push_to_hub(
    model_path="models/fine-tunned/fake_news_detector",
    repo_name="username/fake-news-detector-lora",
    token="hf_...",
    model_type="lora"
)
print(f"LoRA: {url2}")
```

---

## Configuration Reference

### Environment Variables

| Variable | Type | Purpose | Example |
|----------|------|---------|---------|
| `KAGGLE_KERNEL_RUN_TYPE` | str | Auto-set by Kaggle | "notebook" |
| `HF_TOKEN` | str | Hugging Face token | "hf_..." |
| `HF_REPO_ID` | str | Hub repo name | "user/repo" |
| `HF_PRIVATE` | str | Make repo private | "False" |

### Key Classes Summary

| Class | Module | Purpose | Key Methods |
|-------|--------|---------|-------------|
| `DataLoader` | `src/data/loader.py` | Load dataset | `load_data()` |
| `DataPreprocessor` | `src/data/preprocess.py` | Tokenize data | `tokenize_dataset()` |
| `ModelLoader` | `src/model/model.py` | Load/merge models | `load_base_model()`, `merge_and_save_model()`, `push_to_hub()` |
| `LoRAManager` | `src/model/lora.py` | Apply LoRA | `apply_lora()` |
| `ModelTrainer` | `src/tunning/tune.py` | Train model | `train()` |
| `FakeNewsPredictor` | `src/inference/FakeNewsPredictor.py` | Make predictions | `predict()` |

---

## Summary

This project implements a complete end-to-end machine learning pipeline for fake news detection:

1. **Data Pipeline**: Loads 400 articles, tokenizes, and splits for training
2. **Model Training**: Fine-tunes Llama-3.2-1B with LoRA (0.15% params)
3. **Model Merging**: Combines LoRA with base model for deployment
4. **Hub Integration**: Auto-uploads to Hugging Face Hub
5. **Inference**: Makes predictions on new articles

All components are modular, reusable, and production-ready! 🚀

---

**Last Updated**: December 26, 2025
**Framework**: PyTorch + Transformers + PEFT
**Model**: Llama-3.2-1B (1 billion parameters)
