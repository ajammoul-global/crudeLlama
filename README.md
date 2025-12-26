---
base_model: meta-llama/Llama-3.2-1B
library_name: peft
pipeline_tag: text-generation
tags:
- base_model:adapter:meta-llama/Llama-3.2-1B
- lora
- transformers
- fake-news-detection
---

# Fake News Detector - Llama-3.2-1B Fine-tuned with LoRA

Train a fake news detector on Kaggle with **automatic upload to Hugging Face Hub**. Simple, fast, and production-ready.

## 🚀 Quick Start (5 minutes)

### 1️⃣ Create Hugging Face Token
Go to https://huggingface.co/settings/tokens → Create token

### 2️⃣ Setup Kaggle
- Open https://kaggle.com/code → Create notebook
- Add dataset: "fake-news-dataset"
- Add secrets (Settings ⚙️ → Secrets):
  - `HF_TOKEN` = `hf_xxxxx...`
  - `HF_REPO_ID` = `yourusername/fake-news-detector`

### 3️⃣ Copy & Run
Copy these 3 cells into your Kaggle notebook:

**Cell 1:**
```python
!git clone https://github.com/ajammoul-global/crudeLlama.git
%cd crudeLlama
!pip install -q -r Requirements.txt
```

**Cell 2:**
```python
import os
from kaggle_secrets import UserSecretsClient
secrets = UserSecretsClient()
os.environ["HF_TOKEN"] = secrets.get_secret("HF_TOKEN")
os.environ["HF_REPO_ID"] = secrets.get_secret("HF_REPO_ID")
print(f"✅ Ready to train!")
```

**Cell 3:**
```python
!python Main.py
```

### ✅ Done!
Your model will automatically train and upload to Hugging Face Hub in ~50 minutes.

---

## � Documentation

- **Quick Start:** [`KAGGLE_QUICK_START.md`](KAGGLE_QUICK_START.md) (5 min read)
- **Detailed Guide:** [`KAGGLE_SETUP.md`](KAGGLE_SETUP.md) (15 min read)
- **Complete Info:** [`README_KAGGLE.md`](README_KAGGLE.md) (30 min read)
- **All Resources:** [`KAGGLE_RESOURCES.md`](KAGGLE_RESOURCES.md) (navigation)
- **Visual Guide:** [`KAGGLE_VISUAL_GUIDE.md`](KAGGLE_VISUAL_GUIDE.md) (diagrams)
- **Ready To Go:** [`KAGGLE_READY.md`](KAGGLE_READY.md) (summary)

---

## � Model Details

- **Base Model:** meta-llama/Llama-3.2-1B
- **Fine-tuning Method:** LoRA (Low-Rank Adaptation)
- **Framework:** Hugging Face Transformers + PEFT
- **Quantization:** 4-bit (BitsAndBytes)
- **Task:** Fake News Detection (Binary Classification)
- **License:** [Meta Llama License](https://huggingface.co/meta-llama/Llama-3.2-1B)

## 🎯 Use Case

This model detects whether news articles are real or fake. It's fine-tuned on the Fake and Real News Dataset with:
- ✅ 200 fake articles
- ✅ 200 real articles
- ✅ 25 training epochs
- ✅ LoRA adapters (~5MB)
- ✅ Merged complete model (~6GB)

## 🚀 Use in Production

### Load from Hugging Face Hub
```python
from src.inference.FakeNewsPredictor import FakeNewsPredictor

predictor = FakeNewsPredictor(
    model_path="yourusername/fake-news-detector",
    from_hub=True,
    use_merged=True
)

result = predictor.predict(
    title="Article Title",
    text="Article content..."
)
print(result)  # {'prediction': 0, 'label': 'REAL', 'confidence': 0.95}
```

### Load with Transformers
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("yourusername/fake-news-detector")
tokenizer = AutoTokenizer.from_pretrained("yourusername/fake-news-detector")
```

## 📊 Training Details

- **Dataset:** Fake and Real News Dataset (Kaggle)
- **Training Time:** ~40-50 minutes on Kaggle GPU
- **Optimizer:** AdamW with learning rate 2e-4
- **Batch Size:** 8
- **Epochs:** 25
- **Hardware:** Kaggle GPU (32GB, Nvidia P100/T4)

## 🔄 Model Merging

Two versions are saved after training:

| Model | Size | Use Case |
|-------|------|----------|
| **Merged** | ~6GB | Production (single file, faster inference) |
| **LoRA** | ~5MB | Research (flexible, base model required) |

## 📚 Project Files

### Key Directories
```
src/
├─ data/          # Data loading & preprocessing
├─ model/         # Model loading & LoRA
├─ inference/     # Prediction pipeline
├─ tunning/       # Training code
└─ utils/         # Logging & memory utilities

config/
├─ path.py        # Path configuration
├─ model.py       # Model configuration
└─ data.py        # Data configuration

models/
├─ base/          # Base model (Llama-3.2-1B)
└─ fine-tuned/    # Fine-tuned models
```

### Main Scripts
- `Main.py` - Training orchestration (run this on Kaggle!)
- `train_on_kaggle.py` - Kaggle wrapper
- `run.py` - Single prediction from CLI
- `test_kaggle_setup.py` - Verify environment

## 🔧 Local Development

### Install Dependencies
```bash
pip install -r Requirements.txt
```

### Train Locally
```bash
# With HF Hub upload (requires HF_TOKEN)
export HF_TOKEN=hf_xxxxx...
export HF_REPO_ID=yourusername/fake-news-detector
python Main.py

# Local only (no Hub upload)
python Main.py
```

### Test Predictions
```bash
python run.py "Article Title" "Article text..."
```

## ⚙️ Configuration

Edit `config/path.py` to customize:

```python
# Auto-detect Kaggle environment
IS_KAGGLE = os.environ.get('KAGGLE_KERNEL_RUN_TYPE') is not None

# Hugging Face Hub
HF_TOKEN = os.environ.get("HF_TOKEN", None)
HF_REPO_ID = os.environ.get("HF_REPO_ID", "ajammoul-global/fake-news-detector")
PUSH_TO_HUB = HF_TOKEN is not None  # Auto-enable if token exists
PRIVATE_REPO = False

# Model paths
MODEL_OUTPUT_DIR = OUTPUT_DIR / "fake_news_detector"           # LoRA
MERGED_MODEL_DIR = OUTPUT_DIR / "fake_news_detector_merged"    # Complete
```

## 🛠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA out of memory | Already optimized with 4-bit quantization. Check GPU. |
| HF_TOKEN not found | Set in Kaggle Secrets (exact name: `HF_TOKEN`) |
| Import errors | Run: `python test_kaggle_setup.py` |
| Slow training | Normal! ~1 sample/sec is expected. |
| Hub upload fails | Check token permissions (repo.content.write) |

For more: See [`KAGGLE_SETUP.md`](KAGGLE_SETUP.md) troubleshooting section.

## 📞 Support

- **Quick answers:** [`KAGGLE_QUICK_START.md`](KAGGLE_QUICK_START.md)
- **Step-by-step:** [`KAGGLE_SETUP.md`](KAGGLE_SETUP.md)
- **Full reference:** [`PROJECT_DOCUMENTATION.md`](PROJECT_DOCUMENTATION.md)

## 📄 License

This project uses Meta Llama 3.2 (1B) as the base model under the [Meta Llama License](https://huggingface.co/meta-llama/Llama-3.2-1B).

## 🎓 Citation

If you use this model, please cite:

```bibtex
@software{fake_news_detector_2025,
  title={Fake News Detector - Llama-3.2-1B Fine-tuned},
  author={Your Name},
  year={2025},
  howpublished={\url{https://huggingface.co/yourusername/fake-news-detector}},
  note={Fine-tuned on Fake and Real News Dataset}
}
```

---

**Ready to train?** Start with [`KAGGLE_QUICK_START.md`](KAGGLE_QUICK_START.md)! 🚀


[More Information Needed]

## Citation [optional]

<!-- If there is a paper or blog post introducing the model, the APA and Bibtex information for that should go in this section. -->

**BibTeX:**

[More Information Needed]

**APA:**

[More Information Needed]

## Glossary [optional]

<!-- If relevant, include terms and calculations in this section that can help readers understand the model or model card. -->

[More Information Needed]

## More Information [optional]

[More Information Needed]

## Model Card Authors [optional]

[More Information Needed]

## Model Card Contact

[More Information Needed]
### Framework versions

- PEFT 0.18.1.dev0