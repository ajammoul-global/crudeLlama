import os
from pathlib import Path

class PathConfig:
    # Auto-detect environment
    IS_KAGGLE = os.environ.get('KAGGLE_KERNEL_RUN_TYPE') is not None
    
    if IS_KAGGLE:
        # Kaggle paths
        BASE_DIR = Path("/kaggle/working/crudeLlama")
        DATA_DIR = Path("/kaggle/input/fake-news-dataset")  # Your dataset name
        OUTPUT_DIR = Path("/kaggle/working/outputs")
    else:
        # Local paths (works on Windows, Mac, Linux)
        BASE_DIR = Path(__file__).resolve().parent.parent
        DATA_DIR = BASE_DIR / "data"
        OUTPUT_DIR = BASE_DIR / "models" / "fine-tunned"
    
    # ============================================
    # Hugging Face Hub Configuration
    # ============================================
    # Your HuggingFace repo name (format: "username/repo-name")
    # Example: "ajammoul-global/fake-news-detector"
    # For Kaggle: Set via Secrets (HF_REPO_ID)
    HF_REPO_ID = os.environ.get("HF_REPO_ID", "Ali-jammoul/fake-news-detector")
    
    # HF token (get from: https://huggingface.co/settings/tokens)
    # For Kaggle: Set via Kaggle Settings → Secrets as "HF_TOKEN"
    # For Local: Run: huggingface-cli login
    HF_TOKEN = os.environ.get("HF_TOKEN", None)
    
    # Push to Hub options
    # On Kaggle: Automatically enables if HF_TOKEN is set
    PUSH_TO_HUB = HF_TOKEN is not None  # Auto-enable if token exists
    PRIVATE_REPO = os.environ.get("HF_PRIVATE", "False").lower() == "true"
    
    # Data files
    FAKE_CSV = DATA_DIR / "raw" / "Fake.csv"
    TRUE_CSV = DATA_DIR / "raw" / "True.csv"
    
    # Model directories
    MODEL_OUTPUT_DIR = OUTPUT_DIR / "fake_news_detector"           # LoRA adapters only
    MERGED_MODEL_DIR = OUTPUT_DIR / "fake_news_detector_merged"    # Full merged model
    CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
    
    @classmethod
    def ensure_dirs(cls):
        '''Create directories if they don't exist'''
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cls.MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cls.MERGED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        cls.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
