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
    
    # Data files
    FAKE_CSV = DATA_DIR / "raw" / "Fake.csv"
    TRUE_CSV = DATA_DIR / "raw" / "True.csv"
    
    # Model directories
    MODEL_OUTPUT_DIR = OUTPUT_DIR / "fake_news_detector"
    CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
    
    @classmethod
    def ensure_dirs(cls):
        '''Create directories if they don't exist'''
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cls.MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cls.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
