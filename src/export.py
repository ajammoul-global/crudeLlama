"""Export fully merged model as ZIP"""
import os
import shutil
from pathlib import Path
from config import PathConfig

def export_model(merged_model_path=None):
    """
    Export fully merged model as ZIP file.
    
    Args:
        merged_model_path (str): Path to merged model directory.
                                 Defaults to PathConfig.MERGED_MODEL_DIR if not provided.
    """
    if merged_model_path is None:
        merged_model_path = os.path.join(
            PathConfig.OUTPUT_DIR.replace('/fake_news_detector', ''),
            'fake_news_detector_merged'
        )
    
    # Verify merged model exists
    if not os.path.exists(merged_model_path):
        raise FileNotFoundError(
            f"❌ Merged model not found at: {merged_model_path}\n"
            f"Run training first with: python Main.py"
        )
    
    # Get required files
    required_files = [
        'pytorch_model.bin',  # or model.safetensors
        'config.json',
        'tokenizer.json',
        'tokenizer_config.json'
    ]
    
    model_files = os.listdir(merged_model_path)
    has_model = any(f in model_files for f in ['pytorch_model.bin', 'model.safetensors'])
    
    if not has_model:
        raise FileNotFoundError(
            f"❌ Model weights not found in: {merged_model_path}\n"
            f"Expected: pytorch_model.bin or model.safetensors"
        )
    
    print("=" * 60)
    print("EXPORTING FULLY MERGED MODEL")
    print("=" * 60)
    print(f"Source: {merged_model_path}")
    
    # Create ZIP file
    zip_name = 'fake-news-detector-merged'
    zip_path = shutil.make_archive(
        zip_name,
        'zip',
        merged_model_path,
        '.'
    )
    
    zip_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    
    print(f"✅ Model exported: {zip_path}")
    print(f"📦 ZIP size: {zip_size_mb:.1f} MB")
    print("=" * 60)
    print("\n📝 Contents:")
    print("  - pytorch_model.bin (or model.safetensors) - Complete merged model weights")
    print("  - config.json - Model configuration")
    print("  - tokenizer.json - Tokenizer vocab & config")
    print("  - All other model files (generation_config.json, etc.)")
    print("\n✨ This is a fully standalone model - no LoRA adapters needed!")
    print("=" * 60)
    

if __name__ == "__main__":
    export_model()