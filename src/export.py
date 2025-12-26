"""Export model as ZIP"""
import shutil
from config import PathConfig

def export_model():
    """Create ZIP file"""
    print("Creating ZIP file...")
    
    shutil.make_archive(
        'fake-news-detector',
        'zip',
        PathConfig.OUTPUT_DIR
    )
    
    print(f"✓ Model exported: {PathConfig.EXPORT_PATH}")
    

if __name__ == "__main__":
    export_model()