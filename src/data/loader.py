"""Data loading utilities"""
import pandas as pd
from datasets import Dataset
from config import DataConfig, PathConfig

class DataLoader:
    """Load and prepare dataset"""
    
    def __init__(self, config=None):
        self.config = config or DataConfig()
    
    def load_data(self):
        """Load fake news dataset"""
        print("Loading data...")
        
        fake_df = pd.read_csv(PathConfig.FAKE_CSV)
        true_df = pd.read_csv(PathConfig.TRUE_CSV)
        
       
        fake_sample = fake_df.sample(
            n=self.config.SAMPLE_SIZE, 
            random_state=self.config.RANDOM_SEED
        )
        true_sample = true_df.sample(
            n=self.config.SAMPLE_SIZE, 
            random_state=self.config.RANDOM_SEED
        )
        
        
        fake_sample['label'] = 0
        true_sample['label'] = 1
        
        
        df = pd.concat([fake_sample, true_sample]).sample(
            frac=1, 
            random_state=self.config.RANDOM_SEED
        ).reset_index(drop=True)
        
       
        dataset = Dataset.from_pandas(df)
        
       
        data = dataset.train_test_split(
            test_size=self.config.TEST_SIZE,
            seed=self.config.RANDOM_SEED
        )
        
        print(f"✓ Train: {len(data['train'])} | Test: {len(data['test'])}")
        
        return data