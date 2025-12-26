"""Data preprocessing"""
from config import DataConfig

class DataPreprocessor:
    """Format and tokenize data"""
    
    def __init__(self, tokenizer, config=None):
        self.tokenizer = tokenizer
        self.config = config or DataConfig()
    
    def format_example(self, example):
        """Format as instruction"""
        instruction = f"Classify: {example['title'][:self.config.MAX_TITLE_LENGTH]}\nAnswer:"
        label = self.config.LABEL_MAP[example['label']]
        example['text'] = f"{instruction} {label}"
        return example
    
    def tokenize_dataset(self, dataset):
        """Tokenize dataset"""
        print("Formatting and tokenizing...")
        
        
        dataset = dataset.map(self.format_example)
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                truncation=True,
                max_length=256,
                padding='max_length'
            )
        
        tokenized = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset['train'].column_names
        )
        
        print("✓ Tokenization complete!")
        return tokenized