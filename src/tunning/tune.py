"""Training manager"""
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from config import TrainingConfig, PathConfig

class ModelTrainer:
    """Manage model training"""
    
    def __init__(self, model, tokenizer, config=None):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or TrainingConfig()
    
    def create_trainer(self, train_dataset, eval_dataset):
        """Create Trainer instance"""
        # Training arguments
        training_args = TrainingArguments(
            output_dir=PathConfig.OUTPUT_DIR,
            num_train_epochs=self.config.NUM_EPOCHS,
            per_device_train_batch_size=self.config.BATCH_SIZE_TRAIN,
            per_device_eval_batch_size=self.config.BATCH_SIZE_EVAL,
            gradient_accumulation_steps=self.config.GRADIENT_ACCUMULATION_STEPS,
            
            learning_rate=self.config.LEARNING_RATE,
            fp16=self.config.FP16,
            logging_steps=self.config.LOGGING_STEPS,
            save_steps=self.config.SAVE_STEPS,
            eval_strategy=self.config.EVAL_STRATEGY,
            eval_steps=self.config.EVAL_STEPS,
            save_total_limit=self.config.SAVE_TOTAL_LIMIT,
            gradient_checkpointing=False,
            gradient_checkpointing_kwargs=None,
            optim=self.config.OPTIMIZER,
            report_to="none",
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator
        )
        
        return trainer
    
    def train(self, train_dataset, eval_dataset):
        """Train model"""
        print("\nStarting training...")
        
        trainer = self.create_trainer(train_dataset, eval_dataset)
        trainer.train()
        
        print("✓ Training complete!")
        return trainer