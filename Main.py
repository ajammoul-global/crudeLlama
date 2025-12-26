"""Main training script"""
import torch
from src.data.loader import DataLoader
from src.data.preprocess import DataPreprocessor
from src.model.model import ModelLoader
from src.model.lora import LoRAManager
from src.tunning.tune import ModelTrainer
from src.utils.memory import clear_memory, print_memory_stats
from src.utils.logger import print_section, print_step
from config import PathConfig

def main():
    print_section("FAKE NEWS DETECTION - TRAINING")
    
    
    clear_memory()
    
    
    print_step(1, 5, "Loading model...")
    model_loader = ModelLoader()
    model = model_loader.load_base_model()
    tokenizer = model_loader.load_tokenizer()
    print_memory_stats()
   

    # Make it trainable again
        #model = PeftModel.from_pretrained(base_model, FINE_TUNED_PATH)
        #model = model.merge_and_unload()  # Merge LoRA weights into base model
        #model = prepare_model_for_kbit_training(model)
   
    print_step(2, 5, "Applying LoRA...")
    lora_manager = LoRAManager()
    model = lora_manager.apply_lora(model)
    
    
    print_step(3, 5, "Loading data...")
    data_loader = DataLoader()
    dataset = data_loader.load_data()
    
    
    print_step(4, 5, "Preprocessing...")
    preprocessor = DataPreprocessor(tokenizer)
    tokenized_data = preprocessor.tokenize_dataset(dataset)
    
    clear_memory()
    
    
    print_step(5, 5, "Training...")
    trainer_manager = ModelTrainer(model, tokenizer)
    trainer = trainer_manager.train(
        tokenized_data['train'],
        tokenized_data['test']
    )
    
    
    print("\nSaving model...")
    model.save_pretrained(PathConfig.MODEL_OUTPUT_DIR)
    tokenizer.save_pretrained(PathConfig.MODEL_OUTPUT_DIR)
    print(f"✓ LoRA adapters saved to: {PathConfig.MODEL_OUTPUT_DIR}")
    
    
    print("\n" + "="*60)
    print("MERGING LoRA ADAPTERS WITH BASE MODEL")
    print("="*60)
    model_loader = ModelLoader()
    merged_model = model_loader.merge_and_save_model(
        model=model,
        tokenizer=tokenizer,
        adapter_path=str(PathConfig.MODEL_OUTPUT_DIR),
        output_path=str(PathConfig.MERGED_MODEL_DIR)
    )
    
    print("\n✅ Training complete!")
    print(f"\n📦 Available models:")
    print(f"   - LoRA adapters: {PathConfig.MODEL_OUTPUT_DIR}")
    print(f"   - Merged model:  {PathConfig.MERGED_MODEL_DIR}")

if __name__ == "__main__":
    main()