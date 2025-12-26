print("hello world")
import os
import torch
import torch.nn as nn
import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
import pandas as pd
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

print("="*60)
print("FAKE NEWS DETECTION - FINE-TUNING LLAMA 3.2 1B")
print("="*60)

# ========== LOAD MODEL ==========
print("\n[1/6] Loading model...")

model_path = r"C:\Users\lenovo\Downloads\llama-3.2-1b"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    load_in_8bit=True,
    device_map='auto',  # Will use CPU since you have 2GB VRAM
)

tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token  # Set padding token
tokenizer.padding_side = "right"

print("✓ Model loaded!")

# ========== PREPARE MODEL FOR TRAINING ==========
print("\n[2/6] Preparing model for LoRA...")

# Freeze all parameters
for param in model.parameters():
    param.requires_grad = False
    if param.ndim == 1:
        param.data = param.data.to(torch.float32)

model.gradient_checkpointing_enable()
model.enable_input_require_grads()

class CastOutputToFloat(nn.Sequential):
    def forward(self, x):
        return super().forward(x).to(torch.float32)

model.lm_head = CastOutputToFloat(model.lm_head)

# ========== CONFIGURE LORA ==========
print("\n[3/6] Configuring LoRA...")

config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # Fixed: was k_proj, should be q_proj
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)

def print_trainable_parameters(model):
    """Print the number of trainable parameters in the model"""
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"Trainable params: {trainable_params:,} || All params: {all_param:,} || Trainable%: {100 * trainable_params / all_param:.2f}%")

print_trainable_parameters(model)

# ========== LOAD DATA ==========
print("\n[4/6] Loading dataset...")

fake_df = pd.read_csv("C:/Users/lenovo/Desktop/crudeLlama/Fake.csv")
true_df = pd.read_csv("C:/Users/lenovo/Desktop/crudeLlama/True.csv")

# Sample for POC
fake_sample = fake_df.sample(n=500, random_state=42)
true_sample = true_df.sample(n=500, random_state=42)

# Add labels - FIXED: true_sample should be label=1, not 0
fake_sample['label'] = 0  # Fake
true_sample['label'] = 1  # Real (was 0, now fixed!)

# Combine and shuffle
df = pd.concat([fake_sample, true_sample]).sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Dataset size: {len(df)}")
print(f"  - Fake: {(df['label']==0).sum()}")
print(f"  - Real: {(df['label']==1).sum()}")

# Convert to dataset
dataset = Dataset.from_pandas(df)
data = dataset.train_test_split(test_size=0.2, seed=42)

print(f"  - Train: {len(data['train'])}")
print(f"  - Test: {len(data['test'])}")

# ========== FORMAT DATA ==========
print("\n[5/6] Formatting and tokenizing data...")

def format_example(example):
    instruction = f"Classify this news as Real or Fake:\n\nTitle: {example['title']}\n\nText: {example['text'][:500]}\n\nAnswer:"
    label = "Real" if example['label'] == 1 else "Fake"
    example['formatted_text'] = f"{instruction} {label}"
    return example

data = data.map(format_example)

# Tokenize
def tokenize_function(examples):
    return tokenizer(
        examples['formatted_text'],
        truncation=True,
        max_length=512,
        padding='max_length'
    )

tokenized_data = data.map(
    tokenize_function,
    batched=True,
    remove_columns=data['train'].column_names
)

print("✓ Data prepared!")

# ========== TRAINING ==========
print("\n[6/6] Starting training...")

training_args = TrainingArguments(
    output_dir="./fake-news-detector-1b",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=False,  # CPU doesn't support fp16
    logging_steps=10,
    save_steps=50,
    eval_strategy="steps",
    eval_steps=50,
    save_total_limit=2,
    report_to="none",
    warmup_steps=10,
    load_best_model_at_end=True,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data['train'],
    eval_dataset=tokenized_data['test'],
    data_collator=data_collator,
)

print("\nTraining starting (this will take 1-3 hours on CPU)...\n")
trainer.train()

# ========== SAVE MODEL ==========
print("\n✓ Training complete! Saving model...")

model.save_pretrained("./fake-news-detector-1b")
tokenizer.save_pretrained("./fake-news-detector-1b")

print(f"\n{'='*60}")
print("SUCCESS! Model saved to: ./fake-news-detector-1b")
print(f"{'='*60}")
print("\nYou can now use this model to detect fake news!")