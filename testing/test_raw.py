from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Path to your model folder
model_path =r"C:\Users\lenovo\Downloads\llama-3.2-1b"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,  # Use half precision to save memory
    device_map="auto"  # Automatically use GPU if available
)

print("✓ Model loaded successfully!")
print(f"Model size: {model.num_parameters() / 1e9:.2f}B parameters")

# Quick test
prompt = "Hello, my name is"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_length=50)
print(f"\nTest generation: {tokenizer.decode(outputs[0])}")