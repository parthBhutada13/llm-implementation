!pip install -q -U transformers accelerate bitsandbytes

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using device: {device}")

# -q: quiet install (less output)
# -U: upgrade to latest version

# transformers: pre-trained NLP models (Hugging Face)
# accelerate: speeds up model execution on CPU/GPU
# bitsandbytes: memory optimization using 8-bit/4-bit models

# torch: PyTorch library for deep learning

# AutoTokenizer: converts text → tokens
# AutoModelForCausalLM: loads text generation models
# pipeline: simple API for using models

# CUDA: GPU computing (faster than CPU); uses GPU if available, else CPU

model_id = "Qwen/Qwen2.5-1.5B-Instruct"

print("downloading and loading the model")

# AutoTokenizer.from_pretrained(model_id) loads the tokenizer for the given model_id,
# where model_id is a string specifying the model name.

# AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")
# loads the text generation model; model_id is a string, torch_dtype (string option)
# automatically selects the data type (like float16/float32), and device_map (string option)
# automatically assigns the model to CPU/GPU.

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype = "auto", device_map = "auto")

print("model added successfully!")
