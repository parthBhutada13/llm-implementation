!pip install -q -U transformers accelerate bitsandbytes

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using device: {device}")
