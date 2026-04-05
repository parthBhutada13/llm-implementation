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

# prompt is a string containing the user query you want the model to answer.

# messages is a list of dictionaries; each dictionary has "role" (string: system/user/assistant)
# and "content" (string: actual message text) to structure the conversation.

# tokenizer.apply_chat_template(...) converts messages into a formatted text prompt;
# tokenize=False (bool) returns plain text, and add_generation_prompt=True (bool)
# adds a cue for the model to generate a response.

# tokenizer([text], return_tensors="pt") converts text into PyTorch tensors ("pt" = PyTorch),
# and .to(model.device) moves the inputs to the same device (CPU/GPU) as the model.

prompt = "explain what a large language model is to a 3rd year student of aiml in 2 sentences."

messages = [
    {"role": "system", "content": "you are a helpful ai assistant"},
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = True)
model_inputs = tokenizer([text], return_tensors = "pt").to(model.device)

# generate the output
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=250,
    temperature=0.7,
    do_sample=True
)

generated_ids = generated_ids[:, model_inputs["input_ids"].shape[-1]:]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(response)

# chat loop with memory

chat_history = []

def chat_with_memory(user_input):
    chat_history.append({"role": "user", "content": user_input})

    text = tokenizer.apply_chat_template(
        chat_history,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    generated_ids = generated_ids[:, model_inputs["input_ids"].shape[-1]:]

    assistant_answer = tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]

    chat_history.append({"role": "assistant", "content": assistant_answer})

    return assistant_answer
