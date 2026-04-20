# pip install -U "transformers>=4.42" accelerate peft torch --index-url https://download.pytorch.org/whl/cu121
# ^ adjust the torch index-url / CUDA version as needed. CPU works too (slower).

import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import PeftModel

# ---- Settings ----
TOKENIZER_ID = "yav1327/qwen-3-2b-intent-tokenizer-V1"
ADAPTER_ID   = "yav1327/qwen-3-2b-intent-model-V1"
BASE_ID      = "unsloth/Qwen3-1.7B-unsloth-bnb-4bit"          # base model to load under the adapter
MERGE_ADAPTER = False                    # set True to merge LoRA into base weights and save
SAVE_PATH_MERGED = "./qwen3-2b-intent-merged-V1"  # where to save the merged model/tokenizer

# ---- Load tokenizer (yours) ----
tokenizer = AutoTokenizer.from_pretrained(
    TOKENIZER_ID,
    use_fast=True,
    trust_remote_code=True,
)

# ---- Load base model and attach adapter ----
# device_map="auto" will put layers on GPU(s) if available; dtype can be fp16/bf16 for speed
model = AutoModelForCausalLM.from_pretrained(
    BASE_ID,
    device_map="auto",
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    trust_remote_code=True,
)

# Load the PEFT/Unsloth adapter repo (contains adapter_model.safetensors, adapter_config.json)
model = PeftModel.from_pretrained(
    model,
    ADAPTER_ID,
    device_map="auto",
)

# Optionally merge LoRA weights into the base model and unload adapters.
if MERGE_ADAPTER:
    print("Merging adapter into base weights…")
    model = model.merge_and_unload()  # returns a plain transformers model
    # Save a standalone, merged model + your tokenizer for easy reuse/deployment.
    print(f"Saving merged model to: {SAVE_PATH_MERGED}")
    model.save_pretrained(SAVE_PATH_MERGED, safe_serialization=True)
    tokenizer.save_pretrained(SAVE_PATH_MERGED)

# ---- Simple chat helper (uses your chat_template.jinja if present in tokenizer repo) ----
def chat(messages, max_new_tokens=64, temperature=0.1, top_p=0.9):
    # messages: list of {"role": "system"/"user"/"assistant", "content": "…"}
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    gen_cfg = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=temperature > 0,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    with torch.no_grad():
        start = time.time()
        out = model.generate(**inputs, generation_config=gen_cfg)
        end = time.time()
    print(f"Inference time: {end - start:.3f}s")
    return tokenizer.decode(out[0], skip_special_tokens=True)

# ---- Example: intent inference ----
example = [
    {"role": "system", "content": "You will be given a natural language query. Your task is to transform it into a JSON intent object that captures the structured meaning of the query."},
    {"role": "user", "content": "Datasets of forests at regions over 1000ft"}
]

print(chat(example, max_new_tokens=1024))
