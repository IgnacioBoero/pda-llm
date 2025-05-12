from peft import AutoPeftModelForCausalLM          # or ForSeq2SeqLM / …
from transformers import AutoTokenizer
import argparse

parser = argparse.ArgumentParser(description="Merge adapters and save the full checkpoint.")
parser.add_argument("--adapter_id", type=str, required=True, help="The adapter repository ID.")
parser.add_argument("--target_dir", type=str, required=True, help="The directory to save the final checkpoint.")
args = parser.parse_args()

ADAPTER_ID = args.adapter_id
TARGET_DIR = args.target_dir

model = (AutoPeftModelForCausalLM
         .from_pretrained(ADAPTER_ID,
                          torch_dtype="bfloat16",      # match adapter dtype
                          device_map="auto"))
model = model.merge_and_unload()                     # fuse LoRA → base :contentReference[oaicite:0]{index=0}

try:                                                 # adapter may have its own
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_ID, use_fast=True)
except EnvironmentError:                            # fall back to the base
    tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path,
                                              use_fast=True)

# 3️⃣  Save **everything** to one folder
model.save_pretrained(TARGET_DIR, safe_serialization=True)   # model.safetensors + config.json
tokenizer.save_pretrained(TARGET_DIR)                        # vocab, merges, tokenizer.json, …
if model.generation_config is not None:                      # newer Transformers only
    model.generation_config.save_pretrained(TARGET_DIR)      # generation_config.json

print("✅  Full checkpoint written to", TARGET_DIR)
