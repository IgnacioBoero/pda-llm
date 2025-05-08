import pandas as pd
import torch
from transformers import AutoTokenizer

from safe_rlhf.datasets import SupervisedToolsDataset

model_name_or_path = "Team-ACE/ToolACE-2-Llama-3.1-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
tokenizer.model_max_length = 4096
# Instead of "datasets=...", use the correct parameter:
train_dataset = SupervisedToolsDataset(
    [
        ("tools", 0.1),

        # If you have multiple, you can specify [("Foo", 0.6), ("Bar", 0.4), ...]
    ],
    tokenizer=tokenizer,
    lazy_tokenization=False,
    seed=42,
)