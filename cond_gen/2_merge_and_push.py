import sys

import torch

from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

model_name_or_path = sys.argv[1]
output_dir = sys.argv[2]

model = AutoPeftModelForCausalLM.from_pretrained(model_name_or_path)
model = model.merge_and_unload().to(torch.bfloat16)
# model.save_pretrained(output_dir)
model.push_to_hub(output_dir)

tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path,
    trust_remote_code=True,
    use_fast=True,
)
# tokenizer.save_pretrained(output_dir)
tokenizer.push_to_hub(output_dir)
