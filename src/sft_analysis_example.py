import argparse
import os
import pandas as pd
import torch
from vllm import LLM, SamplingParams
from peft import AutoPeftModelForCausalLM
from datasets import load_dataset, Dataset
from utils import MLP, get_embed, plot_hist
from ppo_reward_model import extract_score
import pickle
from embed_text_package.embed_text import Embedder
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm', type=str, required=True)
    args = parser.parse_args()
    
    plot_dir = "../plot/sft"
    os.makedirs(plot_dir, exist_ok=True)
    restart = 64

    if args.llm == "llama":
        model_dir = "../data/sft/lora_10epoch"
        # model = AutoPeftModelForCausalLM.from_pretrained(f'{model_dir}/checkpoint-2400')
        test_dataset = load_dataset("stair-lab/airbench-sft", split="test")
        test_prompts = test_dataset['text'][:5]
    # model = model.merge_and_unload().to(torch.bfloat16)
    # model.save_pretrained(model_dir)
    
    sampling_params = SamplingParams(temperature=0.6, top_p=0.9, max_tokens=256)
    llm = LLM(model=model_dir)
    test_outputs = llm.generate(test_prompts, sampling_params)
    
    test_answers = [sample.text for o in test_outputs for sample in o.outputs]
    print(test_answers)
