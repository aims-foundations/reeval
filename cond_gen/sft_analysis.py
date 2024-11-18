import argparse
import os
import pickle

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from embed_text_package.embed_text_v2 import Embedder
from ppo_reward_model import extract_score
from transformers import GenerationConfig
from utils.utils import plot_hist
from vllm import LLM, SamplingParams


def call_diff(ds, gt_zs, reward_model, restart):
    dataloader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False)
    emb = embdr.get_embeddings(dataloader, "meta-llama/Meta-Llama-3-8B", ["text"])
    embs = emb["text"]

    pred_zs = reward_model.predict(embs).tolist()
    pred_zs = torch.tensor(pred_zs).reshape(-1, restart)
    # >>> batch_size * restart

    # gt_zs = np.array([gt_zs for _ in range(restart)]).T
    gt_zs_tensor = torch.tensor(gt_zs)[:, None].expand(-1, restart)

    best_diffs = torch.abs(pred_zs - gt_zs_tensor).min(dim=1).values
    
    return best_diffs.cpu()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="stair-lab/reeval_airbench_question_generator"
    )
    args = parser.parse_args()

    plot_dir = "../plot/sft_new"
    os.makedirs(plot_dir, exist_ok=True)
    restart = 64

    train_dataset = load_dataset("stair-lab/airbench-ppo", split="train")
    test_dataset = load_dataset("stair-lab/airbench-ppo", split="test")
    train_prompts = train_dataset["text"][:5]
    test_prompts = test_dataset["text"][:5]

    train_gt_zs = [extract_score(p) for p in train_prompts]
    test_gt_zs = [extract_score(p) for p in test_prompts]

    generation_config = GenerationConfig.from_pretrained(args.model)
    sampling_params = SamplingParams(
        n=restart,
        best_of=restart,
        temperature=1, # generation_config.temperature,
        top_p=generation_config.top_p,
        max_tokens=128,
        stop_token_ids=[128001, 128009],
    )
    llm = LLM(
        model=args.model,
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=0.9,
        dtype=torch.float16,
    )

    train_outputs = llm.generate(train_prompts, sampling_params)
    # >>> batch_size
    test_outputs = llm.generate(test_prompts, sampling_params)
    # >>> batch_size
    
    train_answers = [sample.text for o in train_outputs for sample in o.outputs]
    test_answers = [sample.text for o in test_outputs for sample in o.outputs]
    train_answer_df = pd.DataFrame(train_answers, columns=["text"])
    test_answer_df = pd.DataFrame(test_answers, columns=["text"])
    train_answer_dataset = Dataset.from_pandas(train_answer_df)
    test_answer_dataset = Dataset.from_pandas(test_answer_df)

    del llm
    torch.cuda.empty_cache()

    embdr = Embedder()
    embdr.load("meta-llama/Meta-Llama-3-8B", 
               dtype=torch.float16,
               tensor_parallel_size=torch.cuda.device_count(),
               )

    with open("../data/plugin_regression/airbench/bayridge.pkl", "rb") as f:
        reward_model = pickle.load(f)

    train_diffs = call_diff(train_answer_dataset, train_gt_zs, reward_model, restart)
    test_diffs = call_diff(test_answer_dataset, test_gt_zs, reward_model, restart)
    breakpoint()
    plot_hist(
        data=train_diffs,
        plot_path=f"{plot_dir}/sft_diff_hist_train_20epoch.png",
        ylabel=r"train $z$ difference",
    )

    plot_hist(
        data=test_diffs,
        plot_path=f"{plot_dir}/sft_diff_hist_test_20epoch.png",
        ylabel=r"test $z$ difference",
    )
