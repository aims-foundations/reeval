import argparse
import os
import pickle

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from embed_text_package.embed_text import Embedder
from peft import AutoPeftModelForCausalLM
from ppo_reward_model import extract_score
from utils import MLP, plot_hist
from vllm import LLM, SamplingParams


def call_diff(ds, gt_zs, reward_model, restart):
    dataloader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False)
    emb = embdr.get_embeddings(dataloader, "meta-llama/Meta-Llama-3-8B", ["text"])
    embs = emb["text"]

    pred_zs = reward_model.predict(embs).tolist()
    pred_zs = np.array(pred_zs).reshape(-1, restart)
    # >>> batch_size * restart

    gt_zs = np.array([gt_zs for _ in range(restart)]).T

    best_diffs = np.abs(pred_zs - gt_zs).min(axis=-1)
    return best_diffs


if __name__ == "__main__":
    plot_dir = "../plot/sft"
    os.makedirs(plot_dir, exist_ok=True)
    restart = 64

    model_dir = "../data/sft/lora_10epoch"
    model = AutoPeftModelForCausalLM.from_pretrained(f"{model_dir}/checkpoint-2400")
    train_dataset = load_dataset("stair-lab/airbench-sft", split="train")
    test_dataset = load_dataset("stair-lab/airbench-sft", split="test")
    train_prompts = train_dataset["text"]
    test_prompts = test_dataset["text"]
    model = model.merge_and_unload().to(torch.bfloat16)
    model.save_pretrained(model_dir)

    train_gt_zs = [extract_score(p) for p in train_prompts]
    test_gt_zs = [extract_score(p) for p in test_prompts]

    sampling_params = SamplingParams(
        temperature=0.6, n=restart, best_of=2 * restart, top_p=0.9, max_tokens=256
    )
    llm = LLM(model=model_dir)
    train_outputs = llm.generate(train_prompts, sampling_params)
    test_outputs = llm.generate(test_prompts, sampling_params)

    train_answers = [sample.text for o in train_outputs for sample in o.outputs]
    test_answers = [sample.text for o in test_outputs for sample in o.outputs]
    train_answer_df = pd.DataFrame(train_answers, columns=["text"])
    test_answer_df = pd.DataFrame(test_answers, columns=["text"])
    train_answer_dataset = Dataset.from_pandas(train_answer_df)
    test_answer_dataset = Dataset.from_pandas(test_answer_df)

    del llm
    torch.cuda.empty_cache()

    embdr = Embedder()
    embdr.load("meta-llama/Meta-Llama-3-8B")

    with open("../data/plugin_regression/airbench/bayridge.pkl", "rb") as f:
        reward_model = pickle.load(f)

    train_diffs = call_diff(train_answer_dataset, train_gt_zs, reward_model, restart)
    test_diffs = call_diff(test_answer_dataset, test_gt_zs, reward_model, restart)

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
