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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm', type=str, required=True)
    args = parser.parse_args()
    
    plot_dir = "../plot/sft"
    os.makedirs(plot_dir, exist_ok=True)
    
    if args.llm == "llama":
        model_dir = "../data/sft/lora_10epoch"
        model = AutoPeftModelForCausalLM.from_pretrained(f'{model_dir}/checkpoint-2400')
        train_dataset = load_dataset("stair-lab/airbench-sft", split="train")
        test_dataset = load_dataset("stair-lab/airbench-sft", split="test")
        train_prompts = train_dataset['text']
        test_prompts = test_dataset['text']
    model = model.merge_and_unload().to(torch.bfloat16)
    model.save_pretrained(model_dir)
    
    train_gt_zs = [extract_score(p) for p in train_prompts]
    test_gt_zs = [extract_score(p) for p in test_prompts]
    
    sampling_params = SamplingParams(temperature=0.6, n=64, best_of_n=128, top_p=0.9, max_tokens=256)
    llm = LLM(model=model_dir)
    train_outputs = llm.generate(train_prompts, sampling_params)
    test_outputs = llm.generate(test_prompts, sampling_params)
    # batch x n_samples
    
    train_answers = [sample.text for o in train_outputs for sample in o.outputs]
    test_answers = [sample.text for o in test_outputs for sample in o.outputs]
    train_answer_df = pd.DataFrame(train_answers, columns=["text"])
    test_answer_df = pd.DataFrame(test_answers, columns=["text"])
    train_answer_dataset = Dataset.from_pandas(train_answer_df)
    test_answer_dataset = Dataset.from_pandas(test_answer_df)
    
    del llm
    torch.cuda.empty_cache()
    
    # Perfrom batch inference
    def call_diff(ds, gt_zs, reward_model):
        output_diffs = []
        ds_loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False)
        for batch in ds_loader:
            embs = get_embed(batch)
            pred_zs = reward_model.predict(embs).tolist()
            diffs = [abs(a - b) for a, b in zip(pred_zs, gt_zs)]
            output_diffs.extend(diffs)
        return diffs
    
    with open('../data/plugin_regression/airbench/bayridge.pkl', 'rb') as f:
        reward_model = pickle.load(f)
        
    train_diffs = call_diff(train_answer_dataset, train_gt_zs, reward_model)
    test_diffs = call_diff(test_answer_dataset, test_gt_zs, reward_model)
    
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
