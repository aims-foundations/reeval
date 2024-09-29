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

def mlp_predict(model, emb_input):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    emb_input = torch.tensor(emb_input, dtype=torch.float32).to(device)
    with torch.no_grad():
        output = model(emb_input)
    return output.cpu().numpy().flatten()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm', type=str, required=True)
    args = parser.parse_args()
    
    plot_dir = "../plot/sft"
    os.makedirs(plot_dir, exist_ok=True)
    
    model_dir = "../data/sft/agg_llama_10epoch"
    model = AutoPeftModelForCausalLM.from_pretrained(f'{model_dir}/checkpoint-12500')
    train_dataset = load_dataset("stair-lab/aggregate-sft", split="train")
    test_dataset = load_dataset("stair-lab/aggregate-sft", split="test")
    train_prompts = train_dataset['text'][:3000]
    test_prompts = test_dataset['text'][:3000]
    model = model.merge_and_unload().to(torch.bfloat16)
    model.save_pretrained(model_dir)
    
    train_gt_zs = [extract_score(p) for p in train_prompts]
    test_gt_zs = [extract_score(p) for p in test_prompts]
    
    sampling_params = SamplingParams(temperature=0.6, top_p=0.9, max_tokens=256)
    llm = LLM(model=model_dir)
    train_outputs = llm.generate(train_prompts, sampling_params)
    test_outputs = llm.generate(test_prompts, sampling_params)
    train_answers = [o.outputs[0].text for o in train_outputs]
    test_answers = [o.outputs[0].text for o in test_outputs]
    train_answer_df = pd.DataFrame(train_answers, columns=["text"])
    test_answer_df = pd.DataFrame(test_answers, columns=["text"])
    train_answer_dataset = Dataset.from_pandas(train_answer_df)
    test_answer_dataset = Dataset.from_pandas(test_answer_df)
    
    del llm
    torch.cuda.empty_cache()
    
    answer_train_embs = get_embed(train_answer_dataset)
    answer_test_embs = get_embed(test_answer_dataset)
    
    with open(f'../data/plugin_regression/aggregate/mlp.pkl', 'rb') as f:
        model = pickle.load(f)
    train_pred_zs = mlp_predict(model, answer_train_embs).tolist()
    test_pred_zs = mlp_predict(model, answer_test_embs).tolist()
    
    train_diffs = [abs(a - b) for a, b in zip(train_pred_zs, train_gt_zs)]
    test_diffs = [abs(a - b) for a, b in zip(test_pred_zs, test_gt_zs)]
    
    plot_hist(
        data=train_diffs,
        plot_path=f"{plot_dir}/sft_diff_hist_train_agg.png",
        ylabel=r"train $z$ difference",
    )
    
    plot_hist(
        data=test_diffs,
        plot_path=f"{plot_dir}/sft_diff_hist_test_agg.png",
        ylabel=r"test $z$ difference",
    )
