import os
import pickle
import pandas as pd
import torch
from vllm import LLM, SamplingParams
from peft import AutoPeftModelForCausalLM
from ppo_reward_model import extract_score
from datasets import Dataset, load_dataset
from utils import get_embed, plot_hist

if __name__ == "__main__":
    plot_dir = "../plot/ppo"
    os.makedirs(plot_dir, exist_ok=True)
    
    model_dir = "../data/ppo/llama3-ppo"
    model = AutoPeftModelForCausalLM.from_pretrained(f'{model_dir}/checkpoint-478')
    model = model.merge_and_unload().to(torch.bfloat16)
    model.save_pretrained(model_dir)

    test_dataset = load_dataset("stair-lab/airbench-ppo", split="test")
    prompts = test_dataset['text']
    gt_zs = [extract_score(p) for p in prompts]
    
    sampling_params = SamplingParams(temperature=0.6, top_p=0.9, max_tokens=256)
    llm = LLM(model=model_dir)
    outputs = llm.generate(prompts, sampling_params)
    answers = [o.outputs[0].text for o in outputs]
    answer_df = pd.DataFrame(answers, columns=["text"])
    answer_dataset = Dataset.from_pandas(answer_df)
    
    del llm
    torch.cuda.empty_cache()
    
    answer_embs = get_embed(answer_dataset)
    
    with open('../data/plugin_regression/airbench/bayridge.pkl', 'rb') as f:
        reward_model = pickle.load(f)
    pred_zs = reward_model.predict(answer_embs).tolist()
    
    diffs = [abs(a - b) for a, b in zip(pred_zs, gt_zs)]
    
    plot_hist(
        data=diffs,
        plot_path=f"{plot_dir}/ppo_diff_hist.png",
        ylabel=r"$z$ difference",
    )