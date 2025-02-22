import os
from tqdm import tqdm
import pandas as pd
import argparse
import torch
import numpy as np
from torchmetrics.functional import spearman_corrcoef
from torch.distributions import Bernoulli
from huggingface_hub import HfApi, snapshot_download

scenario2metric = {
    # "thai_exam": "exact_match",
    "mmlu": "exact_match",
    "babi_qa": "quasi_exact_match",
    "bbq": "quasi_exact_match",
    "blimp": "exact_match",
    # "bold": "toxic_frac",
    "boolq": "quasi_exact_match",
    "civil_comments": "quasi_exact_match",
    # "code": "test_avg",
    "commonsense": "exact_match",
    # "copyright": "edit_distance",
    # "disinfo": "self_bleu",
    "dyck_language_np=3": "exact_match_indicator",
    "entity_data_imputation": "quasi_exact_match",
    "entity_matching": "quasi_exact_match",
    "gsm": "exact_match_indicator",
    # "ice": "logprob",
    "imdb": "quasi_exact_match",
    "legal_support": "quasi_exact_match",
    "lsat_qa": "exact_match",
    # "math": ["math_equiv", "math_equiv_chain_of_thought"],
    # "mmlu": "exact_match",
    # "msmarco": ["RR@10", "NDCG@10"],
    # "narrative_qa": "f1_score",
    # "natural_qa": "f1_score",
    # "quac": "f1_score",
    # "raft": "quasi_exact_match",
    # "real_toxicity_prompts": "toxic_frac",
    # "summarization_cnndm": "rouge_l",
    # "summarization_xsum": "rouge_l",
    # "synthetic_efficiency": "inference_runtime",
    "synthetic_reasoning": "quasi_exact_match",
    # "synthetic_reasoning_natural": "f1_set_match",
    # "the_pile": "logprob",
    "truthful_qa": "exact_match",
    # "twitter_aae": "logprob",
    "wikifact": "quasi_exact_match",
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, required=True)
    parser.add_argument("--em_sample", type=int, default=500)
    args = parser.parse_args()
    
    torch.manual_seed(0)
    
    data_folder = snapshot_download(repo_id="stair-lab/reeval_another_repo", repo_type="dataset")
    data = pd.read_parquet(f"{data_folder}/{args.scenario}/data.parquet", engine="fastparquet")
    metric = scenario2metric[args.scenario]
    # create a pivot table with model_id as index, instance_id as columns, and exact_match as values
    raw_matrix = data.pivot(index="model_id", columns="instance_id", values=metric)
    # drop the columns with only (0 and nan) or (1 and nan)
    invalid_instance_ids = raw_matrix.columns[raw_matrix.isin([0, np.nan]).all() | raw_matrix.isin([1, np.nan]).all()].values
    matrix = raw_matrix.drop(columns=invalid_instance_ids)
    matrix = torch.tensor(matrix.values).float()
    print(f"scenario: {args.scenario}, before drop: {raw_matrix.shape}, after drop: {matrix.shape}")
    save_dir = f"../data/trad_calibrate/{args.scenario}"
    os.makedirs(save_dir, exist_ok=True)
    torch.save(matrix, f"{save_dir}/matrix.pt")
    
    n_test_takers, n_items = matrix.shape
    # TODO: a better way is turn 80% of non-missing data into nan
    train_mask = torch.bernoulli(torch.full((n_test_takers, n_items), 0.8)).bool()
    train_matrix = matrix.clone()
    train_matrix[~train_mask] = torch.nan
    assert not torch.any(torch.all(torch.isnan(train_matrix), dim=1)) and not torch.any(torch.all(torch.isnan(train_matrix), dim=0))
    torch.save(train_mask, f"{save_dir}/train_mask.pt")

    # fit z
    def closure():
        optim.zero_grad()
        probs = torch.sigmoid(thetas[:, :, None] + z[None, None, :])
        mask = ~torch.isnan(train_matrix)[None, :, :].repeat(args.em_sample, 1, 1)
        repeat_train_matrix = train_matrix[None, :, :].repeat(args.em_sample, 1, 1)
        loss = -Bernoulli(probs=probs[mask]).log_prob(repeat_train_matrix[mask]).mean()
        loss.backward()
        return loss
    z = torch.zeros(n_items, requires_grad=True)
    optim = torch.optim.LBFGS([z], lr=0.1, max_iter=20, history_size=10, line_search_fn="strong_wolfe")
    thetas = torch.randn(args.em_sample, n_test_takers)

    pbar = tqdm(range(100))
    for iteration in pbar:
        if iteration > 0:
            previous_z = z.clone()
            previous_loss = loss.clone()
        
        loss = optim.step(closure)
        
        if iteration > 0:
            d_loss = previous_loss - loss
            d_z = torch.norm(previous_z - z, p=2)
            grad_norm = torch.norm(optim.param_groups[0]["params"][0].grad, p=2)
            pbar.set_postfix({"grad_norm": grad_norm, "d_z": d_z, "d_loss": d_loss})
            if d_loss < 1e-5 and d_z < 1e-5 and grad_norm < 1e-5:
                break
    
    torch.save(z, f"{save_dir}/z.pt")
    print(f"z spearman with CTT: {spearman_corrcoef(z, matrix.nanmean(0))}")
    
    # fit theta
    def closure():
        optim.zero_grad()
        probs = torch.sigmoid(theta[:, None] + z[None, :])
        mask = ~torch.isnan(train_matrix)
        loss = -Bernoulli(probs=probs[mask]).log_prob(train_matrix[mask]).mean()
        loss.backward()
        return loss
    theta = torch.zeros(n_test_takers, requires_grad=True)
    optim = torch.optim.LBFGS([theta], lr=0.1, max_iter=20, history_size=10, line_search_fn="strong_wolfe")
    
    pbar = tqdm(range(100))
    for iteration in pbar:
        if iteration > 0:
            previous_theta = theta.clone()
            previous_loss = loss.clone()
        
        loss = optim.step(closure)
        
        if iteration > 0:
            d_loss = previous_loss - loss
            d_theta = torch.norm(previous_theta - theta, p=2)
            grad_norm = torch.norm(optim.param_groups[0]["params"][0].grad, p=2)
            pbar.set_postfix({"grad_norm": grad_norm, "d_theta": d_theta, "d_loss": d_loss})
            if d_loss < 1e-5 and d_theta < 1e-5 and grad_norm < 1e-5:
                break
            
    torch.save(theta, f"{save_dir}/theta.pt")
    print(f"theta spearman with CTT: {spearman_corrcoef(theta, matrix.nanmean(1))}")