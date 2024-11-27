import os
import io
import torch
import pickle
import numpy as np
import pandas as pd
import requests
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from huggingface_hub import snapshot_download, HfApi
from datasets import load_dataset
from tueplots import bundles
from torchmetrics import SpearmanCorrCoef
from ppo_reward_model import extract_score
from utils.irt import IRT    
    
plt.rcParams.update(bundles.iclr2024())

def calibrate(response_matrix, device, max_epoch=5000):
    n_models, n_questions = response_matrix.shape
    print("Total number of models: ", n_models)
    print("Total number of questions: ", n_questions)
    irt_model = IRT(
        n_questions=n_questions,
        n_testtaker=n_models,
        D=1,
        PL=1,
        amortize_item=False,
        amortize_student=False,
        amortized_question_hyperparams={},
        amortized_model_hyperparams={},
        device=device,
        report_to=None,
    )
    irt_model.fit(
        max_epoch=max_epoch,
        response_matrix=response_matrix,
        method="em",
        embedding=None,
        model_features=None,
    )
    
    # Save results
    pred_abilities = irt_model.get_abilities().detach()
    item_parms = irt_model.get_item_parameters().detach()
    return pred_abilities, item_parms


def infer_abilities(difficulties, response_matrix, max_epoch=3000):
    n_testtaker = response_matrix.shape[0]
    ability = torch.randn(n_testtaker, 1, device=device)
    ability.requires_grad = True
    
    optimizer = torch.optim.Adam([ability], lr=0.01)

    pbar = tqdm(range(max_epoch))
    mask = response_matrix != -1
    masked_response_matrix = response_matrix[mask]

    for _ in pbar:
        prob_matrix = IRT.compute_prob(ability, difficulties, disciminatory=1, guessing=0, loading_factor=1)
        masked_prob_matrix = prob_matrix[mask]

        berns = torch.distributions.Bernoulli(probs=masked_prob_matrix)
        loss = -berns.log_prob(masked_response_matrix).mean()

        # encourage the ability to have mean 0 and std 1            
        mean_ability = torch.mean(abilities, dim=0)
        std_ability = torch.std(abilities, dim=0)
        loss = loss + torch.abs(mean_ability).mean() + torch.abs(std_ability - 1).mean()
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        pbar.set_postfix({"loss_ability": loss.item()})
    
    return ability.detach()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="airbench")
    parser.add_argument("--max_epochs", type=int, default=5000)
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()
    
    #############
    # Calibration
    #############
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    data_folder = snapshot_download(
        repo_id="stair-lab/reeval_responses", repo_type="dataset", local_files_only=True
    )

    result_folder = snapshot_download(
        repo_id="stair-lab/reeval_results", repo_type="dataset", local_files_only=True
    )
    
    generated_questions_folder = snapshot_download(
        repo_id="stair-lab/reeval_generated_questions", repo_type="dataset", local_files_only=True
    )
    
    test_question_df = pd.read_csv(
        f"{generated_questions_folder}/sft/{args.dataset}/train_answers_filtered.csv"
    )
    
    test_dataset = load_dataset(f"stair-lab/{args.dataset}-ppo", split="test")
    test_texts = test_dataset["text"][: len(test_question_df)]
    gt_difficulties = torch.tensor([extract_score(p) for p in test_texts], device=device)
    available_models = pd.read_csv("./configs/model_hf_id.csv")["huggingface_model_id"].values
    
    with open(f"{result_folder}/{args.dataset}/s42_mle_1pl_1d_aq_nl1/abilities.pkl", "rb") as f:
        abilities = pickle.load(f)
        abilities = torch.tensor(abilities, device=device)

    original_response_matrix = torch.load(f"{data_folder}/{args.dataset}/response_matrix.pt").to(
        device=device
    )
    train_model_indices = pickle.load(
        open(f"{result_folder}/{args.dataset}/s42_mle_1pl_1d_aq_nl1/train_model_indices.pkl", "rb")
    ).tolist()
    test_model_indices = pickle.load(
        open(f"{result_folder}/{args.dataset}/s42_mle_1pl_1d_aq_nl1/test_model_indices.pkl", "rb")
    ).tolist()
    
    model_keys = pd.read_csv(f"{data_folder}/{args.dataset}/model_keys.csv")
    list_model_having_ability = list(model_keys["huggingface_model_id"].values)
    
    train_model_idxs = []
    test_model_idxs = []
    train_response_matrix = []
    test_response_matrix = []
    global_model_train_idx = []
    global_model_test_idx = []
    
    for model_name in available_models:
        if model_name not in list_model_having_ability:
            continue
        
        model_raw_idx = list_model_having_ability.index(model_name)
        if model_raw_idx in train_model_indices:
            model_idx = train_model_indices.index(model_raw_idx)
        elif model_raw_idx in test_model_indices:
            model_idx = test_model_indices.index(model_raw_idx)
        else:
            continue
        
        # Load the inference results
        model_name = model_name.replace("/", "_")
        if model_name.endswith("llama-2-7b") or model_name.endswith("llama-2-13b"):
            model_name = model_name + "-hf"
        elif model_name.endswith("Meta-Llama-3-8B"):
            model_name = model_name + "-Instruct"
        if not os.path.exists(f"{generated_questions_folder}/sft/{args.dataset}/{model_name}_with_y.csv"):
            continue
        answer_df = pd.read_csv(f"{generated_questions_folder}/sft/{args.dataset}/{model_name}_with_y.csv")
        if answer_df.shape[0] != len(test_question_df):
            continue
        
        if model_raw_idx in train_model_indices:
            train_response_matrix.append(answer_df["y"].tolist())
            train_model_idxs.append(model_idx)
            global_model_train_idx.append(model_raw_idx)

        elif model_raw_idx in test_model_indices:
            test_response_matrix.append(answer_df["y"].tolist())
            test_model_idxs.append(model_idx)
            global_model_test_idx.append(model_raw_idx)
        
    train_response_matrix = torch.tensor(train_response_matrix, device=device, dtype=torch.float32)
    test_response_matrix = torch.tensor(test_response_matrix, device=device, dtype=torch.float32)
    
    # Filter out columns in response_matrix that have all values equal 0, 1, or -1
    train_mask = (train_response_matrix == 0).all(dim=0) | (train_response_matrix == 1).all(dim=0) | (train_response_matrix == -1).all(dim=0)
    train_response_matrix = train_response_matrix[:, ~train_mask]
    test_response_matrix = test_response_matrix[:, ~train_mask]
    # >>> n_models x n_questions
    
    train_original_response_matrix = original_response_matrix[global_model_train_idx]
    test_original_response_matrix = original_response_matrix[global_model_test_idx]
    num_original_questions = train_original_response_matrix.shape[1]
    # >>> n_models x n_original_questions
    
    train_response_matrix = torch.cat([train_original_response_matrix, train_response_matrix], dim=1)
    # >>> n_models x (n_original_questions, n_questions)
    
    train_abilities = abilities[train_model_idxs]
    # >>> n_models x 1
    
    # pred_abilities, item_parms = calibrate(train_response_matrix, device, max_epoch=args.max_epochs)
    # os.makedirs(f"../results/difficulty_validation/{args.dataset}", exist_ok=True)
    # pickle.dump(pred_abilities, open(f"../results/difficulty_validation/{args.dataset}/abilities.pkl", "wb"))
    # pickle.dump(item_parms, open(f"../results/difficulty_validation/{args.dataset}/item_parms.pkl", "wb"))
    
    pred_abilities = pickle.load(open(f"../results/difficulty_validation/{args.dataset}/abilities.pkl", "rb"))
    item_parms = pickle.load(open(f"../results/difficulty_validation/{args.dataset}/item_parms.pkl", "rb"))
    
    original_difficulties = item_parms[:, 0][:num_original_questions]
    # >>> n_original_questions
    
    new_difficulties = item_parms[:, 0][num_original_questions:]
    # >>> n_questions

    test_original_abilities = infer_abilities(original_difficulties, test_original_response_matrix, max_epoch=args.max_epochs)
    test_new_abilities = infer_abilities(new_difficulties, test_response_matrix, max_epoch=args.max_epochs)
    
    # Compute the MAE between the predicted and ground truth difficulties
    sm_fn = SpearmanCorrCoef()
    sp_corr_z = sm_fn(gt_difficulties[~train_mask].flatten(), new_difficulties.flatten()).item()
    print(f"Difficulty Correlation: {sp_corr_z}")
    
    # plot the scatter plot for difficulties 
    plt.scatter(gt_difficulties[~train_mask].flatten().cpu().numpy(), new_difficulties.flatten().cpu().numpy())
    plt.xlabel("Difficulty from Real Data")
    plt.ylabel("Difficulty from Calibration")
    # plot a trend line
    z = np.polyfit(gt_difficulties[~train_mask].flatten().cpu().numpy(), new_difficulties.flatten().cpu().numpy(), 1)
    p = np.poly1d(z)
    plt.plot(gt_difficulties[~train_mask].flatten().cpu().numpy(), p(gt_difficulties[~train_mask].flatten().cpu().numpy()), "r--")
    plt.savefig(f"../plot/sft/{args.dataset}/irt_difficulty.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # Compute the Spearman correlation between the predicted and ground truth abilities
    sp_corr_train = sm_fn(train_abilities.flatten(), pred_abilities.flatten()).item()
    print(f"Ability Spearman with Ability from Real Data (train set): {sp_corr_train}")
    
    # Compute the Spearman correlation between the predicted and CTT score
    ctt_scores = torch.tensor(list(model_keys["ctt_score"].values), device=device)[global_model_train_idx]
    sp_corr_train_ctt = sm_fn(pred_abilities.flatten(), ctt_scores).item()
    print(f"Ability Spearman with CTT Score (train set): {sp_corr_train_ctt}")
    
    # Compute the Spearman correlation between the predicted and ground truth abilities
    sp_corr_test = sm_fn(test_original_abilities.flatten(), test_new_abilities.flatten()).item()
    print(f"Ability Spearman with Ability from Real Data (test set): {sp_corr_test}")

    # Save the results
    os.makedirs("../results/difficulty_validation", exist_ok=True)
    with open(f"../results/difficulty_validation/{args.dataset}.txt", "w") as f:
        f.write(f"Difficulty Correlation: {sp_corr_z}\n")
        f.write(f"Ability Spearman (train): {sp_corr_train}\n")
        f.write(f"Ability Spearman with CTT Score (train): {sp_corr_train_ctt}")
        f.write(f"Ability Spearman (test): {sp_corr_test}")
    
    # Draw histogram
    plt.hist(gt_difficulties[~train_mask].cpu().detach().numpy(), bins=50, alpha=0.5, label="Target Difficulty")
    plt.hist(new_difficulties.cpu().detach().numpy(), bins=50, alpha=0.5, label="Difficulty from Calibration")
    plt.legend(loc="upper right")
    plt.savefig(f"../plot/sft/{args.dataset}/irt_difficulty_hist.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # Draw the scatter plot for abilities
    plt.scatter(train_abilities.flatten().cpu().numpy(), pred_abilities.flatten().cpu().numpy())
    plt.xlabel("Ability from Original questions")
    plt.ylabel("Ability from Generated questions")
    # plot a trend line
    z = np.polyfit(train_abilities.flatten().cpu().numpy(), pred_abilities.flatten().cpu().numpy(), 1)
    p = np.poly1d(z)
    plt.plot(train_abilities.flatten().cpu().numpy(), train_abilities.flatten().cpu().numpy(), "r--")
    plt.savefig(f"../plot/sft/{args.dataset}/irt_ability.png", dpi=300, bbox_inches="tight")
    plt.close()
