import argparse
import io
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import torch
from amortized_irt import IRT
from datasets import load_dataset
from huggingface_hub import HfApi, snapshot_download
from ppo_reward_model import extract_score
from torchmetrics import SpearmanCorrCoef
from tqdm import tqdm
from tueplots import bundles

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
        prob_matrix = IRT.compute_prob(
            ability, difficulties, disciminatory=1, guessing=0, loading_factor=1
        )
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


def save_matrix(dataset, generator_name, response_matrix, hf_models, data_folder):
    upload_api = HfApi()
    matrix_file = io.BytesIO()
    torch.save(response_matrix, matrix_file)
    generator_short_name = generator_name.split("/")[-1]
    upload_api.upload_file(
        repo_id="stair-lab/reeval_response_generated_questions",
        repo_type="dataset",
        path_in_repo=f"{dataset}/{generator_short_name}_response_matrix.pt",
        path_or_fileobj=matrix_file,
    )

    model_keys = pd.read_csv(f"{data_folder}/{args.dataset}/model_keys.csv")

    # Filter row in model_keys that have huggingface_model_id in huggingface_model_id
    model_keys = model_keys[model_keys["huggingface_model_id"].isin(hf_models)]

    # Sort the model_keys based on the order of `hf_models` variable
    model_keys = (
        model_keys.set_index("huggingface_model_id").loc[hf_models].reset_index()
    )
    assert len(model_keys) == response_matrix.shape[0]

    new_model_keys = io.BytesIO()
    model_keys.to_csv(new_model_keys, index=False)
    upload_api.upload_file(
        repo_id="stair-lab/reeval_response_generated_questions",
        repo_type="dataset",
        path_in_repo=f"{dataset}/{generator_short_name}_model_keys.csv",
        path_or_fileobj=new_model_keys,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="airbench")
    parser.add_argument("--max_epochs", type=int, default=5000)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--question_generators", type=str, nargs="+", required=True)
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()

    #############
    # Calibration
    #############
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_folder = snapshot_download(
        repo_id="stair-lab/reeval_responses",
        repo_type="dataset",
    )

    result_folder = snapshot_download(
        repo_id="stair-lab/reeval_results",
        repo_type="dataset",
    )

    generated_questions_folder = snapshot_download(
        repo_id="stair-lab/reeval_generated_questions",
        repo_type="dataset",
    )

    response_result_folder = snapshot_download(
        repo_id="stair-lab/reeval_response_generated_questions",
        repo_type="dataset",
    )

    available_models = pd.read_csv("./configs/model_hf_id.csv")[
        "huggingface_model_id"
    ].values

    with open(
        f"{result_folder}/{args.dataset}/s42_mle_1pl_1d_aq_nl1/abilities.pkl", "rb"
    ) as f:
        abilities = pickle.load(f)
        abilities = torch.tensor(abilities, device=device)

    original_response_matrix = torch.load(
        f"{data_folder}/{args.dataset}/response_matrix.pt"
    ).to(device=device)
    train_model_indices = pickle.load(
        open(
            f"{result_folder}/{args.dataset}/s42_mle_1pl_1d_aq_nl1/train_model_indices.pkl",
            "rb",
        )
    ).tolist()
    test_model_indices = pickle.load(
        open(
            f"{result_folder}/{args.dataset}/s42_mle_1pl_1d_aq_nl1/test_model_indices.pkl",
            "rb",
        )
    ).tolist()

    model_keys = pd.read_csv(f"{data_folder}/{args.dataset}/model_keys.csv")
    list_model_having_ability = list(model_keys["huggingface_model_id"].values)
    # this will keep the nans

    train_model_idxs = []
    test_model_idxs = []
    train_response_matrix = []
    test_response_matrix = []
    global_model_train_idx = []
    global_model_test_idx = []

    train_model_names = []
    test_model_names = []

    for avai_model_name in available_models:
        if avai_model_name not in list_model_having_ability:
            continue

        model_raw_idx = list_model_having_ability.index(avai_model_name)
        if model_raw_idx in train_model_indices:
            model_idx = train_model_indices.index(model_raw_idx)
        elif model_raw_idx in test_model_indices:
            model_idx = test_model_indices.index(model_raw_idx)
        else:
            continue

        # Load the inference results
        model_name = avai_model_name.replace("/", "_")
        if model_name.endswith("_llama-2-7b") or model_name.endswith("_llama-2-13b"):
            model_name = model_name + "-hf"
        elif model_name.endswith("Meta-Llama-3-8B"):
            model_name = model_name + "-Instruct"

        for qgi, question_generator in enumerate(args.question_generators):
            model_short_name = question_generator.split("/")[-1]
            if model_short_name == "reeval_question_generator_sft":
                model_short_name = ""
            else:
                model_short_name = "_" + model_short_name

            if not os.path.exists(
                f"{generated_questions_folder}/sft/{args.dataset}{model_short_name}/{model_name}_with_y.csv"
            ):
                if not os.path.exists(
                    f"{generated_questions_folder}/sft/{args.dataset}{model_short_name}/{model_name}_with_y.pkl"
                ):
                    break
                else:
                    using_pickle = True
            else:
                using_pickle = False

            if using_pickle:
                answer_df = pickle.load(
                    open(
                        f"{generated_questions_folder}/sft/{args.dataset}{model_short_name}/{model_name}_with_y.pkl",
                        "rb",
                    )
                )
            else:
                answer_df = pd.read_csv(
                    f"{generated_questions_folder}/sft/{args.dataset}{model_short_name}/{model_name}_with_y.csv"
                )

            if answer_df.shape[0] != args.num_samples:
                break

            if model_raw_idx in train_model_indices:
                train_response_matrix.append(answer_df["y"].tolist())
                if model_idx not in train_model_idxs:
                    train_model_idxs.append(model_idx)
                    global_model_train_idx.append(model_raw_idx)
                    train_model_names.append(avai_model_name)

            elif model_raw_idx in test_model_indices:
                test_response_matrix.append(answer_df["y"].tolist())
                if model_idx not in test_model_idxs:
                    test_model_idxs.append(model_idx)
                    global_model_test_idx.append(model_raw_idx)
                    test_model_names.append(avai_model_name)

    n_qgenerator = len(args.question_generators)
    train_response_matrix = torch.tensor(
        train_response_matrix, device=device, dtype=torch.float32
    )
    test_response_matrix = torch.tensor(
        test_response_matrix, device=device, dtype=torch.float32
    )
    # >>> (n_models * n_qgenerator=2, n_questions=1000)

    train_response_matrix = train_response_matrix.view(
        -1, n_qgenerator, train_response_matrix.shape[-1]
    )
    train_response_matrix = train_response_matrix.permute(1, 0, 2)
    # >>> n_qgenerator=2 x n_models x n_questions=1000
    test_response_matrix = test_response_matrix.view(
        -1, n_qgenerator, test_response_matrix.shape[-1]
    )
    test_response_matrix = test_response_matrix.permute(1, 0, 2)

    # Filter out columns in response_matrix that have all values equal 0, 1, or -1
    train_response_matrices = []
    test_response_matrices = []
    masks = []
    for mi, mat in enumerate(train_response_matrix):
        # mi = 0, 1, mat = n_models x n_questions
        # train_mask = (
        #     (mat == 0).all(dim=0) | (mat == 1).all(dim=0) | (mat == -1).all(dim=0)
        # )
        train_mask = []
        for col_data in mat.T:
            if set(col_data.unique()).issubset({0, -1}) or set(col_data.unique()).issubset({1, -1}):
                train_mask.append(True)
            else:
                train_mask.append(False)
        masks.append(train_mask)

        train_response_matrices.append(mat[:, ~train_mask])
        test_response_matrices.append(test_response_matrix[mi, :, ~train_mask])
        # >>> n_models x n_questions

    num_additional_questions = [mat.shape[1] for mat in train_response_matrices]

    # Save the response matrices
    for generator, train_mat, test_mat in zip(
        args.question_generators, train_response_matrices, test_response_matrices
    ):
        joint_mat = torch.cat([train_mat, test_mat], dim=0)
        joint_names = train_model_names + test_model_names
        save_matrix(args.dataset, generator, joint_mat, joint_names, data_folder)

    train_original_response_matrix = original_response_matrix[global_model_train_idx]
    test_original_response_matrix = original_response_matrix[global_model_test_idx]
    num_original_questions = train_original_response_matrix.shape[1]
    # >>> n_models x n_original_questions

    train_response_matrix = torch.cat(
        [train_original_response_matrix, *train_response_matrices], dim=1
    )
    # >>> n_models x (n_original_questions, n_questions)

    train_abilities = abilities[train_model_idxs]
    # >>> n_models x 1

    # pred_abilities, item_parms = calibrate(train_response_matrix, device, max_epoch=args.max_epochs)
    # os.makedirs(f"../results/difficulty_validation/{args.dataset}", exist_ok=True)
    # pickle.dump(pred_abilities, open(f"../results/difficulty_validation/{args.dataset}/abilities.pkl", "wb"))
    # pickle.dump(item_parms, open(f"../results/difficulty_validation/{args.dataset}/item_parms.pkl", "wb"))

    pred_abilities = pickle.load(
        open(f"../results/difficulty_validation/{args.dataset}/abilities.pkl", "rb")
    )
    item_parms = pickle.load(
        open(f"../results/difficulty_validation/{args.dataset}/item_parms.pkl", "rb")
    )

    original_difficulties = item_parms[:, 0][:num_original_questions]
    # >>> n_original_questions

    new_difficulties = []
    for num_aqi in range(len(num_additional_questions)):
        start_idx = num_original_questions + sum(num_additional_questions[:num_aqi])
        end_idx = start_idx + num_additional_questions[num_aqi]
        new_difficulties.append(item_parms[:, 0][start_idx:end_idx])
        # >>> n_questions

    test_original_abilities = infer_abilities(
        original_difficulties, test_original_response_matrix, max_epoch=args.max_epochs
    )
    test_new_abilities = []
    for new_difficulty, test_res_mat in zip(new_difficulties, test_response_matrices):
        test_new_abilities.append(
            infer_abilities(new_difficulty, test_res_mat, max_epoch=args.max_epochs)
        )

    # Compute the MAE between the predicted and ground truth difficulties
    sm_fn = SpearmanCorrCoef()
    os.makedirs("../results/difficulty_validation", exist_ok=True)
    res_file = open(f"../results/difficulty_validation/{args.dataset}.txt", "w")

    for mi, new_difficulty in enumerate(new_difficulties):
        model_short_name = args.question_generators[mi].split("/")[-1]
        ds_model_short_name = ""
        if "mistral" in model_short_name:
            ds_model_short_name = "_Mistral-7B-Instruct-v0.3"
        else:
            ds_model_short_name = "_Meta-Llama-3.1-8B-Instruct"

        test_dataset = load_dataset(
            f"stair-lab/reeval-ppo", args.dataset+ds_model_short_name, split="train"
        )
        test_texts = test_dataset["text"][: args.num_samples]
        gt_difficulties = torch.tensor(
            [extract_score(p) for p in test_texts], device=device
        )

        mask = masks[mi]
        sp_corr_z = sm_fn(gt_difficulties[~mask], new_difficulty.flatten()).item()
        print(f"Model{mi} - Difficulty Spearman: {sp_corr_z}")
        res_file.write(f"Model{mi} - Difficulty Spearman: {sp_corr_z}\n")

        # plot the scatter plot for difficulties
        plt.scatter(
            gt_difficulties[~mask].flatten().cpu().numpy(),
            new_difficulty.flatten().cpu().numpy(),
        )
        plt.xlabel("Difficulty from Real Data")
        plt.ylabel("Difficulty from Calibration")
        # plot a trend line
        z = np.polyfit(
            gt_difficulties[~mask].flatten().cpu().numpy(),
            new_difficulty.flatten().cpu().numpy(),
            1,
        )
        p = np.poly1d(z)
        plt.plot(
            gt_difficulties[~mask].flatten().cpu().numpy(),
            p(gt_difficulties[~mask].flatten().cpu().numpy()),
            "r--",
        )
        plt.savefig(
            f"../plot/sft/{args.dataset}/irt_difficulty_{mi}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # Draw histogram
        plt.hist(
            gt_difficulties[~mask].cpu().detach().numpy(),
            bins=50,
            alpha=0.5,
            label="Target Difficulty",
        )
        plt.hist(
            new_difficulty.cpu().detach().numpy(),
            bins=50,
            alpha=0.5,
            label="Difficulty from Calibration",
        )
        plt.legend(loc="upper right")
        plt.savefig(
            f"../plot/sft/{args.dataset}/irt_difficulty_hist_{mi}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    # Compute the Spearman correlation between the predicted and ground truth abilities
    sp_corr_train = sm_fn(train_abilities.flatten(), pred_abilities.flatten()).item()
    print(f"Ability Spearman with Ability from Real Data (train set): {sp_corr_train}")
    res_file.write(
        f"Ability Spearman with Ability from Real Data (train set): {sp_corr_train}\n"
    )

    # Draw the scatter plot for abilities
    plt.scatter(
        train_abilities.flatten().cpu().numpy(), pred_abilities.flatten().cpu().numpy()
    )
    plt.xlabel("Ability from Original questions")
    plt.ylabel("Ability from Generated questions")
    # plot a trend line
    z = np.polyfit(
        train_abilities.flatten().cpu().numpy(),
        pred_abilities.flatten().cpu().numpy(),
        1,
    )
    p = np.poly1d(z)
    plt.plot(
        train_abilities.flatten().cpu().numpy(),
        train_abilities.flatten().cpu().numpy(),
        "r--",
    )
    plt.savefig(
        f"../plot/sft/{args.dataset}/irt_ability.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # Compute the Spearman correlation between the predicted and CTT score
    ctt_scores = torch.tensor(list(model_keys["ctt_score"].values), device=device)[
        global_model_train_idx
    ]
    sp_corr_train_ctt = sm_fn(pred_abilities.flatten(), ctt_scores).item()
    print(f"Ability Spearman with CTT Score (train set): {sp_corr_train_ctt}")
    res_file.write(
        f"Ability Spearman with CTT Score (train set): {sp_corr_train_ctt}\n"
    )

    for mi, test_new_ability in enumerate(test_new_abilities):
        # Compute the Spearman correlation between the predicted and ground truth abilities
        sp_corr_test = sm_fn(
            test_original_abilities.flatten(), test_new_ability.flatten()
        ).item()
        print(
            f"Model {mi} - Ability Spearman with Ability from Real Data (test set): {sp_corr_test}"
        )
        res_file.write(
            f"Model {mi} - Ability Spearman with Ability from Real Data (test set): {sp_corr_test}\n"
        )

    # Save the results
    res_file.close()
