import argparse
import os
import pickle

import pandas as pd
import torch
from tqdm import tqdm
from utils.constants import DATASETS
from utils.utils import (
    accuracy_plot,
    error_bar_plot_single,
    goodness_of_fit,
    goodness_of_fit_plot,
    str2bool,
    theta_corr_plot,
)
from huggingface_hub import snapshot_download

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--D", type=int, default=1)
    parser.add_argument("--PL", type=int, default=1)
    parser.add_argument(
        "--fitting_method", type=str, default="mle", choices=["mle", "mcmc", "em"]
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_epoch", type=int, default=5000)
    parser.add_argument("--amortized_question", type=str2bool, default=False)
    parser.add_argument("--amortized_student", type=str2bool, default=False)
    args = parser.parse_args()
    args = parser.parse_args()

    plot_dir = f"../../plot/{args.fitting_method}_{args.PL}pl{'_amortized' if args.amortized else ''}_calibration"
    os.makedirs(plot_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gof_means, gof_stds = [], []
    corr_ctt_means, corr_ctt_stds = [], []
    corr_helm_means, corr_helm_stds = [], []
    plugin_gof_train_means, plugin_gof_test_means = [], []
    amor_gof_train_means, amor_gof_test_means = [], []

    for dataset in tqdm(DATASETS):
        print(f"Processing {dataset}")
        data_folder = snapshot_download(
            repo_id="stair-lab/reeval_responses", repo_type="dataset"
        )

        response_matrix = torch.load(f"{data_folder}/{args.dataset}/response_matrix.pt").to(
            device=device, dtype=torch.float32
        )

        result_folder = snapshot_download(
            repo_id="stair-lab/reeval_results", repo_type="dataset",
        )

        # load the train/test indices for question and student
        train_question_indices = pickle.load(
            open(f"{result_folder}/{dataset}/train_indices.pkl", "rb")
        )
        test_question_indices = pickle.load(
            open(f"{result_folder}/{dataset}/test_indices.pkl", "rb")
        )
        train_student_indices = pickle.load(
            open(f"{result_folder}/{dataset}/train_indices.pkl", "rb")
        )
        test_student_indices = pickle.load(
            open(f"{result_folder}/{dataset}/test_indices.pkl", "rb")
        )

        if args.amortized_question:
            item_parameters_nn = pickle.load(
                open(f"{result_folder}/{dataset}/item_parameters_nn.pkl", "rb")
            )
            
            item_embeddings = torch.load(
                f"{data_folder}/{args.dataset}/item_embeddings.pt",
            ).to(device=device)

            item_embeddings_train = item_embeddings[train_question_indices]
            item_embeddings_test = item_embeddings[test_question_indices]
            
            item_parms_train = item_parameters_nn(item_embeddings_train)
            item_parms_test = item_parameters_nn(item_embeddings_test)
            
            if args.PL == 1:
                
            
        else:
            item_parms = pickle.load(
                open(f"{result_folder}/{dataset}/item_parms.pkl", "rb")
            )            
            item_parms = torch.tensor(item_parms, device=device)
            
        if args.amortized_student:
            student_parameters_nn = pickle.load(
                open(f"{result_folder}/{dataset}/student_parameters_nn.pkl", "rb")
            )

            model_keys = pd.read_csv(f"{data_folder}/{args.dataset}/model_keys.csv")
            model_features = model_keys["flop"].tolist()
            model_features = torch.tensor(
                model_features, dtype=torch.float32, device=device
            )
            model_features = torch.log(model_features)
            model_features = torch.stack(
                [model_features, torch.ones_like(model_features)], dim=1
            )

            # Fill nan with -1
            model_features[torch.isnan(model_features)] = -1
            
            student_embeddings_train = model_features[train_student_indices]
            student_embeddings_test = model_features[test_student_indices]
        else:
            abilities = pickle.load(
                open(f"{result_folder}/{dataset}/abilities.pkl", "rb")
            )
            abilities = torch.tensor(abilities, device=device)

        # metric 1: GOF
        gof_mean, gof_std = goodness_of_fit_plot(
            z=item_parms,
            theta=abilities,
            y=response_matrix,
            plot_path=f"{plot_dir}/goodness_of_fit_{dataset}",
        )
        gof_means.append(gof_mean)
        gof_stds.append(gof_std)

        # metric 2: correlation with CTT
        corr_ctt_mean, corr_ctt_std = theta_corr_plot(
            mode="ctt",
            theta=abilities,
            y=response_matrix,
            plot_path=f"{plot_dir}/theta_corr_ctt_{dataset}",
        )
        corr_ctt_means.append(corr_ctt_mean)
        corr_ctt_stds.append(corr_ctt_std)

        # metric 3: correlation with HELM
        corr_helm_mean, corr_helm_std = theta_corr_plot(
            mode="helm",
            data_folder="../../data",
            theta=abilities,
            dataset=dataset,
            plot_path=f"{plot_dir}/theta_corr_helm_{dataset}",
        )
        corr_helm_means.append(corr_helm_mean)
        corr_helm_stds.append(corr_helm_std)

        # metric 4: Accuracy
        acc_mean, acc_std = accuracy_plot(
            item_parms=item_parms,
            theta=abilities,
            y=response_matrix,
            plot_path=f"{plot_dir}/accuracy_{dataset}",
        )

    #     plugin_train_indices = pd.read_csv(
    #         f"../../data/plugin_regression/{dataset}/train_0.csv"
    #     )["index"].values
    #     plugin_test_indices = pd.read_csv(
    #         f"../../data/plugin_regression/{dataset}/test_0.csv"
    #     )["index"].values

    #     plugin_gof_train_mean, _ = goodness_of_fit(
    #         z=torch.tensor(item_parms[plugin_train_indices], dtype=torch.float32),
    #         theta=torch.tensor(abilities, dtype=torch.float32),
    #         y=torch.tensor(y[:, plugin_train_indices], dtype=torch.float32),
    #     )
    #     plugin_gof_train_means.append(plugin_gof_train_mean)

    #     plugin_gof_test_mean, _ = goodness_of_fit(
    #         z=torch.tensor(item_parms[plugin_test_indices], dtype=torch.float32),
    #         theta=torch.tensor(abilities, dtype=torch.float32),
    #         y=torch.tensor(y[:, plugin_test_indices], dtype=torch.float32),
    #     )
    #     plugin_gof_test_means.append(plugin_gof_test_mean)

    #     amor_train_indices = pd.read_csv(
    #         f"../../data/amor_calibration/{dataset}/z_train_0.csv"
    #     )["index"].values
    #     amor_test_indices = pd.read_csv(
    #         f"../../data/amor_calibration/{dataset}/z_test_0.csv"
    #     )["index"].values

    #     amor_gof_train_mean, _ = goodness_of_fit(
    #         z=torch.tensor(item_parms[amor_train_indices], dtype=torch.float32),
    #         theta=torch.tensor(abilities, dtype=torch.float32),
    #         y=torch.tensor(y[:, amor_train_indices], dtype=torch.float32),
    #     )
    #     amor_gof_train_means.append(amor_gof_train_mean)

    #     amor_gof_test_mean, _ = goodness_of_fit(
    #         z=torch.tensor(item_parms[amor_test_indices], dtype=torch.float32),
    #         theta=torch.tensor(abilities, dtype=torch.float32),
    #         y=torch.tensor(y[:, amor_test_indices], dtype=torch.float32),
    #     )
    #     amor_gof_test_means.append(amor_gof_test_mean)

    # plugin_gof_df_train = pd.DataFrame(
    #     {
    #         "datasets": DATASETS,
    #         "gof_means": plugin_gof_train_means,
    #     }
    # )
    # plugin_gof_df_train.to_csv(f"{plot_dir}/nonamor4plugin_gof_train.csv", index=False)

    # plugin_gof_df_test = pd.DataFrame(
    #     {
    #         "datasets": DATASETS,
    #         "gof_means": plugin_gof_test_means,
    #     }
    # )
    # plugin_gof_df_test.to_csv(f"{plot_dir}/nonamor4plugin_gof_test.csv", index=False)

    # amor_gof_df_train = pd.DataFrame(
    #     {
    #         "datasets": DATASETS,
    #         "gof_means": amor_gof_train_means,
    #     }
    # )
    # amor_gof_df_train.to_csv(f"{plot_dir}/nonamor4amor_gof_train.csv", index=False)

    # amor_gof_df_test = pd.DataFrame(
    #     {
    #         "datasets": DATASETS,
    #         "gof_means": amor_gof_test_means,
    #     }
    # )
    # amor_gof_df_test.to_csv(f"{plot_dir}/nonamor4amor_gof_test.csv", index=False)

    # gof_df = pd.DataFrame(
    #     {"datasets": DATASETS, "gof_means": gof_means, "gof_stds": gof_stds}
    # )
    # gof_df.to_csv(f"{plot_dir}/nonamor_calibration_gof.csv", index=False)

    # ctt_df = pd.DataFrame(
    #     {
    #         "datasets": DATASETS,
    #         "corr_ctt_means": corr_ctt_means,
    #         "corr_ctt_stds": corr_ctt_stds,
    #     }
    # )
    # ctt_df.to_csv(f"{plot_dir}/nonamor_calibration_corr_ctt.csv", index=False)

    # helm_df = pd.DataFrame(
    #     {
    #         "datasets": [d for d in DATASETS if d != "airbench"],
    #         "corr_helm_means": corr_helm_means,
    #         "corr_helm_stds": corr_helm_stds,
    #     }
    # )
    # helm_df.to_csv(f"{plot_dir}/nonamor_calibration_corr_helm.csv", index=False)

    # error_bar_plot_single(
    #     datasets=DATASETS,
    #     means=gof_means,
    #     stds=gof_stds,
    #     plot_path=f"{plot_dir}/nonamor_calibration_summarize_gof",
    #     xlabel=r"Goodness of Fit",
    # )

    # error_bar_plot_single(
    #     datasets=DATASETS,
    #     means=corr_ctt_means,
    #     stds=corr_ctt_stds,
    #     plot_path=f"{plot_dir}/nonamor_calibration_summarize_theta_corr_ctt",
    #     xlabel=r"$\theta$ correlation with CTT",
    # )

    # error_bar_plot_single(
    #     datasets=[d for d in DATASETS if d != "airbench"],
    #     means=corr_helm_means,
    #     stds=corr_helm_stds,
    #     plot_path=f"{plot_dir}/nonamor_calibration_summarize_theta_corr_helm",
    #     xlabel=r"$\theta$ correlation with HELM",
    # )
