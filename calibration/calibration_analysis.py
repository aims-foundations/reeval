import argparse
import copy
import itertools
import os
import pickle

import matplotlib.pyplot as plt
import pandas as pd
import torch
from gen_figures.plot import goodness_of_fit_plot, theta_corr_plot, auc_roc_plot
from huggingface_hub import snapshot_download
from tqdm import tqdm
from tueplots import bundles
from utils.constants import DATASETS
from utils.irt import IRT
from utils.utils import arg2str
# supress warning 
import warnings
warnings.filterwarnings("ignore")


plt.rcParams.update(bundles.iclr2024())

# Turn off warnings
import warnings
warnings.filterwarnings("ignore")

def get_amortized_questions(result_path, args):
    item_parameters_nn = pickle.load(
        open(f"{result_path}/item_parameters_nn.pkl", "rb")
    )
    item_embeddings = torch.load(f"{data_path}/item_embeddings.pt").to(
        device=args.device
    )

    item_parms = item_parameters_nn(item_embeddings)
    item_parms = IRT.apply_item_constrains(item_parms, D=args.D, PL=args.PL).detach()
    return item_parms

def get_amortized_students(result_path, args):
    student_parameters_nn = pickle.load(
        open(f"{result_path}/student_parameters_nn.pkl", "rb")
    )

    model_keys = pd.read_csv(f"{data_path}/model_keys.csv")
    model_features = model_keys["flop"].tolist()
    model_features = torch.tensor(
        model_features, dtype=torch.float32, device=args.device
    )
    model_features = torch.log(model_features).unsqueeze(-1)

    # Fill nan with -1
    model_features[torch.isnan(model_features)] = -1
    abilities = student_parameters_nn(model_features).detach()
    
    abilities = IRT.apply_student_constrains(abilities, model_features)
    return abilities

def mask_student_whose_feature_missing(items):
    abilities_mask = ~torch.isnan(items[0]).any(dim=1)
    return [item[abilities_mask] for item in items]



if __name__ == "__main__":
    fig, axs = plt.subplots(4)
    D = [1]
    PL = [1] # [1, 2, 3]
    fitting_methods = ["mle"] # ["em", "mle"]
    amortized_question = [False] # [False, True]
    amortized_student = [False] # [False, True]
    seeds = [42]
    nls = [1]

    cartesian_product = itertools.product(
        D, PL, fitting_methods, amortized_question, amortized_student, seeds, nls
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_folder = snapshot_download(
        repo_id="stair-lab/reeval_responses", repo_type="dataset"
    )
    result_folder = snapshot_download(
        repo_id="stair-lab/reeval_results", repo_type="dataset"
    )

    for arg_list in cartesian_product:
        parser = argparse.ArgumentParser()
        args = parser.parse_args()
        args.D = arg_list[0]
        args.PL = arg_list[1]
        args.fitting_method = arg_list[2]
        args.amortized_question = arg_list[3]
        args.amortized_student = arg_list[4]
        args.seed = arg_list[5]
        args.n_layers = arg_list[6]
        args.hidden_dim = None
        args.device = device

        metrics = {"train": {"mean": [], "std": []}, "test": {"mean": [], "std": []}}
        gof = copy.deepcopy(metrics)
        corr_ctt = copy.deepcopy(metrics)
        corr_helm = copy.deepcopy(metrics)
        auc = copy.deepcopy(metrics)

        list_datasets = []
        for dataset in tqdm(DATASETS):
            # if dataset != "airbench":
            #     continue
            list_datasets.append(dataset)

            # Setup the arguments
            args.dataset = dataset
            dataset_and_method_name = arg2str(args)
            plot_dir = f"../plot/{dataset_and_method_name}"
            os.makedirs(plot_dir, exist_ok=True)

            print(f"Processing {dataset_and_method_name}")

            # Load the data
            data_path = f"{data_folder}/{dataset}"
            result_path = f"{result_folder}/{dataset_and_method_name}"

            # Load the train/test indices for question and student
            train_question_indices = pickle.load(
                open(f"{result_path}/train_question_indices.pkl", "rb")
            )
            test_question_indices = pickle.load(
                open(f"{result_path}/test_question_indices.pkl", "rb")
            )
            train_student_indices = pickle.load(
                open(f"{result_path}/train_model_indices.pkl", "rb")
            )
            test_student_indices = pickle.load(
                open(f"{result_path}/test_model_indices.pkl", "rb")
            )
            model_keys = pd.read_csv(f"{data_path}/model_keys.csv")

            response_matrix_full = torch.load(f"{data_path}/response_matrix.pt").to(
                device=device, dtype=torch.float32
            )
            response_matrix_train = response_matrix_full[train_student_indices][
                :, train_question_indices
            ]

            helm_score = torch.tensor(
                model_keys["helm_score"].to_numpy(), device=device
            ).reshape(-1, 1)
            helm_score_train = helm_score[train_student_indices]

            ctt_score = torch.tensor(
                model_keys["ctt_score"].to_numpy(), device=device
            ).reshape(-1, 1)
            ctt_score_train = ctt_score[train_student_indices]

            if args.amortized_question and args.amortized_student:
                item_parms = get_amortized_questions(result_path, args)
                item_parms_train = item_parms[train_question_indices]
                item_parms_test = item_parms[test_question_indices]

                abilities = get_amortized_students(result_path, args)
                abilities_train = abilities[train_student_indices]
                abilities_test = abilities[test_student_indices]
                
                response_matrix_test = response_matrix_full[test_student_indices][
                    :, test_question_indices
                ]
                
                helm_score_test = helm_score[test_student_indices]
                ctt_score_test = ctt_score[test_student_indices]
                
                # filter nan values                
                items = [abilities_train, response_matrix_train, helm_score_train, ctt_score_train]
                items = mask_student_whose_feature_missing(items)
                abilities_train, response_matrix_train, helm_score_train, ctt_score_train = items

                items = [abilities_test, response_matrix_test, helm_score_test, ctt_score_test]
                items = mask_student_whose_feature_missing(items)
                abilities_test, response_matrix_test, helm_score_test, ctt_score_test = items

            elif args.amortized_question and not args.amortized_student:
                item_parms = get_amortized_questions(result_path, args)
                item_parms_train = item_parms[train_question_indices]
                item_parms_test = item_parms[test_question_indices]

                abilities_train = pickle.load(
                    open(f"{result_path}/abilities.pkl", "rb")
                )
                abilities_train = torch.tensor(abilities_train, device=device)

                # since we are *testing* the generalizability of amortized question parameter prediction
                # on the train students, we need to use the ability of the train students
                abilities_test = abilities_train

                response_matrix_test = response_matrix_full[train_student_indices][
                    :, test_question_indices
                ]

                helm_score_test = helm_score[train_student_indices]
                ctt_score_test = ctt_score[train_student_indices]

            elif not args.amortized_question and args.amortized_student:
                item_parms_train = pickle.load(
                    open(f"{result_path}/item_parms.pkl", "rb")
                )
                item_parms_train = torch.tensor(item_parms_train, device=device)

                # since we are *testing* the generalizability of amortized student ability prediction
                # on the train items, we need to use the item parameters of the train items
                item_parms_test = item_parms_train

                abilities = get_amortized_students(result_path, args)
                abilities_train = abilities[train_student_indices]
                abilities_test = abilities[test_student_indices]

                response_matrix_test = response_matrix_full[test_student_indices][
                    :, train_question_indices
                ]

                helm_score_test = helm_score[test_student_indices]
                ctt_score_test = ctt_score[test_student_indices]
                
                # filter nan values
                # breakpoint()

                items = [abilities_train, response_matrix_train, helm_score_train, ctt_score_train]
                items = mask_student_whose_feature_missing(items)
                abilities_train, response_matrix_train, helm_score_train, ctt_score_train = items       

                items = [abilities_test, response_matrix_test, helm_score_test, ctt_score_test]
                items = mask_student_whose_feature_missing(items)
                abilities_test, response_matrix_test, helm_score_test, ctt_score_test = items

            else:
                item_parms_train = pickle.load(
                    open(f"{result_path}/item_parms.pkl", "rb")
                )
                item_parms_train = torch.tensor(item_parms_train, device=device)
                item_parms_test = None

                abilities_train = pickle.load(
                    open(f"{result_path}/abilities.pkl", "rb")
                )
                abilities_train = torch.tensor(abilities_train, device=device)
                abilities_test = None

                response_matrix_test = None
                helm_score_test = None
                ctt_score_test = None

            train_test_iters = [
                (item_parms_train, abilities_train, response_matrix_train, helm_score_train, ctt_score_train, "train"),
                (item_parms_test, abilities_test, response_matrix_test, helm_score_test, ctt_score_test, "test")
            ]

            for train_test_iter in train_test_iters:
                item_parms, abilities, response_matrix, helm_score, ctt_score, is_train = train_test_iter
             
                if item_parms is None and abilities is None:
                    continue

                # metric 1: GOF
                gof_mean, gof_std = goodness_of_fit_plot(
                    z=item_parms,
                    theta=abilities,
                    y=response_matrix,
                    plot_path=f"{plot_dir}/goodness_of_fit_{is_train}",
                )
                gof[is_train]["mean"].append(gof_mean)
                gof[is_train]["std"].append(gof_std)
                print(
                    f"{dataset_and_method_name} {is_train} GOF: {gof_mean:.4f} ± {gof_std:.4f}"
                )

                # metric 2: correlation with CTT
                corr_ctt_mean, corr_ctt_std = theta_corr_plot(
                    mode="ctt",
                    theta=abilities,
                    ctt_score=ctt_score,
                    plot_path=f"{plot_dir}/theta_corr_ctt_{is_train}",
                )
                corr_ctt[is_train]["mean"].append(corr_ctt_mean)
                corr_ctt[is_train]["std"].append(corr_ctt_std)
                print(
                    f"{dataset_and_method_name} {is_train} corr_ctt: {corr_ctt_mean:.4f} ± {corr_ctt_std:.4f}"
                )

                # metric 3: correlation with HELM
                corr_helm_mean, corr_helm_std = theta_corr_plot(
                    mode="helm",
                    theta=abilities,
                    helm_score=helm_score,
                    plot_path=f"{plot_dir}/theta_corr_helm_{is_train}",
                )
                corr_helm[is_train]["mean"].append(corr_helm_mean)
                corr_helm[is_train]["std"].append(corr_helm_std)
                print(
                    f"{dataset_and_method_name} {is_train} corr_helm: {corr_helm_mean:.4f} ± {corr_helm_std}"
                )

                # metric 4: AUC-ROC
                auc_mean ,auc_std = auc_roc_plot(
                    item_parms=item_parms,
                    theta=abilities,
                    y=response_matrix,
                    plot_path=f"{plot_dir}/auc_roc_{is_train}",
                    bootstrap_size=2
                )
                auc[is_train]["mean"].append(auc_mean)
                auc[is_train]["std"].append(auc_std)
                print(
                    f"{dataset_and_method_name} {is_train} AUC-ROC: {auc_mean:.4f} ± {auc_std:.4f}"
                )

        # x = range(len(DATASETS))
        x = range(len(list_datasets))

        for is_train in ["train", "test"]:
            if len(gof[is_train]["mean"]) == 0:
                continue

            c = "blue" if is_train == "train" else "red"

            axs[0].plot(x, gof[is_train]["mean"], color=c)
            axs[0].errorbar(
                x=x, y=gof[is_train]["mean"], yerr=gof[is_train]["std"], color=c
            )

            axs[1].plot(x, corr_ctt[is_train]["mean"], color=c)
            axs[1].errorbar(
                x=x,
                y=corr_ctt[is_train]["mean"],
                yerr=corr_ctt[is_train]["std"],
                color=c,
            )

            axs[2].plot(x, corr_helm[is_train]["mean"], color=c)
            axs[2].errorbar(
                x=x,
                y=corr_helm[is_train]["mean"],
                yerr=corr_helm[is_train]["std"],
                color=c,
            )

            axs[3].plot(x, auc[is_train]["mean"], color=c)
            axs[3].errorbar(
                x=x, y=auc[is_train]["mean"], yerr=auc[is_train]["std"], color=c
            )

    axs[0].set_title("Goodness of Fit")
    axs[1].set_title("Correlation with CTT")
    axs[2].set_title("Correlation with HELM")
    axs[3].set_title("AUC-ROC")
    plt.xticks(x, list_datasets, rotation=90)
    plt.savefig(f"../plot/calibration_analysis.png", bbox_inches="tight", dpi=300)
