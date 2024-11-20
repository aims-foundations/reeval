import argparse
import os
import pickle

import pandas as pd
import torch
import wandb
from datasets import concatenate_datasets, load_dataset
from huggingface_hub import snapshot_download, HfApi
from utils.irt import IRT
from utils.utils import set_seed, str2bool


def calibrate(
    response_matrix,
    D,
    PL,
    fitting_method="mle",
    max_epoch=30000,
    amortized_question=False,
    amortized_student=False,
    amortized_question_hyperparams={},
    amortized_model_hyperparams={},
    item_embeddings=None,
    model_features=None,
    device="cpu",
):
    n_models, n_questions = response_matrix.shape

    irt_model = IRT(
        n_questions=n_questions,
        n_testtaker=n_models,
        D=D,
        PL=PL,
        amortize_item=amortized_question,
        amortize_student=amortized_student,
        amortized_question_hyperparams=amortized_question_hyperparams,
        amortized_model_hyperparams=amortized_model_hyperparams,
        device=device,
    )
    irt_model.fit(
        max_epoch=max_epoch,
        response_matrix=response_matrix,
        method=fitting_method,
        embedding=item_embeddings,
        model_features=model_features,
    )
    return irt_model


if __name__ == "__main__":
    wandb.init(project="reeval")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--D", type=int, default=1)
    parser.add_argument("--PL", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fitting_method", type=str, default="mle", choices=["mle", "mcmc", "em"]
    )
    parser.add_argument("--max_epoch", type=int, default=5000)
    parser.add_argument("--amortized_question", type=str2bool, default=False)
    parser.add_argument("--amortized_student", type=str2bool, default=False)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_folder = snapshot_download(
        repo_id="stair-lab/reeval_responses", repo_type="dataset"
    )
    print("Loading data...")
    response_matrix = torch.load(f"{data_folder}/{args.dataset}/response_matrix.pt").to(
        device=device
    )

    print("Splitting data...")
    all_indices = torch.randperm(response_matrix.shape[1])
    train_indices = all_indices[: int(0.8 * response_matrix.shape[1])]
    test_indices = all_indices[int(0.8 * response_matrix.shape[1]) :]

    # select training data
    response_matrix = response_matrix[:, train_indices]

    # output_dir = f"../../data/{args.fitting_method}_{args.PL}pl{'_amortized' if args.amortized_question else ''}_calibration/{args.dataset}"
    output_dir = f"../../results/calibration/{args.dataset}/s{args.seed}_{args.fitting_method}_{args.PL}pl_{args.D}d{'_aq' if args.amortized_question else ''}{'_as' if args.amortized_student else ''}"
    os.makedirs(output_dir, exist_ok=True)

    # Loading data for amortized calibration
    if args.amortized_question:
        # load item embeddings
        item_embeddings = torch.load(
            f"{data_folder}/{args.dataset}/item_embeddings.pt",
        ).to(device=device)

        # select training data
        item_embeddings = item_embeddings[train_indices]

        amortized_question_hyperparams = {
            "input_dim": item_embeddings.shape[1],
            "n_layers": 1,
            "hidden_dim": None,
        }
    else:
        item_embeddings = None
        amortized_question_hyperparams = None

    if args.amortized_student:
        # load flop
        # compute log(flop) for each student
        # make the model_features = [1, log(flop)]
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

        amortized_model_hyperparams = {
            "input_dim": 2,
            "n_layers": 1,
            "hidden_dim": None,
        }
    else:
        model_features = None
        amortized_model_hyperparams = None

    print("Calibrating...")
    irt_model = calibrate(
        response_matrix=response_matrix,
        D=args.D,
        PL=args.PL,
        fitting_method=args.fitting_method,
        max_epoch=args.max_epoch,
        amortized_question=args.amortized_question,
        amortized_student=args.amortized_student,
        amortized_question_hyperparams=amortized_question_hyperparams,
        amortized_model_hyperparams=amortized_model_hyperparams,
        item_embeddings=item_embeddings,
        model_features=model_features,
        device=device,
    )

    abilities = irt_model.get_abilities().cpu().detach().tolist()
    item_parms = irt_model.get_item_parameters().cpu().detach().tolist()
    
    wandb.finish()

    pickle.dump(abilities, open(f"{output_dir}/abilities.pkl", "wb"))
    pickle.dump(item_parms, open(f"{output_dir}/item_parms.pkl", "wb"))

    # save the indices for train and test
    pickle.dump(train_indices, open(f"{output_dir}/train_indices.pkl", "wb"))
    pickle.dump(test_indices, open(f"{output_dir}/test_indices.pkl", "wb"))
    
    if args.amortized_question:
        torch.save(irt_model.item_parameters_nn.state_dict(), f"{output_dir}/item_parameters_nn.pt")
        
    if args.amortized_student:
        torch.save(irt_model.student_parameters_nn.state_dict(), f"{output_dir}/student_parameters_nn.pt")
        
    # stair-lab/reeval_results
    hf_repo_path  = f"{args.dataset}/s{args.seed}_{args.fitting_method}_{args.PL}pl_{args.D}d{'_aq' if args.amortized_question else ''}{'_as' if args.amortized_student else ''}"
    upload_api = HfApi()
    upload_api.create_repo(repo_id="stair-lab/reeval_results", repo_type="dataset", exist_ok=True)
    upload_api.upload_folder(
        folder_path=output_dir,
        repo_id="stair-lab/reeval_results",
        repo_type="dataset",
        path_in_repo=hf_repo_path,
    )