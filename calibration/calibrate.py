import argparse
import os
import pickle

import pandas as pd
import torch
import wandb
from amortized_irt import IRT
from check_calibration_results import check_results
from huggingface_hub import HfApi, snapshot_download
from utils.utils import arg2str, set_seed, str2bool

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
    parser.add_argument("--train_size", type=float, default=0.8)
    parser.add_argument("--max_epoch", type=int, default=5000)
    parser.add_argument("--amortized_question", type=str2bool, default=False)
    parser.add_argument("--amortized_student", type=str2bool, default=False)
    parser.add_argument("--n_layers", type=int, default=1)
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--report_to", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="../results/calibration")
    parser.add_argument(
        "--output_hf_repo", type=str, default="stair-lab/reeval_results"
    )
    parser.add_argument("--force_run", type=str2bool, default=False)
    parser.add_argument(
        "--embedder_name", type=str, default="meta-llama/Meta-Llama-3-8B"
    )

    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedder_short_name = args.embedder_name.split("/")[1]
    if not args.amortized_question:
        embedder_short_name = ""
        
    data_folder = snapshot_download(
        repo_id="stair-lab/reeval_matrices", repo_type="dataset"
    )
    output_dir = os.path.join(args.output_dir, arg2str(args))
    os.makedirs(output_dir, exist_ok=True)

    print(f"Output directory: {output_dir}")
    if not args.force_run and check_results(
        output_dir, args.amortized_question, args.amortized_student
    ):
        print("Already calibrated")
        wandb.finish()
        exit(0)

    print("Loading data...")
    response_matrix = torch.load(f"{data_folder}/{args.dataset}/response_matrix.pt").to(
        device=device, dtype=torch.float32
    )

    print("Splitting data...")
    all_question_indices = torch.randperm(response_matrix.shape[1])
    train_question_indices = all_question_indices[
        : int(args.train_size * response_matrix.shape[1])
    ]
    test_question_indices = all_question_indices[
        int(args.train_size * response_matrix.shape[1]) :
    ]

    all_model_indices = torch.randperm(response_matrix.shape[0])
    train_model_indices = all_model_indices[
        : int(args.train_size * response_matrix.shape[0])
    ]
    test_model_indices = all_model_indices[
        int(args.train_size * response_matrix.shape[0]) :
    ]

    # select training data
    response_matrix = response_matrix[train_model_indices]
    response_matrix = response_matrix[:, train_question_indices]

    # Loading data for amortized calibration
    if args.amortized_question:
        _, embedder_name = args.embedder_name.split("/")
        # load item embeddings
        item_embeddings = torch.load(
            f"{data_folder}/{args.dataset}/{embedder_name}_item_embeddings.pt",
        ).to(device=device)

        # select training data
        item_embeddings = item_embeddings[train_question_indices]

        if item_embeddings.shape[0] > 4 * 4096:
            n_layers = args.n_layers
            hidden_dim = args.hidden_dim
        else:
            n_layers = 1
            hidden_dim = None

        amortized_question_hyperparams = {
            "input_dim": item_embeddings.shape[1],
            "n_layers": n_layers,
            "hidden_dim": hidden_dim,
        }
    else:
        item_embeddings = None
        amortized_question_hyperparams = None

    if args.amortized_student:
        # load flop as model features
        model_keys = pd.read_csv(f"{data_folder}/{args.dataset}/model_keys.csv")
        model_features = model_keys["flop"].tolist()
        model_features = torch.tensor(
            model_features, dtype=torch.float32, device=device
        )
        model_features = model_features[train_model_indices]
        model_features = torch.log(model_features).unsqueeze(-1)
        # Fill nan with -1
        model_features[torch.isnan(model_features)] = -1

        if args.dataset == "combined_data":
            amortized_model_hyperparams = {
                "input_dim": 1,
                "n_layers": 2,
                "hidden_dim": 64,
            }
        else:
            amortized_model_hyperparams = {
                "input_dim": 1,
                "n_layers": 1,
                "hidden_dim": None,
            }
    else:
        model_features = None
        amortized_model_hyperparams = None

    print("Calibrating...")
    n_models, n_questions = response_matrix.shape
    irt_model = IRT(
        D=args.D,
        PL=args.PL,
        device=device,
        report_to=args.report_to,
    )
    irt_model.fit(
        max_epoch=args.max_epoch,
        response_matrix=response_matrix,
        method=args.fitting_method,
        embedding=item_embeddings,
        model_features=model_features,
        amortized_question_hyperparams=amortized_question_hyperparams,
        amortized_model_hyperparams=amortized_model_hyperparams,
    )

    # save results
    abilities = irt_model.get_abilities().cpu().detach().tolist()
    item_parms = irt_model.get_item_parameters().cpu().detach().tolist()

    pickle.dump(abilities, open(f"{output_dir}/abilities.pkl", "wb"))
    pickle.dump(item_parms, open(f"{output_dir}/item_parms.pkl", "wb"))

    pickle.dump(
        train_question_indices, open(f"{output_dir}/train_question_indices.pkl", "wb")
    )
    pickle.dump(
        test_question_indices, open(f"{output_dir}/test_question_indices.pkl", "wb")
    )
    pickle.dump(
        train_model_indices, open(f"{output_dir}/train_model_indices.pkl", "wb")
    )
    pickle.dump(test_model_indices, open(f"{output_dir}/test_model_indices.pkl", "wb"))

    if args.amortized_question:
        torch.save(
            irt_model.item_parameters_nn.state_dict(),
            f"{output_dir}/item_parameters_nn.pt",
        )
        pickle.dump(
            irt_model.item_parameters_nn,
            open(f"{output_dir}/item_parameters_nn.pkl", "wb"),
        )

    if args.amortized_student:
        torch.save(
            irt_model.ability_nn.state_dict(), f"{output_dir}/student_parameters_nn.pt"
        )
        pickle.dump(
            irt_model.ability_nn, open(f"{output_dir}/student_parameters_nn.pkl", "wb")
        )

    upload_api = HfApi()
    upload_api.create_repo(
        repo_id=f"{args.output_hf_repo}",
        repo_type="dataset",
        exist_ok=True,
    )
    
    if embedder_short_name != "":
        embedder_short_name += "/"
    upload_api.upload_folder(
        repo_id=f"{args.output_hf_repo}",
        folder_path=f"{output_dir}",
        repo_type="dataset",
        path_in_repo=f"{embedder_short_name}{arg2str(args)}",
    )

    wandb.finish()
