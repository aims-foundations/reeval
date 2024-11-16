import argparse
import os
import pickle

import pandas as pd
import torch
from datasets import concatenate_datasets, load_dataset
from utils.irt import IRT
from utils.utils import set_seed, str2bool


def calibrate(
    response_matrix,
    D,
    PL,
    fitting_method="mle",
    max_epoch=30000,
    amortized=False,
    amortized_model_hyperparams={},
    item_embeddings=None,
    device="cpu",
):
    n_models, n_questions = response_matrix.shape

    irt_model = IRT(
        n_questions=n_questions,
        n_testtaker=n_models,
        D=D,
        PL=PL,
        amortize_item=amortized,
        amortized_model_hyperparams=amortized_model_hyperparams,
    )
    irt_model = irt_model.to(device)
    irt_model.fit(
        max_epoch=max_epoch,
        response_matrix=response_matrix,
        method=fitting_method,
        embedding=item_embeddings if amortized else None,
    )
    return irt_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--D", type=int, default=1)
    parser.add_argument("--PL", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fitting_method", type=str, default="mle", choices=["mle", "mcmc", "em"]
    )
    parser.add_argument("--max_epoch", type=int, default=3000)
    parser.add_argument("--amortized", type=str2bool, default=False)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dir = "../../data/pre_calibration"
    output_dir = f"../../data/{args.fitting_method}_{args.PL}pl{'_amortized' if args.amortized else ''}_calibration/{args.dataset}"
    os.makedirs(output_dir, exist_ok=True)

    # Loading data for amortized calibration
    if args.amortized:
        hf_repo = f"stair-lab/reeval_{args.dataset}-embed"
        dataset_info = load_dataset(hf_repo, split=None)
        splits = dataset_info.keys()
        datasets = [load_dataset(hf_repo, split=split) for split in splits]
        dataset = concatenate_datasets(datasets)
        item_embeddings = torch.tensor(
            dataset["embed"], dtype=torch.float32, device=device
        )

        amortized_model_hyperparams = {
            "input_dim": item_embeddings.shape[1],
            "n_layers": 1,
            "hidden_dim": None,
        }
    else:
        amortized_model_hyperparams = None

    y = pd.read_csv(f"{input_dir}/{args.dataset}/matrix.csv", index_col=0).values
    response_matrix = torch.tensor(y, dtype=torch.float32, device=device)

    irt_model = calibrate(
        response_matrix=response_matrix,
        D=args.D,
        PL=args.PL,
        fitting_method=args.fitting_method,
        max_epoch=args.max_epoch,
        amortized=args.amortized,
        amortized_model_hyperparams=amortized_model_hyperparams,
        item_embeddings=item_embeddings if args.amortized else None,
        device=device,
    )

    abilities = irt_model.get_abilities().cpu().detach().tolist()
    item_parms = irt_model.get_item_parameters().cpu().detach().tolist()

    pickle.dump(abilities, open(f"{output_dir}/abilities.pkl", "wb"))
    pickle.dump(item_parms, open(f"{output_dir}/item_parms.pkl", "wb"))
