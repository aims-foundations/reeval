import argparse
import os
import pandas as pd
import torch
import pickle
from utils.irt import IRT
from utils.utils import set_seed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--D", type=int, default=1)
    parser.add_argument("--PL", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fitting_method", type=str, default="mle", choices=["mle", "mcmc", "em"])
    parser.add_argument("--max_epoch", type=int, default=3000)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dir = "../../data/pre_calibration"
    output_dir = f"../../data/{args.fitting_method}_{args.PL}pl_calibration/{args.dataset}"
    os.makedirs(output_dir, exist_ok=True)

    y = pd.read_csv(f"{input_dir}/{args.dataset}/matrix.csv", index_col=0).values    
    response_matrix = torch.tensor(y, dtype=torch.float32, device=device)
    n_models, n_questions = response_matrix.shape

    irt_model = IRT(n_questions, n_models, args.D, args.PL)
    irt_model = irt_model.to(device)
    irt_model.fit(max_epoch=args.max_epoch, response_matrix=response_matrix, method=args.fitting_method)
    abilities = irt_model.get_abilities().cpu().detach().tolist()
    item_parms = irt_model.get_item_parameters().cpu().detach().tolist()
    
    pickle.dump(abilities, open(f"{output_dir}/abilities.pkl", "wb"))
    pickle.dump(item_parms, open(f"{output_dir}/item_parms.pkl", "wb"))