import argparse
import pickle

import matplotlib.pyplot as plt

import pandas as pd
from huggingface_hub import snapshot_download
from utils.utils import str2bool

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--D", type=int, default=1)
    parser.add_argument("--PL", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fitting_method", type=str, default="mle", choices=["mle", "mcmc", "em"]
    )
    parser.add_argument("--amortized_question", type=str2bool, default=False)
    parser.add_argument("--amortized_student", type=str2bool, default=False)
    args = parser.parse_args()

    output_dir = f"../results/calibration/{args.dataset}/s{args.seed}_{args.fitting_method}_{args.PL}pl_{args.D}d{'_aq' if args.amortized_question else ''}{'_as' if args.amortized_student else ''}"

    # Load abilities.pkl
    with open(f"{output_dir}/abilities.pkl", "rb") as f:
        abilities = pickle.load(f)

    # Load list of model keys
    data_folder = snapshot_download(
        repo_id="stair-lab/reeval_responses", repo_type="dataset"
    )
    model_keys = pd.read_csv(f"{data_folder}/{args.dataset}/model_keys.csv")

    for d in range(args.D):
        list_thetas = []
        list_names = []
        for ri, row in model_keys.iterrows():
            model_hf_id = row["huggingface_model_id"]
            model_name = row["model_name"]

            # Check if model name is nan
            if pd.isna(model_hf_id) and not model_name.startswith("openai"):
                continue

            list_thetas.append(abilities[ri][d])
            list_names.append(model_name)

        # Sort by ability
        list_thetas, list_names = zip(
            *sorted(zip(list_thetas, list_names), key=lambda x: x[0])
        )
        for i in range(len(list_thetas)):
            print(f"{list_names[i]}: {list_thetas[i]}")
        # Plot histogram
        plt.hist(list_thetas, bins=20)
        plt.xlabel("Ability")
        plt.ylabel("Count")
        plt.title("Ability Distribution")
        plt.savefig(f"{output_dir}/ability_distribution_{d}.png")
        plt.close()

    # Read model list
    # model_list = pd.read_csv(f"./model_list.csv")

    # # Get huggingface model id
    # list_model_hf_id = []
    # for model_name in model_list["model_name"]:
    #     model_hf_id = model_keys[model_keys["model_name"] == model_name]["huggingface_model_id"]
    #     if len(model_hf_id) != 0:
    #         model_hf_id = model_hf_id.values[0]
    #     else:
    #         model_hf_id = model_name
    #     list_model_hf_id.append(model_hf_id)
    #     print(f"{model_name}: {model_hf_id}")

    # # Save list model hf id using pandas
    # model_hf_id_df = pd.DataFrame(list_model_hf_id, columns=["huggingface_model_id"])
    # model_hf_id_df.to_csv(f"./model_hf_id.csv", index=False)
