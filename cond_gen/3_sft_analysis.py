import argparse
import gc
import io
import os
import pickle

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from embed_text_package.embed_text_v2 import Embedder
from gen_figures.plot import plot_hist
from huggingface_hub import HfApi, snapshot_download
from ppo_reward_model import extract_score
from transformers import GenerationConfig
from vllm import LLM, SamplingParams


def call_diff(ds, gt_zs, reward_model, restart, batch_size, device):
    dataloader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)
    emb = embdr.get_embeddings(dataloader, "meta-llama/Meta-Llama-3-8B", ["text"])
    embs = torch.tensor(emb["text"], device=device)

    pred_zs = reward_model(embs).reshape(-1, restart)
    # >>> batch_size * restart

    gt_zs = torch.tensor([gt_zs for _ in range(restart)], device=device).T
    mae = torch.abs(pred_zs - gt_zs).min(dim=-1)

    # Select the pred_zs using mae.indices
    pred_zs = pred_zs[range(len(mae.indices)), mae.indices]

    return pred_zs.tolist(), mae.values.tolist(), mae.indices.tolist()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="stair-lab/reeval_question_generator_sft"
    )

    parser.add_argument(
        "--embedder_name", type=str, default="meta-llama/Meta-Llama-3-8B"
    )
    parser.add_argument("--dataset", type=str, default="airbench")
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--num_restarts", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--run_generation", action="store_true")
    args = parser.parse_args()

    upload_api = HfApi()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_short_name = args.model.split("/")[-1]
    ds_model_short_name = ""
    if model_short_name == "reeval_question_generator_sft":
        model_short_name = ""
    else:
        model_short_name = "_" + model_short_name
        if "mistral" in model_short_name:
            ds_model_short_name = "-Mistral-7B-Instruct-v0.3"

    plot_dir = f"../plot/sft/{args.dataset}{model_short_name}"
    os.makedirs(plot_dir, exist_ok=True)

    generation_dir = f"../results/generated_questions/{args.dataset}{model_short_name}"
    os.makedirs(generation_dir, exist_ok=True)

    train_dataset = load_dataset(
        f"stair-lab/reeval{ds_model_short_name}-ppo", args.dataset, split="train"
    )
    train_prompts = train_dataset["text"][: args.num_samples]

    train_gt_zs = [extract_score(p) for p in train_prompts]

    if args.run_generation:
        generation_config = GenerationConfig.from_pretrained(args.model)
        sampling_params = SamplingParams(
            n=args.num_restarts,
            best_of=2 * args.num_restarts,
            temperature=generation_config.temperature,
            top_p=generation_config.top_p,
            max_tokens=256,
            stop_token_ids=(
                generation_config.eos_token_id
                if isinstance(generation_config.eos_token_id, list)
                else [generation_config.eos_token_id]
            ),
        )
        llm = LLM(
            model=args.model,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=torch.cuda.device_count(),
            dtype=torch.float16,
            max_num_seqs=128,
        )
        train_outputs = llm.generate(train_prompts, sampling_params)

        train_answers = [sample.text for o in train_outputs for sample in o.outputs]
        train_answer_df = pd.DataFrame(train_answers, columns=["text"])
        train_answer_dataset = Dataset.from_pandas(train_answer_df)

        # Save the answers
        train_answers_file = io.BytesIO()
        train_answer_df.to_csv(train_answers_file, index=False)
        upload_api.upload_file(
            repo_id="stair-lab/reeval_generated_questions",
            repo_type="dataset",
            path_in_repo=f"sft/{args.dataset}{model_short_name}/train_answers.csv",
            path_or_fileobj=train_answers_file,
        )
        train_answer_df.to_csv(f"{generation_dir}/train_answers.csv", index=False)

        del llm
        torch.cuda.empty_cache()
        gc.collect()

    else:
        # Load the generated answers
        # train_answer_df = pd.read_csv(f"{generation_dir}/train_answers.csv")
        hf_folder = snapshot_download(
            repo_id="stair-lab/reeval_generated_questions", repo_type="dataset"
        )
        train_answer_df = pd.read_csv(
            f"{hf_folder}/sft/{args.dataset}{model_short_name}/train_answers.csv",
            engine="python",
        )

        train_answers = train_answer_df["text"].tolist()
        num_restarts = int(len(train_answers) / len(train_prompts))
        # 128, because of the bug of vLLM, they return num_restarts instead of 64

        train_answers = train_answers[: args.num_samples * num_restarts] # 128000
        train_answer_dataset = Dataset.from_pandas(
            train_answer_df[: args.num_samples * num_restarts]
        )
        # >>> num_samples * num_restarts

        # Load the embedding model
        embdr = Embedder()
        embdr.load(
            args.embedder_name,
            gpu_memory_utilization=0.8,
            tensor_parallel_size=torch.cuda.device_count(),
            dtype=torch.float16,
        )

        # Load the reward model
        result_folder = snapshot_download(
            repo_id="stair-lab/reeval_results", repo_type="dataset"
        )

        with open(
            f"{result_folder}/{args.dataset}/s42_mle_1pl_1d_aq_nl1/item_parameters_nn.pkl",
            "rb",
        ) as f:
            reward_model = pickle.load(f)
            reward_model = reward_model.to(device)

        difficulty_predictor = lambda x: reward_model(x)[:, 0]

        # Re-calculating the number of restarts due to vLLM's bug
        train_diffs, train_maes, train_indices = call_diff(
            train_answer_dataset,
            train_gt_zs,
            difficulty_predictor,
            num_restarts,
            batch_size=args.batch_size,
            device=device,
        )

        # Reshape the answers to List[List[str]]
        train_answers = [
            train_answers[i : i + num_restarts]
            for i in range(0, len(train_answers), num_restarts)
        ]

        # Picke the answers using the indices
        train_answers = [
            train_answers[i][train_indices[i]] for i in range(len(train_answers))
        ]

        # Saving the results
        pickle.dump(train_diffs, open(f"{generation_dir}/train_diffs.pkl", "wb"))
        pickle.dump(train_maes, open(f"{generation_dir}/train_maes.pkl", "wb"))
        pickle.dump(train_indices, open(f"{generation_dir}/train_indices.pkl", "wb"))

        train_diffs_file = io.BytesIO()
        pickle.dump(train_diffs, train_diffs_file)
        upload_api.upload_file(
            repo_id="stair-lab/reeval_generated_questions",
            repo_type="dataset",
            path_in_repo=f"sft/{args.dataset}{model_short_name}/train_diffs.pkl",
            path_or_fileobj=train_diffs_file,
        )
        train_maes_file = io.BytesIO()
        pickle.dump(train_maes, train_maes_file)
        upload_api.upload_file(
            repo_id="stair-lab/reeval_generated_questions",
            repo_type="dataset",
            path_in_repo=f"sft/{args.dataset}{model_short_name}/train_maes.pkl",
            path_or_fileobj=train_maes_file,
        )
        train_indices_file = io.BytesIO()
        pickle.dump(train_indices, train_indices_file)
        upload_api.upload_file(
            repo_id="stair-lab/reeval_generated_questions",
            repo_type="dataset",
            path_in_repo=f"sft/{args.dataset}{model_short_name}/train_indices.pkl",
            path_or_fileobj=train_indices_file,
        )

        # Save the answers
        train_answer_df = pd.DataFrame(train_answers, columns=["text"])
        train_answer_df.to_csv(
            f"{generation_dir}/train_answers_filtered.csv", index=False
        )
        train_answer_file = io.BytesIO()
        train_answer_df.to_csv(train_answer_file, index=False)
        upload_api.upload_file(
            repo_id="stair-lab/reeval_generated_questions",
            repo_type="dataset",
            path_in_repo=f"sft/{args.dataset}{model_short_name}/train_answers_filtered.csv",
            path_or_fileobj=train_answer_file,
        )

        train_maes = pickle.load(
            open(
                f"{hf_folder}/sft/{args.dataset}{model_short_name}/train_maes.pkl", "rb"
            )
        )
        # Plot the histograms
        plot_hist(
            data=train_maes,
            plot_path=f"{plot_dir}/mae_hist.png",
            ylabel=r"train $z$ difference",
        )
