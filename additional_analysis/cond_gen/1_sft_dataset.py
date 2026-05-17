import argparse
import copy
import pickle

import numpy as np
import pandas as pd
from datasets import Dataset
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
from utils.utils import set_seed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--fitting_method", type=str, default="mle")
    parser.add_argument("--PL", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    # Set seed
    set_seed(42)

    # Load the tokenizer
    model_short_name = args.model.split("/")[-1]
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Get the dataset description
    # description = DESCRIPTION_MAP[args.dataset]

    # Load the difficulty scores
    item_parms_folder = snapshot_download(
        repo_id="stair-lab/reeval_results",
        repo_type="dataset",
    )
    item_parms = pickle.load(
        open(
            f"{item_parms_folder}/{args.dataset}/s42_{args.fitting_method}_{args.PL}pl_1d_nl1/item_parms.pkl",
            "rb",
        )
    )
    # >>> n_questions x (3 + D)
    difficulty = np.array(item_parms)[:, 0].tolist()

    # Load the question dataset
    data_folder = snapshot_download(
        repo_id="stair-lab/reeval_matrices", repo_type="dataset"
    )
    question_df = pd.read_csv(f"{data_folder}/{args.dataset}/question_keys.csv")
    breakpoint()

    train_indices = pickle.load(
        open(
            f"{item_parms_folder}/{args.dataset}/s42_{args.fitting_method}_{args.PL}pl_1d_nl1/train_question_indices.pkl",
            "rb",
        )
    )
    question_df = question_df.iloc[train_indices]

    if args.dataset == "air-bench/air_bench_2024":
        # remove question that are too long (which also includes the answer)
        prompt_len = np.array([len(prompt) for prompt in question_df["prompt"]])
        question_df = question_df[prompt_len < 1024]
        difficulty = [
            difficulty[i] for i in range(len(difficulty)) if prompt_len[i] < 1024
        ]

    question_dataset = Dataset.from_pandas(question_df)

    # Ensure the length of the dataset and difficulty scores are the same
    assert len(question_dataset) == len(difficulty)

    if args.dataset == "classic/mmlu":
        claim = (
            "The test taker knows how to use the most appropriate statistical "
            "techniques to analyze data and draw valid conclusions."
        )
    elif args.dataset == "air-bench/air_bench_2024":
        claim = (
            "The test taker has the knowledge to answer knowledge-intensive "
            "questions across 57 domains"
        )

    # Define the chat template
    user_content = (
        "You are acting as a world-class psychometrician. Your job is to create a question with a specified "
        "difficulty score. A lower score indicates a more difficult question, making it more likely that the "
        "test taker will answer incorrectly. The correctness of the response determines whether a specific "
        "claim is accepted or rejected. You should design the question to seek appropriate evidence to accept "
        f"or reject a given claim. The claim: {claim}. "
        "Difficulty: %s. "
        "Question: "
    )

    sft_chat = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": "%s"},
    ]
    template = tokenizer.apply_chat_template(
        sft_chat, tokenize=False, add_generation_prompt=False
    )

    # Process the text
    def process_text(text, difficulty):
        return {"text": template % (round(difficulty, 2), text)}

    # Rename `prompt` to  `text`
    if args.dataset == "classic/mmlu":
        original_difficulty = copy.deepcopy(difficulty)
        difficulty = []
        list_raw_question = []
        for i, question in enumerate(question_dataset):
            question = question["prompt"].split("Question: ")[-1]
            if (
                "A." not in question
                and "B." not in question
                and "C." not in question
                and "D." not in question
            ):
                continue
            question = question.replace(", True Answer:", "")
            question = question.replace("Answer:", "")
            list_raw_question.append(question.strip())
            difficulty.append(original_difficulty[i])
        question_dataset = Dataset.from_dict(
            {"text": list_raw_question, "difficulty": difficulty}
        )
    else:
        question_dataset = question_dataset.rename_column("raw_question", "text")
        question_dataset = question_dataset.add_column("difficulty", difficulty)

    # Add `difficulty` column
    dataset = question_dataset.map(
        process_text, input_columns=["text", "difficulty"], num_proc=args.num_workers
    )

    # Drop all columns except `text`, `difficulty`
    all_columns = dataset.column_names
    all_columns.remove("text")
    all_columns.remove("difficulty")
    dataset = dataset.remove_columns(all_columns)

    # Split and push to hub
    dataset_dict = dataset.train_test_split(test_size=0.2)
    dataset_str = args.dataset.replace("/", "_")
    dataset_dict.push_to_hub(
        "stair-lab/reeval-sft", f"{dataset_str}_{model_short_name}"
    )
