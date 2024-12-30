import argparse
import pickle

import numpy as np
import pandas as pd
from datasets import Dataset
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
from utils.constants import DATASET_FOLDER, DESCRIPTION_MAP
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
    description = DESCRIPTION_MAP[args.dataset]

    # Load the difficulty scores
    item_parms_folder = snapshot_download(
        repo_id=f"stair-lab/reeval_results",
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
    train_indices = pickle.load(
        open(
            f"{item_parms_folder}/{args.dataset}/s42_{args.fitting_method}_{args.PL}pl_1d_nl1/train_question_indices.pkl",
            "rb",
        )
    )
    question_df = question_df.iloc[train_indices]
    question_dataset = Dataset.from_pandas(question_df)

    # Ensure the length of the dataset and difficulty scores are the same
    assert len(question_dataset) == len(difficulty)

    # Define the chat template
    sft_chat = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": (
                """Generate a question with a given difficulty score, which range from -5 to 5. """
                """The lower the score is, the more difficult the question is. """
                """Hence a model is more likely to fail the questions. """
                """Output only the question and nothing else. """
                f"""Dataset description: {description}. """
                """Difficulty: %s. Question: """
            ),
        },
        {"role": "assistant", "content": """%s"""},
    ]
    template = tokenizer.apply_chat_template(
        sft_chat, tokenize=False, add_generation_prompt=False
    )

    # Process the text
    def process_text(text, difficulty):
        return {"text": template % (round(difficulty, 2), text)}

    # Rename `prompt` to  `text`
    question_dataset = question_dataset.rename_column("prompt", "text")

    # Add `difficulty` column
    question_dataset = question_dataset.add_column("difficulty", difficulty)
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
        f"stair-lab/reeval-sft", f"{dataset_str}_{model_short_name}"
    )
