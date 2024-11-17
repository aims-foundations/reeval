import argparse
import os
import pickle

import numpy as np
import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv
from huggingface_hub import login, snapshot_download
from transformers import AutoTokenizer
from utils.constants import DESCRIPTION_MAP


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--fitting_method", type=str, default="mle")
    parser.add_argument("--PL", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--model", type=str, default="bayridge", choices=["bayridge", "mlp"]
    )
    args = parser.parse_args()

    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    login(token=hf_token)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

    description = DESCRIPTION_MAP[args.dataset]

    # Load the question dataset
    data_folder = snapshot_download(
        repo_id="stair-lab/reeval_responses", repo_type="dataset"
    )
    question_info_df = pd.read_csv(f"{data_folder}/{args.dataset}/search.csv")
    question_df = question_info_df.loc[
        question_info_df["is_deleted"] != 1, ["text"]
    ].reset_index(drop=True)
    question_dataset = Dataset.from_pandas(question_df)

    # Load the difficulty scores
    item_parms_folder = snapshot_download(
        repo_id=f"stair-lab/reeval_{args.fitting_method}_calibration",
        repo_type="dataset",
    )
    item_parms = pickle.load(
        open(f"{item_parms_folder}/{args.PL}pl/{args.dataset}/item_parms.pkl", "rb")
    )
    # >>> n_questions x (3 + D)
    difficulty = np.array(item_parms)[:, 0].tolist()

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

    # Add `difficulty` column
    question_dataset = question_dataset.add_column("difficulty", difficulty)
    dataset = question_dataset.map(
        process_text, input_columns=["text", "difficulty"], num_proc=args.num_workers
    )

    # Split and push to hub
    dataset_dict = dataset.train_test_split(test_size=0.2)
    dataset_dict.push_to_hub("stair-lab/reeval-sft", args.dataset)
