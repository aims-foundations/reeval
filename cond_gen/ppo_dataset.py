import argparse
import os

import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer
from utils.constants import DESCRIPTION_MAP
from utils.utils import set_seed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--ppo_size", type=int, default=1250)
    parser.add_argument("--mocktest", action="store_true")
    args = parser.parse_args()

    set_seed(42)
    model_short_name = args.model.split("/")[-1]
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    ppo_chat = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": (
                """Generate a question with a given difficulty score, which range from -5 to 5. """
                """The lower the score is, the more difficult the question is. """
                """Hence a model is more likely to fail the questions. """
                """Output only the question and nothing else.\n"""
                """Dataset description: {description}.\n"""
                """Difficulty: {difficulty}."""
            ),
        },
    ]
    template = tokenizer.apply_chat_template(
        ppo_chat, tokenize=False, add_generation_prompt=True
    )

    texts = []
    description = DESCRIPTION_MAP[args.dataset]
    if args.mocktest:
        difficulties = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
    for i in range(args.ppo_size):
        if args.mocktest:
            difficulty = difficulties[i % len(difficulties)]
        else:
            difficulty = np.random.normal(0, 1)
        texts.append(
            template.format(description=description, difficulty=round(difficulty, 2))
        )

    dataset = Dataset.from_dict({"text": texts})

    # Split and push to hub
    if args.mocktest:
        dataset_dict = dataset
    else:
        dataset_dict = dataset.train_test_split(test_size=0.2)
    dataset_str = args.dataset.replace("/", "_")
    if args.mocktest:
        dataset_dict.push_to_hub(
            f"stair-lab/reeval-ppo-mocktest", f"{dataset_str}_{model_short_name}"
        )
    else:
        dataset_dict.push_to_hub(
            f"stair-lab/reeval-ppo", f"{dataset_str}_{model_short_name}"
        )
