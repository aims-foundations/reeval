import argparse
import os

import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer
from utils.constants import DESCRIPTION_MAP

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--ppo_size", type=int, default=1250)
    args = parser.parse_args()
    model_short_name = args.model.split("/")[-1]
    if model_short_name == "Meta-Llama-3.1-8B-Instruct":
        model_short_name = ""
    else:
        model_short_name = "-" + model_short_name
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    ppo_chat = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": (
                """Generate a question with a given difficulty score, which range from -5 to 5. """
                """The lower the score is, the more difficult the question is. """
                """Hence a model is more likely to fail the questions. """
                """Output only the question and nothing else. """
                """Dataset description: %s. """
                """Difficulty: %s. Question: """
            ),
        },
    ]
    template = tokenizer.apply_chat_template(
        ppo_chat, tokenize=False, add_generation_prompt=True
    )

    texts = []
    description = DESCRIPTION_MAP[args.dataset]
    for i in range(args.ppo_size):
        difficulty = np.random.normal(0, 1)
        texts.append(template % (description, round(difficulty, 2)))

    dataset = Dataset.from_dict({"text": texts})

    # Split and push to hub
    dataset_dict = dataset.train_test_split(test_size=0.2)
    dataset_dict.push_to_hub(f"stair-lab/reeval{model_short_name}-ppo", args.dataset)
    # stair-lab/reeval-Mistral-7B-Instruct-v0.3-ppo
    # stair-lab/reeval-ppo