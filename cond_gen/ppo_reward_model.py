import pickle
import re

import pandas as pd

import torch
from datasets import Dataset
from embed_text_package.embed_text_v2 import Embedder
from huggingface_hub import snapshot_download
from lampo.reward_model import RewardModelTemplate
from torch.utils.data import DataLoader

MODEL_NAME = "meta-llama/Meta-Llama-3-8B"
BATCH_SIZE = 4


def extract_score(input_str: str) -> float:
    match = re.search(r"Difficulty: ([-+]?\d*\.\d+|\d+)", input_str)
    return float(match.group(1))


def extract_dataset_description(input_str: str) -> str:
    start_idx = input_str.find("### DATASET:")
    end_idx = input_str.find("Difficulty:")
    return input_str[start_idx:end_idx].strip()


def format_input(dataset_desc, input_str):
    return dataset_desc + ", ### PROMPT: " + input_str


class MyRewardModel(RewardModelTemplate):
    def __init__(self, config):
        self.reg_model = None
        self.emb_model = None
        self.load()

    async def compute(
        self, messages
    ):  # messages: List[list[qstr,astr], list[qstr,astr]]
        print(f"messages[0][0]: {messages[0][0]}")
        print(f"messages[0][1]: {messages[0][1]}")
        print(f"len(messages): {len(messages)}")
        gt_scores = [extract_score(m[0]) for m in messages]
        ds_descs = [extract_dataset_description(m[0]) for m in messages]

        answers = [
            format_input(ds_desc, m[1]) for ds_desc, m in zip(ds_descs, messages)
        ]
        answer_dataset = Dataset.from_dict({"text": answers})

        dataloader = DataLoader(answer_dataset, batch_size=BATCH_SIZE)
        emb = self.emb_model.get_embeddings(dataloader, MODEL_NAME, ["text"])

        pred_scores = self.difficulty_predictor(emb["text"]).tolist()
        rewards = [-abs(a - b) for a, b in zip(pred_scores, gt_scores)]

        print(f"gt scores: {gt_scores}")
        print(f"pred scores: {pred_scores}")
        print(f"reward scores: {rewards}")

        return rewards

    def load(self):
        result_folder = snapshot_download(
            repo_id="stair-lab/reeval_results", repo_type="dataset"
        )

        with open(
            f"{result_folder}/combined_data/s42_em_1pl_1d_aq/item_parameters_nn.pkl",
            "rb",
        ) as f:
            self.reward_model = pickle.load(f)

        self.difficulty_predictor = lambda x: self.reward_model(x)[:, 0]

        print("Loaded embedding model")
        self.emb_model = Embedder()
        self.emb_model.load(
            MODEL_NAME,
            # tensor_parallel_size=torch.cuda.device_count(),
            dtype=torch.float16,
        )

    def unload(self):
        pass
