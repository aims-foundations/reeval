import pickle
import re
import pandas as pd
from lampo.reward_model import RewardModelTemplate
from datasets import Dataset
from utils import get_embed

def extract_score(input_str: str) -> float:
    match = re.search(r'Difficulty: ([-+]?\d*\.\d+|\d+)', input_str)
    return float(match.group(1))

class MyRewardModel(RewardModelTemplate):
    def __init__(self, config):
        self.model = None
        self.load()

    async def compute(self, messages):
        gt_scores = [extract_score(m[0]) for m in messages]
        
        answers = [m[1] for m in messages]
        answer_df = pd.DataFrame(answers, columns=["text"])
        answer_dataset = Dataset.from_pandas(answer_df)
        answer_embs = get_embed(answer_dataset)
        pred_scores = self.model.predict(answer_embs).tolist()
        
        rewards = [-abs(a - b) for a, b in zip(pred_scores, gt_scores)]
        return rewards
    
    def load(self):
        with open('../data/plugin_regression/airbench/bayridge.pkl', 'rb') as f:
            self.model = pickle.load(f)

    def unload(self):
        pass