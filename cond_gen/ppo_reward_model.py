import pickle
import re
import pandas as pd
from lampo.reward_model import RewardModelTemplate
from datasets import Dataset
from embed_text_package.embed_text import Embedder
from torch.utils.data import DataLoader
from utils import MLP

def extract_score(input_str: str) -> float:
    match = re.search(r'Difficulty: ([-+]?\d*\.\d+|\d+)', input_str)
    return float(match.group(1))

class MyRewardModel(RewardModelTemplate):
    def __init__(self, config):
        self.reg_model = None
        self.emb_model = None
        self.load()

    async def compute(self, messages): # messages: List[list[qstr,astr], list[qstr,astr]]
        print(f"messages[0][0]: {messages[0][0]}")
        print(f"messages[0][1]: {messages[0][1]}")
        print(f"len(messages): {len(messages)}")
        gt_scores = [extract_score(m[0]) for m in messages]
        
        answers = [m[1] for m in messages]
        answer_df = pd.DataFrame(answers, columns=["text"])
        answer_dataset = Dataset.from_pandas(answer_df)
        
        bs = len(answers)
        cols_to_be_embded = ['text']
        model_name="meta-llama/Meta-Llama-3-8B"
        
        dataloader = DataLoader(answer_dataset, batch_size=bs)
        emb = self.emb_model.get_embeddings(
            dataloader, model_name, cols_to_be_embded
        )
        
        answer_embs = emb['text']
        pred_scores = self.reg_model.predict(answer_embs).tolist()
        
        rewards = [-abs(a-b) for a, b in zip(pred_scores, gt_scores)]
        print(f"gt scores: {gt_scores}")
        print(f"pred scores: {pred_scores}")
        print(f"reward scores: {rewards}")
        
        return rewards
    
    def load(self,):
        with open('../data/plugin_regression/airbench/bayridge.pkl', 'rb') as f:
            self.reg_model = pickle.load(f)

        model_name="meta-llama/Meta-Llama-3-8B"
        embdr = Embedder()
        embdr.load(model_name)
        self.emb_model = embdr 

    def unload(self):
        pass