import pickle
import re
from lampo.reward_model import RewardModelTemplate
from embed_text_package.embed_text import Embedder
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

class MessageDataset(Dataset):
    def __init__(self, messages):
        self.messages = {"question_text": messages}

    def __len__(self):
        return len(self.data["question_text"])

    def __getitem__(self, idx):
        return {"question_text": self.data["question_text"][idx]}
    
def extract_score(input_str: str) -> int:
    match = re.search(r'Your task is to output a prompt at score (\d+)', input_str)
    return int(match.group(1))

class MyRewardModel(RewardModelTemplate):
    def __init__(self, config):
        self.model = None
        self.load()

    async def compute(self, messages):
        objective_scores = [extract_score(m) for m in messages]
        
        dataset = MessageDataset(messages)
        model_name = "meta-llama/Meta-Llama-3-8B"
        cols_to_be_embded = ['question_text']
        embdr = Embedder()
        embdr.load(model_name)
        dataloader = DataLoader(dataset, batch_size=len(messages))
        emb = embdr.get_embeddings(
            dataloader, model_name, cols_to_be_embded
        )
        X = emb['question_text']
        scores = self.model.predict(X).tolist()

        rewards = [-abs(a - b) for a, b in zip(scores, objective_scores)]
        return rewards

    def load(self):
        with open('../../data/real/ppo/bayesian_ridge_model.pkl', 'rb') as f:
            self.model = pickle.load(f)

    def unload(self):
        pass