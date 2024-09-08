import pickle
from lampo.reward_model import RewardModelTemplate
from embed_text_package.embed_text import Embedder
from torch.utils.data import DataLoader
import torch
from torch.utils.data import Dataset

class MessageDataset(Dataset):
    def __init__(self, messages):
        self.messages = {"question_text": messages}

    def __len__(self):
        return len(self.data["question_text"])

    def __getitem__(self, idx):
        return {"question_text": self.data["question_text"][idx]}

class MyRewardModel(RewardModelTemplate):
    def __init__(self, config):
        self.model = None
        self.load()

    async def compute(self, messages):
        """
        It receives a list of messages (strings) and returns a list of scores 
        """
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
        
        scores = self.model.predict(X)
        return scores.tolist()

    def load(self):
        with open('../data/real/auto_gen/bayesian_ridge_model.pkl', 'rb') as f:
            self.model = pickle.load(f)

    def unload(self):
        pass