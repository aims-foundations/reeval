from embed_text_package.embed_text import Embedder
from torch.utils.data import DataLoader
from datasets import load_dataset

if __name__ == "__main__":
    dataset = load_dataset("stair-lab/airbench-difficulty", split="whole")
    cols_to_be_embded = ['question_text']
    bs = 1024
    model_name = "meta-llama/Meta-Llama-3-8B"
    
    embdr = Embedder()
    embdr.load(model_name)
    dataloader = DataLoader(dataset.with_format("torch"), bs)
    emb = embdr.get_embeddings(
        dataloader, model_name, cols_to_be_embded
    )