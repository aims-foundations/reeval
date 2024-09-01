from embed_text_package.embed_text import Embedder
from datasets import load_dataset

if __name__ == "__main__":
    dataset = load_dataset("stair-lab/airbench-difficulty")
    print(dataset)

    # model_name = "<HF_repo>/<HF_model>"
    # embdr = Embedder()
    # embdr.load(model_name)
    # emb = embdr.get_embeddings(dataloader, MODEL_NAME, cols_to_be_embded)