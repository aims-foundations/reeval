from embed_text_package.embed_text import Embedder

model_name = "<HF_repo>/<HF_model>"
embdr = Embedder()
embdr.load(model_name)
emb = embdr.get_embeddings(dataloader, MODEL_NAME, cols_to_be_embded)