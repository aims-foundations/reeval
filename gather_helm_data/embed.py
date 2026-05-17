from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from transformers import GPT2TokenizerFast
from vllm import LLM


def embed_sub_batch(idx, gpu_id, sub_batch, model_name):
    llm = LLM(
        model=model_name,
        task="embed",
        gpu_memory_utilization=0.95,
        # enable_chunked_prefill=False,
        # enforce_eager=True,
        tensor_parallel_size=1,
        # tokenizer_pool_size=8,
        # tokenizer_mode="mistral", # use this when querying Mistral
        device=f"cuda:{gpu_id}",
    )
    outputs = llm.embed(sub_batch)
    embs = [o.outputs.embedding for o in outputs]
    return idx, embs


if __name__ == "__main__":
    gpu_ids = [1, 2, 3]
    num_gpus = len(gpu_ids)
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    # When use Mistral to do embedding, need tokenizer_mode="mistral" and vllm==0.8.1
    # model_name = "mistralai/Mistral-7B-Instruct-v0.3"

    # Load and filter prompts
    df = pd.read_pickle("../data/long.pkl")
    df = df.dropna(subset=["dicho_score", "input.text"])
    unique_df = df.drop_duplicates(subset="input.text").reset_index(drop=True)
    print(unique_df["scenario"].unique())
    print(len(unique_df["scenario"].unique()))
    unique_df["input.text"] = unique_df["input.text"].astype(str)
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    encodings = tokenizer(
        unique_df["input.text"].tolist(),
        add_special_tokens=False,
        return_length=True,
    )
    unique_df["token_length"] = encodings["length"]
    filtered = unique_df.loc[
        unique_df["token_length"] < 2048, ["input.text", "token_length"]
    ].sort_values("token_length", ascending=False)
    unique_prompts = filtered["input.text"].tolist()
    print(len(unique_prompts))

    # Split into sub‐batches for each GPU
    sub_batches = [unique_prompts[j::num_gpus] for j in range(num_gpus)]

    # Parallel embedding with index reconstruction
    ordered_embs = [None] * len(unique_prompts)
    with ProcessPoolExecutor(max_workers=num_gpus) as executor:
        tasks = [
            executor.submit(embed_sub_batch, j, gpu_id, sb, model_name)
            for j, (gpu_id, sb) in enumerate(zip(gpu_ids, sub_batches))
        ]
        for future in as_completed(tasks):
            j, embs = future.result()
            for k, emb in enumerate(embs):
                orig_idx = j + k * num_gpus
                ordered_embs[orig_idx] = emb

    # save result
    question_embedding = pd.DataFrame(
        {
            "question": unique_prompts,
            "embedding": ordered_embs,
        }
    )
    safe_name = model_name.replace("/", "_")
    out_path = Path("../data") / f"embed_{safe_name}.pkl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    question_embedding.to_pickle(out_path)
