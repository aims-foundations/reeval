from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm
from vllm import LLM

# List of GPU ids to use (adjust based on your available GPUs)
gpu_ids = [0, 1, 2, 4, 6, 7]  # This gives you 5 models (update if needed)
num_models = len(gpu_ids)
# model_name = "meta-llama/Llama-3.1-8B-Instruct"
model_name = "Alibaba-NLP/gte-Qwen2-7B-instruct"
# model_name = "mistralai/Mistral-7B-Instruct-v0.3"


def embed_sub_batch(gpu_id, sub_batch):
    # Reinitialize the model in this process on the specified GPU.
    llm = LLM(
        model=model_name,
        task="embed",
        gpu_memory_utilization=0.9,
        # enable_chunked_prefill=False,
        # enforce_eager=True,
        tensor_parallel_size=1,
        # tokenizer_pool_size=8,
        device=f"cuda:{gpu_id}",
    )
    outputs = llm.embed(sub_batch)
    # Extract and return the embedding from each output object.
    return [o.outputs.embedding for o in outputs]


results = pd.read_pickle("results_perplexity_forthattempt.pkl")
results = results.dropna(subset=["dicho_score"])
filtered = results.loc[
    results["token_length"] < 2048, ["request.prompt", "token_length"]
]
filtered = filtered.drop_duplicates(subset="request.prompt")
filtered = filtered.sort_values("token_length", ascending=False)
unique_prompts = filtered["request.prompt"].tolist()

outputs = []
batch_size = len(unique_prompts)

for i in tqdm(range(0, len(unique_prompts), batch_size)):
    batch = unique_prompts[i : i + batch_size]
    # Split the batch into sub-batches for each GPU.
    # This slicing method works even if len(batch) isn’t divisible by num_models.
    sub_batches = [batch[j::num_models] for j in range(num_models)]

    futures = []
    with ProcessPoolExecutor(max_workers=num_models) as executor:
        for gpu_id, sub_batch in zip(gpu_ids, sub_batches):
            if sub_batch:  # Only submit if there’s work.
                futures.append(executor.submit(embed_sub_batch, gpu_id, sub_batch))
        for future in as_completed(futures):
            outputs.extend(future.result())

    # Optionally, save partial results to disk.
    question_embedding = pd.DataFrame(
        {"question": unique_prompts[: len(outputs)], "embedding": outputs}
    )
    # question_embedding.to_pickle(f"unique_prompts_embeddings_Mistral-7B-Instruct-v0.3.pkl")
    question_embedding.to_pickle("unique_prompts_embeddings_gte-Qwen2-7B-instruct.pkl")
