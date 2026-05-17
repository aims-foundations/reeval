import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import GenerationConfig
from vllm import LLM, SamplingParams


def find_common_prefix(strings):
    """Find the longest common prefix shared by all strings in a group."""
    if not strings:
        return ""
    common_prefix = strings[0]
    for s in strings[1:]:
        while not s.startswith(common_prefix) and common_prefix:
            common_prefix = common_prefix[:-1]
    return common_prefix


def detect_subgroups(strings, threshold=0.6):
    """
    Group strings such that each subgroup's common prefix (updated as strings are added)
    has a length that is at least `threshold` fraction of the shorter string compared.
    """
    if not strings:
        return []

    sorted_strings = sorted(strings)
    clusters = []
    current_cluster = [sorted_strings[0]]
    current_prefix = sorted_strings[0]

    for s in sorted_strings[1:]:
        new_common_prefix = find_common_prefix([current_prefix, s])
        min_length = min(len(current_prefix), len(s))
        ratio = len(new_common_prefix) / min_length if min_length > 0 else 0

        if ratio >= threshold:
            current_cluster.append(s)
            current_prefix = new_common_prefix
        else:
            clusters.append(current_cluster)
            current_cluster = [s]
            current_prefix = s
    clusters.append(current_cluster)
    return clusters


def create_batches(prompts, token_lengths, max_tokens_per_batch):
    """
    Create batches such that each batch contains at most max_batch_size prompts
    and the total token count in the batch does not exceed max_tokens_per_batch.
    Prompts that individually exceed max_tokens_per_batch are skipped.
    """
    batches = []
    current_batch = []
    current_tokens = 0

    for prompt, token_count in tqdm(zip(prompts, token_lengths), total=len(prompts)):
        if current_batch and current_tokens + token_count > max_tokens_per_batch:
            batches.append(current_batch)
            current_batch = []
            current_tokens = 0

        current_batch.append(prompt)
        current_tokens += token_count

    if current_batch:
        batches.append(current_batch)
    return batches


model_name = "meta-llama/Llama-3.1-8B-Instruct"
checkpoint = "results_perplexity_forthattempt.pkl"
max_token_length = 2048
vllm_model = LLM(
    model_name,
    gpu_memory_utilization=0.9,
    enable_chunked_prefill=False,
    enforce_eager=True,
    dtype=torch.float16,
    swap_space=32,
    max_num_seqs=128,
    tensor_parallel_size=8,
)
tokenizer = vllm_model.get_tokenizer()
generation_config = GenerationConfig.from_pretrained(model_name)
sampling_params = SamplingParams(n=1, temperature=1, max_tokens=1, prompt_logprobs=0)

if os.path.exists(checkpoint):
    results = pd.read_pickle(checkpoint)
    batch_start = int(open(checkpoint + ".batch_count").read())
else:
    results = pd.read_pickle("../gather_helm_data/helm_tables/responses.pkl")
    for group in tqdm(results["groups"].unique()):
        # Preserve order while removing duplicates
        example_strings = results.loc[
            results["groups"] == group, "request.prompt"
        ].to_list()
        unique_example_strings = []
        seen = set()
        for s in example_strings:
            if s not in seen:
                unique_example_strings.append(s)
                seen.add(s)

        subgroups = detect_subgroups(unique_example_strings, threshold=0.8)
        mapping = {}
        mapping_i = {}
        for subgroup in subgroups:
            if len(subgroup) == 1:
                # Singleton subgroup: no common prefix exists, so don't strip anything.
                s = subgroup[0]
                mapping[s] = s
                mapping_i[s] = 0
            else:
                common_prefix = find_common_prefix(subgroup)
                for s in subgroup:
                    mapping[s] = s[len(common_prefix) :].strip()
                    mapping_i[s] = len(common_prefix)

        # Create a boolean mask for the current group and assign mapped values
        mask = results["groups"] == group
        results.loc[mask, "question_start_index"] = results.loc[
            mask, "request.prompt"
        ].map(mapping_i)

    results["perplexity"] = np.nan
    results["question_start_index"] = results["question_start_index"].astype(int)
    batch_start = 0

    unique_prompts = list(results["request.prompt"].unique())
    encoded = tokenizer(unique_prompts, add_special_tokens=False)
    lengths = [len(ids) for ids in encoded["input_ids"]]
    unique_prompt_to_length = dict(zip(unique_prompts, lengths))
    results["token_length"] = results["request.prompt"].map(unique_prompt_to_length)

    # Precompute the token index corresponding to the question portion.
    # This is computed as the number of tokens in the substring up to the question_start_index.
    unique_mapping = results[
        ["request.prompt", "question_start_index"]
    ].drop_duplicates(subset="request.prompt")
    prompt_to_q_index = dict(
        zip(unique_mapping["request.prompt"], unique_mapping["question_start_index"])
    )

    # For each unique prompt, compute its token index (number of tokens in prompt[:question_start_index])
    question_token_indices = {
        prompt: len(tokenizer.encode(prompt[:q_index]))
        for prompt, q_index in prompt_to_q_index.items()
    }
    results["question_token_index"] = results["request.prompt"].map(
        question_token_indices
    )

    results.to_pickle(checkpoint)
    with open(checkpoint + ".batch_count", "w") as f:
        f.write(str(batch_start))

results = results.dropna(subset=["dicho_score"])
filtered = results.loc[
    results["token_length"] < 2048, ["request.prompt", "token_length"]
]
filtered = filtered.drop_duplicates(subset="request.prompt")
filtered = filtered.sort_values("token_length", ascending=False)
unique_prompts = filtered["request.prompt"].tolist()
unique_token_lengths = filtered["token_length"].tolist()

print("Start creating batches")
batches = create_batches(unique_prompts, unique_token_lengths, max_token_length)
existing = results.loc[
    pd.notna(results["perplexity"]), ["request.prompt", "perplexity"]
]
prompt_to_perp = dict(zip(existing["request.prompt"], existing["perplexity"]))

unique_mapping = results[["request.prompt", "question_token_index"]].drop_duplicates(
    subset="request.prompt"
)
question_token_indices = dict(
    zip(unique_mapping["request.prompt"], unique_mapping["question_token_index"])
)

print(f"fraction of filtered prompts: {len(unique_prompts) / len(unique_mapping):.2%}")

skip_count = 0
for i, batch_prompts in tqdm(
    enumerate(batches[batch_start:]), total=len(batches[batch_start:])
):
    batch_outputs = vllm_model.generate(batch_prompts, sampling_params, use_tqdm=False)
    for prompt, output in zip(batch_prompts, batch_outputs):
        token_index = question_token_indices[prompt]  # 1
        logprobs_list = output.prompt_logprobs[token_index:]
        target_logprobs = [
            list(logprobs.values())[0].logprob for logprobs in logprobs_list
        ]
        if target_logprobs:
            prompt_to_perp[prompt] = np.exp(-np.mean(target_logprobs))
        else:
            skip_count += 1

    if (i > 0 and i % 10000 == 0) or i == len(batches) - 1:
        print(f"progress {i} / {len(batches)}, skipped {skip_count} items so far")
        results["perplexity"] = (
            results["request.prompt"].map(prompt_to_perp).astype(float)
        )
        results.to_pickle(checkpoint)
        with open(checkpoint + ".batch_count", "w") as f:
            f.write(str(i))
