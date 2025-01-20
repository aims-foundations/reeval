import argparse
import csv
import gc

import io
import os
import pickle

import pandas as pd
import torch
from datasets import load_dataset
from huggingface_hub import HfApi, snapshot_download
from ppo_reward_model import extract_score
from transformers import AutoTokenizer, GenerationConfig
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel

DEFAULT_CHAT_TEMPLATE = """{{- bos_token }}
{#- System message #}
{%- if messages[0]['role'] == 'system' %}
{%- set system_message = messages[0]['content']|trim %}
{%- set messages = messages[1:] %}
{%- else %}
{%- set system_message = \"\" %}
{%- endif %}
{{- \"SYSTEM:\\n\" }}
{{- system_message }}
{{- \"END_SYSTEM\n\" }}

{%- for message in messages %}
{%- if message['role'] in ['user', 'assistant'] %}
{{- message['role'].upper() + ':\\n' + message['content'] | trim + '\\nEND_' + message['role'].upper() + '\\n' }}
{%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
{{- 'ASISSTANT:\\n' }}
{%- endif %}"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--question_generator", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="air-bench/air_bench_2024")
    parser.add_argument("--force_run", action="store_true")
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--skip_openai", action="store_true")
    args = parser.parse_args()

    upload_api = HfApi()

    generator_short_name = args.question_generator.split("/")[-1]
    generator_short_name = generator_short_name.replace("reeval_", "")
    ds_short_name = args.dataset.replace("/", "_")

    output_dir = f"../data/sft_analysis/{ds_short_name}_{generator_short_name}"
    os.makedirs(output_dir, exist_ok=True)

    # Load list of model keys
    # data_folder = snapshot_download(
    #     repo_id="stair-lab/reeval_responses", repo_type="dataset"
    # )
    # model_keys = pd.read_csv(f"{data_folder}/{args.dataset}/model_keys.csv")
    available_models = pd.read_csv("./configs/model_hf_id.csv")

    # Load prompts
    if args.smoke_test:
        test_question_df = pd.DataFrame({"text": ["What is the capital of US?"]})
        gt_difficulties = [0.5]
    else:
        generated_questions_folder = snapshot_download(
            repo_id="stair-lab/reeval_generated_questions", repo_type="dataset"
        )
        test_question_df = pd.read_csv(
            f"{generated_questions_folder}/sft/{ds_short_name}_{generator_short_name}/train_answers_filtered.csv"
        )  # 1000
        #### BUG????#### test_dataset = load_dataset(f"stair-lab/{args.dataset}-ppo", split="test")
        test_dataset = load_dataset(
            f"stair-lab/reeval-ppo",
            f"{ds_short_name}_{generator_short_name}",
            split="train",
        )
        test_texts = test_dataset["text"][: len(test_question_df)]
        gt_difficulties = [extract_score(p) for p in test_texts]

    # Iterate over models
    # for ri, row in available_models.iterrows():
    #     model_name = row["huggingface_model_id"]
    #     # Check if model name is nan
    #     if pd.isna(model_name):
    #         continue

    #     # Check model_name in available_models
    #     if model_name not in available_models["huggingface_model_id"].values:
    #         continue

    model_name = args.model

    if args.smoke_test and model_name != "meta-llama/Llama-3.1-8B-Instruct":
        exit(0)

    if args.skip_openai and "openai" in model_name:
        exit(0)

    if not args.force_run and os.path.exists(
        f"{output_dir}/{model_name.replace('/', '_')}.csv"
    ):
        exit(0)

    # Clear memory
    torch.cuda.empty_cache()
    gc.collect()

    # Try to load generation config
    try:
        generation_config = GenerationConfig.from_pretrained(model_name)
        sampling_params = SamplingParams(
            temperature=generation_config.temperature,
            top_p=generation_config.top_p,
            max_tokens=256,
            stop_token_ids=generation_config.eos_token_id,
        )
    except:
        # Use default sampling params
        sampling_params = SamplingParams(
            max_tokens=256,
        )

    # Load model
    llm = LLM(
        model=model_name,
        gpu_memory_utilization=0.9,
        tensor_parallel_size=torch.cuda.device_count(),
        dtype=torch.float16,
        trust_remote_code=True,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.chat_template is None:
        # In case the model does not have a chat template, create a default one
        tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE

    ########################################################
    # Generate outputs
    # Format prompts using tokenizer applied-t
    test_prompts = []
    for i, row in test_question_df.iterrows():
        conver = [{"role": "user", "content": row["text"]}]

        test_prompt = tokenizer.apply_chat_template(
            conver, tokenize=False, add_generation_prompt=True
        )
        test_prompts.append(test_prompt)

    outputs = llm.generate(test_prompts, sampling_params)
    answers = [o.outputs[0].text for o in outputs]
    ########################################################

    # Save results
    results = pd.DataFrame(
        {
            "difficulty": gt_difficulties,
            "answer": answers,
        }
    )
    model_name = model_name.replace("/", "_")
    pickle.dump(results, open(f"{output_dir}/{model_name}.pkl", "wb"))

    results_file = io.BytesIO()
    results.to_csv(results_file, index=False)
    upload_api.upload_file(
        repo_id="stair-lab/reeval_generated_questions",
        repo_type="dataset",
        path_in_repo=f"sft/{ds_short_name}_{generator_short_name}/{model_name}.csv",
        path_or_fileobj=results_file,
    )
    results.to_csv(
        f"{output_dir}/{model_name}.csv",
        index=False,
        quoting=csv.QUOTE_NONE,
        escapechar="\\",
    )

    # Delete model
    destroy_model_parallel()
    del llm.llm_engine.model_executor.driver_worker
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    torch.distributed.destroy_process_group()
    print("Successfully delete the llm pipeline and free the GPU memory!")
