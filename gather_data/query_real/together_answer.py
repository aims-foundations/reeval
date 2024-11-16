import csv
import os

import datasets
from bacher import Batcher
from dotenv import load_dotenv
from huggingface_hub import login

if __name__ == "__main__":
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    together_key = os.getenv("TOGETHERAI_KEY")

    output_dir = "../../../gather_data/query_real/answer"
    os.makedirs(output_dir, exist_ok=True)

    login(token=hf_token)
    airbench = datasets.load_dataset("stanford-crfm/air-bench-2024", split="test")

    model_strings = [
        # "zero-one-ai/Yi-34B-Chat",
        "Austism/chronos-hermes-13b",
        "cognitivecomputations/dolphin-2.5-mixtral-8x7b",
        # "databricks/dbrx-instruct",
        "deepseek-ai/deepseek-coder-33b-instruct",
        # "deepseek-ai/deepseek-llm-67b-chat",
        "garage-bAInd/Platypus2-70B-instruct",
        "google/gemma-2b-it",
        "google/gemma-7b-it",
        "Gryphe/MythoMax-L2-13b",
        "lmsys/vicuna-13b-v1.5",
        "lmsys/vicuna-7b-v1.5",
        "codellama/CodeLlama-13b-Instruct-hf",
        "codellama/CodeLlama-34b-Instruct-hf",
        "codellama/CodeLlama-70b-Instruct-hf",
        "codellama/CodeLlama-7b-Instruct-hf",
        "meta-llama/Llama-2-70b-chat-hf",
        "meta-llama/Llama-2-13b-chat-hf",
        "meta-llama/Llama-2-7b-chat-hf",
        # "meta-llama/Llama-3-8b-chat-hf",
        # "meta-llama/Llama-3-70b-chat-hf",
        "mistralai/Mistral-7B-Instruct-v0.1",
        "mistralai/Mistral-7B-Instruct-v0.2",
        # "mistralai/Mistral-7B-Instruct-v0.3",
        # "mistralai/Mixtral-8x7B-Instruct-v0.1",
        # "mistralai/Mixtral-8x22B-Instruct-v0.1",
        "NousResearch/Nous-Capybara-7B-V1p9",
        "NousResearch/Nous-Hermes-2-Mistral-7B-DPO",
        "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
        "NousResearch/Nous-Hermes-2-Mixtral-8x7B-SFT",
        "NousResearch/Nous-Hermes-llama-2-7b",
        "NousResearch/Nous-Hermes-Llama2-13b",
        "NousResearch/Nous-Hermes-2-Yi-34B",
        "openchat/openchat-3.5-1210",
        "Open-Orca/Mistral-7B-OpenOrca",
        "Qwen/Qwen1.5-0.5B-Chat",
        "Qwen/Qwen1.5-1.8B-Chat",
        "Qwen/Qwen1.5-4B-Chat",
        "Qwen/Qwen1.5-7B-Chat",
        "Qwen/Qwen1.5-14B-Chat",
        "Qwen/Qwen1.5-32B-Chat",
        # "Qwen/Qwen1.5-72B-Chat",
        "Qwen/Qwen1.5-110B-Chat",
        "Qwen/Qwen2-72B-Instruct",
        "snorkelai/Snorkel-Mistral-PairRM-DPO",
        "Snowflake/snowflake-arctic-instruct",
        ### "togethercomputer/alpaca-7b",
        "teknium/OpenHermes-2-Mistral-7B",
        "teknium/OpenHermes-2p5-Mistral-7B",
        ### "togethercomputer/Llama-2-7B-32K-Instruct",
        ### "togethercomputer/RedPajama-INCITE-Chat-3B-v1",
        ### "togethercomputer/RedPajama-INCITE-7B-Chat",
        "togethercomputer/StripedHyena-Nous-7B",
        ### "Undi95/ReMM-SLERP-L2-13B",
        "Undi95/Toppy-M-7B",
        "WizardLM/WizardLM-13B-V1.2",
        "upstage/SOLAR-10.7B-Instruct-v1.0",
    ]
    print(len(model_strings))

    for model_string in model_strings:
        print(model_string)

        batcher = Batcher(
            api_name="together",
            api_key=together_key,
            model_name=model_string,
            temperature=0,
            num_workers=64,
            max_token=128,
        )

        row_list = []
        question_list = []
        for i in range(len(airbench["cate-idx"])):
            row_list.append(
                [
                    airbench[i]["cate-idx"],
                    airbench[i]["l2-name"],
                    airbench[i]["l3-name"],
                    airbench[i]["l4-name"],
                    airbench[i]["prompt"],
                ]
            )
            question_list.append(airbench[i]["prompt"])

        result_list = batcher.handle_message_list(question_list)

        model_string = model_string.replace("/", "_")
        with open(
            f"{output_dir}/answer_{model_string}_result.csv",
            "w",
            newline="",
            encoding="utf-8",
        ) as outfile:
            writer = csv.writer(outfile)
            writer.writerow(
                ["cate-idx", "l2-name", "l3-name", "l4-name", "prompt", "response"]
            )

            for i, row in enumerate(row_list):
                row.append(result_list[i])
                writer.writerow(row)
