import csv
import os

import datasets
from bacher import Batcher
from dotenv import load_dotenv
from huggingface_hub import login


def extract_content(tag, text):
    start_idx = text.find(tag)
    start_of_content = start_idx + len(tag)
    if tag == "##the_score: ":
        end_idx = text.find("\n", start_of_content)
    else:
        end_idx = text.find(".\n##", start_of_content)
    if end_idx == -1:
        content = text[start_of_content:].strip()
    else:
        content = text[start_of_content:end_idx].strip()
    return content


if __name__ == "__main__":
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    openai_key = os.getenv("OPENAI_KEY")

    input_dir = "../../../gather_data/query_real/answer"
    output_dir = "../../../gather_data/query_real/eval"
    os.makedirs(output_dir, exist_ok=True)

    batcher = Batcher(
        api_key=openai_key,
        model_name="gpt-4o",
        temperature=0,
        num_workers=64,
    )

    login(token=hf_token)
    judge_prompts = datasets.load_dataset(
        "stanford-crfm/air-bench-2024", "judge_prompts", split="test"
    )

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

    for model_string in model_strings:
        model_string = model_string.replace("/", "_")
        with open(
            f"{input_dir}/answer_{model_string}_result.csv",
            "r",
            newline="",
            encoding="utf-8",
        ) as infile:
            reader = csv.reader(infile)
            header = next(reader)
            row_list = [row for row in reader]

        question_list = []
        for row in row_list:
            cate_idx = row[0]
            prompt = row[4]
            answer = row[5]

            for data in judge_prompts:
                if data["cate-idx"] == cate_idx:
                    judge_prompt = data["judge_prompt"]

            question_list.append(
                judge_prompt.replace("{{QUESTION}}", prompt).replace(
                    "{{ANSWER}}", answer
                )
            )

        result_list = batcher.handle_message_list(question_list)
        reason_list = [
            extract_content("##short_reasoning: ", result) for result in result_list
        ]
        score_list = [
            float(extract_content("##the_score: ", result)) for result in result_list
        ]

        with open(
            f"{output_dir}/eval_{model_string}_result.csv",
            "w",
            newline="",
            encoding="utf-8",
        ) as outfile:
            writer = csv.writer(outfile)
            writer.writerow(
                [
                    "cate-idx",
                    "l2-name",
                    "l3-name",
                    "l4-name",
                    "prompt",
                    "response",
                    "score_reason",
                    "score",
                ]
            )

            for i, row in enumerate(row_list):
                cate_idx, l2_name, l3_name, l4_name, prompt, response = row
                reason = reason_list[i]
                score = score_list[i]
                writer.writerow(
                    [
                        cate_idx,
                        l2_name,
                        l3_name,
                        l4_name,
                        prompt,
                        response,
                        reason,
                        score,
                    ]
                )
