import os
from dotenv import load_dotenv
import csv
import argparse
from huggingface_hub import login
import datasets
from bacher_modify import GPTBatcher

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
    hf_token = os.getenv('HF_TOKEN')
    
    os.makedirs("../../../data/real/pre_irt_data/eval", exist_ok=True)

    parser = argparse.ArgumentParser(description='get eval score using get-as-a-judge')
    parser.add_argument('--start', type=int, required=True, help='Start index')
    parser.add_argument('--end', type=int, required=True, help='End index')

    args = parser.parse_args()
    start = args.start
    end = args.end

    load_dotenv()
    openai_key = os.getenv('OPENAI_KEY')
    batcher_eval = GPTBatcher(api_key=openai_key,
                            model_name='gpt-4o',
                            temperature=0,
                            num_workers=64,
                            )

    login(token = hf_token)
    judge_prompts = datasets.load_dataset("stanford-crfm/air-bench-2024", "judge_prompts", split="test")

    model_string_list = [
        'meta-llama/Meta-Llama-3-8B-Instruct',
        ]
    subset_model_string_list = model_string_list[start-1:end]

    for model_string in subset_model_string_list:
        model_string = model_string.replace('/','_')
        with open(f'../../../data/real/pre_irt_data/answer/answer_{model_string}_result.csv', 'r', newline='', encoding='utf-8') as infile:
            reader = csv.reader(infile)
            header = next(reader)
            row_list = [row for row in reader]

        question_list = []
        for row in row_list:
            cate_idx = row[0]
            prompt = row[4]
            answer = row[5]

            for data in judge_prompts:
                if data['cate-idx'] == cate_idx:
                    judge_prompt = data['judge_prompt']
            
            question_list.append(judge_prompt.replace("{{QUESTION}}", prompt).replace("{{ANSWER}}", answer))

        result_list = batcher_eval.handle_message_list(question_list)
        reason_list = [extract_content("##short_reasoning: ", result) for result in result_list]
        score_list = [float(extract_content("##the_score: ", result)) for result in result_list]

        with open(f'../../../data/real/pre_irt_data/eval/eval_{model_string}_result.csv', 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(['cate-idx', 'l2-name', 'l3-name', 'l4-name', 'prompt', 'response', 'score_reason', 'score'])

            for i, row in enumerate(row_list):
                cate_idx, l2_name, l3_name, l4_name, prompt, response = row
                reason = reason_list[i]
                score = score_list[i]

                writer.writerow([cate_idx, l2_name, l3_name, l4_name, prompt, response, reason, score])
