import re
import argparse
import pandas as pd
from huggingface_hub import HfApi, snapshot_download

scenario2pattern = {
    "mmlu/mmlu": r"(Question:)"
    # TODO: this pattern do not work for all scenarios, but split by '\n\n' is also problemetic, should write a scenario2pattern config
}

def extract_last_question(pattern, text):
    # Split into individual questions using 'Question:' as the delimiter
    questions = re.split(pattern, text)
  
    # Remove empty strings and strip whitespace
    questions = [q.strip() for q in questions if q.strip()]
    
    # Get the last question (ignoring the last empty answer)
    last_question = questions[-1]
    
    # Remove the 'Answer:' part if it exists
    last_question = re.split(r'Answer:', last_question)[0].strip()
    
    return last_question

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)  # mmlu/mmlu
    args = parser.parse_args()

    data_folder = snapshot_download(
        repo_id="stair-lab/reeval_matrices", repo_type="dataset"
    )
    question_keys = pd.read_csv(f"{data_folder}/{args.dataset}/question_keys.csv")
    pattern = scenario2pattern[args.dataset]
    
    last_questions = [extract_last_question(pattern, question_keys["prompt"][i]) for i in range(len(question_keys["prompt"]))]
    question_keys["zero_shot"] = last_questions

    # question_keys.to_csv(f"{data_folder}/{args.dataset}/question_keys.csv", index=False)

    print(question_keys)
    print(question_keys["zero_shot"][0])
