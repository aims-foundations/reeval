import re
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict
import pandas as pd
import os
from huggingface_hub import login
from dotenv import load_dotenv
from datasets import load_dataset, concatenate_datasets
from sklearn.model_selection import train_test_split

def extract_before_prompt(text):
    before_prompt_pattern = r'^(.*?)(?=### PROMPT:)'
    before_prompt_match = re.search(before_prompt_pattern, text)
    return before_prompt_match.group(1).strip()

def extract_after_prompt(text):
    after_prompt_pattern = r'### PROMPT:\s*(.*)'
    after_prompt_match = re.search(after_prompt_pattern, text)
    return after_prompt_match.group(1).strip()

if __name__ == "__main__":
    load_dotenv()
    hf_token = os.getenv('HF_TOKEN')
    login(token=hf_token)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    
    hf_repo = f"stair-lab/reeval_aggregate-embed"
    dataset_info = load_dataset(hf_repo, split=None)
    splits = dataset_info.keys()
    datasets = [load_dataset(hf_repo, split=split) for split in splits]
    dataset = concatenate_datasets(datasets)
    
    if len(dataset) > 125000:
        dataset = dataset.select(range(125000))
    
    zs = dataset['z']
    
    sft_chat = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": (
            """Generate a question with a given difficulty score, which range from -5 to 5. """
            """The lower the score is, the more difficult the question is. """
            """Hence a model is more likely to fail the questions. """
            """Output only the question and nothing else. """
            """Topic: %s"""
            """Difficulty: %s. Question: """
            )
        },
        {"role": "assistant", "content": """%s"""},
    ]
    template = tokenizer.apply_chat_template(sft_chat, tokenize=False, add_generation_prompt=False)
    
    new_texts = []
    for i in range(len(dataset)):
        text =  dataset[i]['text']
        topic = extract_before_prompt(text)
        question = extract_after_prompt(text)
        text = template % (topic, round(zs[i], 2), question)
        new_texts.append(text)
    print(new_texts[0])
        
    push_df = pd.DataFrame(new_texts, columns=['text'])
    train_df, test_df = train_test_split(push_df, test_size=0.2, random_state=42)
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    dataset_dict = DatasetDict({
        "train": train_dataset,
        "test": test_dataset
    })
    dataset_dict.push_to_hub(f'stair-lab/aggregate-sft')
