import argparse
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict
import pandas as pd
import os
from huggingface_hub import login
from dotenv import load_dotenv
from datasets import load_dataset, concatenate_datasets
from sklearn.model_selection import train_test_split
import pickle

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, choices=['sft', 'ppo'])
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--model', type=str, default='mlp', choices=['ridge', 'mlp'])
    args = parser.parse_args()
    
    load_dotenv()
    hf_token = os.getenv('HF_TOKEN')
    login(token=hf_token)
    
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    hf_repo = f"stair-lab/reeval_{args.dataset}-embed"
    dataset_train = load_dataset(hf_repo, split="train")
    dataset_test = load_dataset(hf_repo, split="test")
    dataset = concatenate_datasets([dataset_train, dataset_test])
    
    if len(dataset) > 125000:
        dataset = dataset.select(range(125000))
    
    model_path = f'../data/plugin_regression/{args.dataset}/{args.model}.pkl'
    with open(model_path, 'rb') as f:
        reg_model = pickle.load(f)
    embs = dataset['embed']
    pred_zs = reg_model.predict(embs).tolist()
    
    ppo_chat = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": (
            """Generate a question with a given difficulty score, which range from -5 to 5. """
            """The lower the score is, the more difficult the question is. """
            """Hence a model is more likely to fail the questions. """
            """Output only the question and nothing else. """
            """Difficulty: %s. Question: """
        )
        },
    ]
    sft_chat = ppo_chat + [{"role": "assistant", "content": """%s"""}]
    
    if args.task == 'ppo':
        template = tokenizer.apply_chat_template(ppo_chat, tokenize=False, add_generation_prompt=True)
    elif args.task == 'sft':
        template = tokenizer.apply_chat_template(sft_chat, tokenize=False, add_generation_prompt=False)
    
    new_texts = []
    for i in range(len(dataset)):
        z = pred_zs[i]
        question =  dataset[i]['text']
        if args.task == 'ppo':
            text = template % round(z, 2)
        elif args.task == 'sft':
            text = template % (round(z, 2), question)
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
    dataset_dict.push_to_hub(f'stair-lab/{args.dataset}-{args.task}')
