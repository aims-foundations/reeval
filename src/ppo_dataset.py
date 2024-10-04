import argparse
import numpy as np
import torch
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict
import pandas as pd
import os
from huggingface_hub import login
from dotenv import load_dotenv
from datasets import load_dataset, concatenate_datasets
from sklearn.model_selection import train_test_split
import pickle
from utils import MLP

def mlp_predict(model, emb_input):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    emb_input = torch.tensor(emb_input, dtype=torch.float32).to(device)
    with torch.no_grad():
        output = model(emb_input)
    return output.cpu().numpy().flatten()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--ppo_size', type=int, default=125000)
    parser.add_argument('--model', type=str, default='bayridge', choices=['bayridge', 'mlp'])
    args = parser.parse_args()
    
    load_dotenv()
    hf_token = os.getenv('HF_TOKEN')
    login(token=hf_token)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    
    hf_repo = f"stair-lab/reeval_{args.dataset}-embed"
    dataset_train = load_dataset(hf_repo, split="train")
    dataset_test = load_dataset(hf_repo, split="test")
    dataset = concatenate_datasets([dataset_train, dataset_test])
    
    embs = dataset['embed']
    model_path = f'../data/plugin_regression/{args.dataset}/{args.model}.pkl'
    if args.model == 'bayridge':
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        pred_zs = model.predict(embs).tolist()
    elif args.model == 'mlp':
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        pred_zs = mlp_predict(model, embs).tolist()
    mean_pred_z = np.mean(np.array(pred_zs))
    std_pred_z = np.std(np.array(pred_zs))
    print(f"z mean: {mean_pred_z}, std: {std_pred_z}")
    
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
    template = tokenizer.apply_chat_template(ppo_chat, tokenize=False, add_generation_prompt=True)
    
    new_texts = []
    for i in range(args.ppo_size):
        z = np.random.normal(mean_pred_z, std_pred_z)
        text = template % round(z, 2)
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
    dataset_dict.push_to_hub(f'stair-lab/{args.dataset}-ppo-{args.llm}')
