import random
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict
import pandas as pd
import os
from huggingface_hub import login
from dotenv import load_dotenv
from datasets import load_dataset
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    load_dotenv()
    hf_token = os.getenv('HF_TOKEN')
    login(token=hf_token)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    dataset = load_dataset("stair-lab/airbench-difficulty", use_auth_token=True, split="whole")
    
    chat = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": 
            """Generate question with a given difficulty, which range from -5 to 5. """
            """The lower the score is, the more difficult the question is, """
            """hence a model is more likely to fail the questions. """
            """Output only the question and nothing else."""
            """Difficulty: "%s". Question: """
        },
    ]
    
    template = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    
    new_texts = []
    for _ in range(1000):
        indices = random.sample(range(len(dataset)), 3)
        
        prompt1, score1 = dataset[indices[0]]['question_text'], dataset[indices[0]]['z3']
        prompt2, score2 = dataset[indices[1]]['question_text'], dataset[indices[1]]['z3']
        prompt3, score3 = dataset[indices[2]]['question_text'], dataset[indices[2]]['z3']
        
        target_score = random.uniform(-5, 5)
        
        text = template % (prompt1, round(score1,2), prompt2, round(score2,2), prompt3, round(score3,2), round(target_score,2))
        new_texts.append(text)
        
    df = pd.DataFrame(new_texts, columns=['text'])
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    dataset_dict = DatasetDict({
        "train": train_dataset,
        "test": test_dataset
    })
    dataset_dict.push_to_hub('stair-lab/airbench-ppo')
