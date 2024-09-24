from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict
import pandas as pd
import os
from huggingface_hub import login
from dotenv import load_dotenv
from datasets import load_dataset, concatenate_datasets

if __name__ == "__main__":
    load_dotenv()
    hf_token = os.getenv('HF_TOKEN')
    login(token=hf_token)
    
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    hf_repo = "stair-lab/reeval_airbench-embed"
    dataset_train = load_dataset(hf_repo, split="train")
    dataset_test = load_dataset(hf_repo, split="test")
    dataset = concatenate_datasets([dataset_train, dataset_test])
    
    chat = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": (
            """Generate question with a given difficulty, which range from -5 to 5. """
            """The lower the difficulty is, the more difficult the question is, """
            """hence a model is more likely to fail the questions. """
            """Output only the question and nothing else."""
            """Difficulty: "%s". Question: """
        )
        },
        {"role": "assistant", "content": """\"%s\""""},
    ]
    
    template = tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )
    print(template)
    
    new_texts = []
    for i in range(len(dataset)):
        z = dataset[i]['z']
        question =  dataset[i]['text']
        text = template % (question, round(z, 2))
        new_texts.append(text)
        
    push_df = pd.DataFrame(new_texts, columns=['text'])
    push_dataset = Dataset.from_pandas(push_df)
    push_dict = DatasetDict({
        "train": push_dataset,
    })
    push_dict.push_to_hub('stair-lab/airbench-sft')
