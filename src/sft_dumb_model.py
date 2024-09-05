from datasets import load_dataset
from trl import SFTTrainer

dataset = load_dataset("stair-lab/airbench-fintune", split="train")

trainer = SFTTrainer(
    "Qwen/Qwen2-72B-Instruct",
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=512,
)

trainer.train()