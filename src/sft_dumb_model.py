from datasets import load_dataset
from trl import SFTTrainer

dataset = load_dataset("stair-lab/airbench-fintune", split="train")

trainer = SFTTrainer(
    "google/gemma-7b-it",
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=512,
)

trainer.train()