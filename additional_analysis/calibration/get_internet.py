import pickle
import random

from datasets import load_dataset

fw = load_dataset(
    "HuggingFaceFW/fineweb", name="CC-MAIN-2024-10", split="train", streaming=True
)
sampled_dataset = fw.filter(lambda x: random.random() < 0.1)
data = []
count = 0
total_size = 16_000_000  # 4096 * 4096 * 3
bathch_size = 50_000
for sample in sampled_dataset.take(total_size):
    data.append(sample["text"])
    count = count + 1
    if count > 0 and count % bathch_size == 0 or count == total_size:
        print(f"Saving {count} samples")
        with open("sample_internet.pkl", "wb") as f:
            pickle.dump(data, f)
