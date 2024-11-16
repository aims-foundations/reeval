from datasets import concatenate_datasets, DatasetDict, load_dataset

from utils import DATASETS

dataset_dict = {}
for dataset_name in DATASETS:
    # Load the train and test parts of each dataset
    dataset = load_dataset(f"stair-lab/reeval_{dataset_name}-embed")

    # Merge train and test splits
    merged_dataset = concatenate_datasets([dataset["train"], dataset["test"]])

    dataset_dict[dataset_name] = merged_dataset

hf_dataset_dict = DatasetDict(dataset_dict)
hf_dataset_dict.push_to_hub("stair-lab/reeval_individual-embed")
