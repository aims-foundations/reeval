    
if __name__ == "__main__":

    with open(save_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Convert back to DataFrame to create datasets
    push_df = pd.DataFrame(data)

    # Split the data into train and test
    split_index = int(0.8 * len(push_df))
    push_train_df, push_test_df = push_df[:split_index], push_df[split_index:]

    # Convert to Hugging Face dataset format
    push_train_dataset = Dataset.from_pandas(push_train_df.reset_index(drop=True))
    push_test_dataset = Dataset.from_pandas(push_test_df.reset_index(drop=True))

    # Create a dataset dictionary and push to Hugging Face
    push_dataset_dict = DatasetDict({
        'train': push_train_dataset,
        'test': push_test_dataset
    })
    push_dataset_dict.push_to_hub(hf_repo)
