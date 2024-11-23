from huggingface_hub import snapshot_download
import pandas as pd
from tqdm import tqdm
from utils.constants import DATASETS

if __name__ == "__main__":
    DATASETS = sorted(DATASETS)
    data_folder = snapshot_download(
        repo_id="stair-lab/reeval_responses", repo_type="dataset"
    )
    model_names = []
    for dataset in tqdm(DATASETS):
        model_name = pd.read_csv(f"{data_folder}/{dataset}/matrix.csv", index_col=0).index.tolist()
        model_names.extend(model_name)
    print(list(set(model_names)))
    model_names = sorted(list(set(model_names)))
    
    # create a dataframe, each row is a model, each column is a dataset, the value is 1 if the model is in the dataset
    df = pd.DataFrame(0, index=model_names, columns=DATASETS)
    for dataset in DATASETS:
        model_name = pd.read_csv(f"{data_folder}/{dataset}/matrix.csv", index_col=0).index.tolist()
        df.loc[model_name, dataset] = 1
    df.to_csv(f"{data_folder}/model_dataset_stat.csv")
    
    # generate a plot, turn 1 into checkmark
    import matplotlib.pyplot as plt
    import seaborn as sns
    df = pd.read_csv(f"{data_folder}/model_dataset_stat.csv", index_col=0)
    df = df.replace(1, "✔")
    plt.figure(figsize=(20, 10))
    sns.heatmap(df, cmap="coolwarm", cbar=False)
    plt.savefig(f"{data_folder}/model_dataset_stat.png")
    plt.show()
    
    