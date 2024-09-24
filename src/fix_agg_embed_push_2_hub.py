from datasets import Dataset, DatasetDict
import pandas as pd
import os
from huggingface_hub import login
from dotenv import load_dotenv
from utils import DATASETS
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
     # 加载环境变量和 Hugging Face 令牌
    load_dotenv()
    hf_token = os.getenv('HF_TOKEN')
    login(token=hf_token)
    
    # 设置输入和输出路径
    input_dir = '/lfs/local/0/nqduc/.cache/huggingface/hub/datasets--stair-lab--reeval-agg_embed_folder/snapshots/07bcb8c88effd0fbd9b5811a0dc84235ebcdb1bf'
    output_dir = f'{input_dir}/new'

    # 读取所有数据集并整合成一个大的 DataFrame
    agg_df = pd.concat(
        [pd.read_csv(f'{output_dir}/new_embed_{dataset}.csv') for dataset in DATASETS],
        ignore_index=True
    )
    
    # 将 embed 列整合为一个嵌套的 list of float，长度为 4096
    agg_df['embed'] = agg_df[[f'embed_{i}' for i in range(4096)]].values.tolist()

    # 删除原有的 4096 个独立的 embed 列
    agg_df = agg_df.drop(columns=[f'embed_{i}' for i in range(4096)])

    # 随机打乱并划分为 80% 的训练集和 20% 的测试集
    train_df, test_df = train_test_split(agg_df, test_size=0.2, random_state=42)

    # 将数据集转换为 Hugging Face 的 Dataset 格式
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    # 创建一个 DatasetDict 来包含训练集和测试集
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'test': test_dataset
    })

    # 推送到 Hugging Face Hub
    dataset_dict.push_to_hub("stair-lab/reeval_aggregate-embed_2")
