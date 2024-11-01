import pandas as pd
import json
from utils import DATASETS

if __name__ == "__main__":
    model_names = []
    for dataset in DATASETS:
        input_path = f'../data/pre_calibration/{dataset}/matrix.csv'
        model_name = pd.read_csv(input_path, index_col=0).index.tolist()
        model_names.extend(model_name)

    model_names = sorted(list(set(model_names)))
    model_df = pd.DataFrame({
        'model_id': range(len(model_names)),
        'model_names': model_names
    })

    model_df.to_csv('configs/model_id.csv', index=False)

    # model_names = list(set(model_names))
    # model_name_dict = {name: idx for idx, name in enumerate(model_names)}

    # output_path = 'configs/model_id.csv'
    # with open(output_path, 'w') as json_file:
    #     json.dump(model_name_dict, json_file, ensure_ascii=False, indent=4)
