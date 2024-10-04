import argparse
import pandas as pd
import re
import requests
from tqdm import tqdm

def delete_model_name(filename):
    return re.sub(r'model=[^,]*,?', '', filename).strip(',')

def get_question_count(exp_string, leaderboard):
    if leaderboard == "lite":
        base_url = 'https://storage.googleapis.com/crfm-helm-public/lite/benchmark_output/runs/v1.'
        max_version = 7
    elif leaderboard == "classic":
        base_url = 'https://storage.googleapis.com/crfm-helm-public/benchmark_output/runs/v0.'
        max_version = 4
    elif leaderboard == "mmlu":
        base_url = 'https://storage.googleapis.com/crfm-helm-public/mmlu/benchmark_output/runs/v1.'
        max_version = 8

    for i in range(max_version + 1):
        url = f'{base_url}{i}.0/{exp_string}/scenario_state.json'
        response = requests.get(url)
        if response.status_code == 200:
            json_data = response.json()
            question_count = len(json_data.get('request_states', []))
            return question_count

    print(f"Could not retrieve data for {exp_string} from any version, return 0")
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--leaderboard', type=str, required=True, choices=['classic', 'lite', 'mmlu'])
    args = parser.parse_args()

    input_path = f'../../../data/gather_data/crawl_real/crawl_dataset_name_{args.leaderboard}.csv'
    df_input = pd.read_csv(input_path)

    df_input['cleaned_run'] = df_input['Run'].apply(delete_model_name)
    dataset_names = df_input['cleaned_run'].unique().tolist()
    model_counts = df_input.groupby('cleaned_run').size().tolist()
    first_run_list = df_input.groupby('cleaned_run')['Run'].first().tolist()
    
    output_path = f'../../../data/gather_data/crawl_real/dataset_info_stats_{args.leaderboard}.csv'

    for i, exp_string in enumerate(tqdm(first_run_list)):
        question_count = get_question_count(exp_string, args.leaderboard)
        df_new = pd.DataFrame([{
            'dataset_name': dataset_names[i],
            'model_count': model_counts[i],
            'question_count': question_count
        }])

        if i == 0:
            df_output = df_new
        else:
            df_exist = pd.read_csv(output_path)
            df_output = pd.concat([df_exist, df_new], ignore_index=True)
    
        df_output = df_output.sort_values(
            by=['model_count', 'question_count'], ascending=[False, False]
        )
        df_output.to_csv(output_path, index=False)    
        


