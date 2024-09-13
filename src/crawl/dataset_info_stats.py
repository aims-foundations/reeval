import pandas as pd
import re
import requests
from tqdm import tqdm

def clean_dataset_name(name):
    return re.sub(r'model=[^,]*,?', '', name).strip(',')

def get_question_count(exp_string, exp):
    if exp == "lite":
        base_url = 'https://storage.googleapis.com/crfm-helm-public/lite/benchmark_output/runs/v1.'
        max_version = 7
    elif exp == "classic":
        base_url = 'https://storage.googleapis.com/crfm-helm-public/benchmark_output/runs/v0.'
        max_version = 4
    version_found = False
    for i in range(max_version + 1):
        url = f'{base_url}{i}.0/{exp_string}/scenario_state.json'
        response = requests.get(url)
        if response.status_code == 200:
            json_data = response.json()
            question_count = len(json_data.get('request_states', []))
            version_found = True
            return question_count
        else:
            print(f"Failed to retrieve data for {exp_string} with version {i}, status code: {response.status_code}")
    
    if not version_found:
        print(f"Could not retrieve data for {exp_string} from any version v1.0.0 to v1.{max_version}.0, return 0")
        return 0


if __name__ == "__main__":
    exp = "classic"

    file_path = f'../../data/real/crawl/crawl_dataset_name_{exp}.csv'
    df = pd.read_csv(file_path)

    df['cleaned_run'] = df['Run'].apply(clean_dataset_name)
    dataset_names = df['cleaned_run'].unique().tolist()
    model_counts = df.groupby('cleaned_run').size().tolist()
    
    first_run_list = df.groupby('cleaned_run')['Run'].first().tolist()
    output_path = f'../../data/real/crawl/dataset_info_stats_{exp}.csv'

    all_results = []
    for i, exp_string in enumerate(tqdm(first_run_list)):
        question_count = get_question_count(exp_string, exp)
        all_results.append({
            'dataset_name': dataset_names[i],
            'model_count': model_counts[i],
            'question_count': question_count
        })

        result_df = pd.DataFrame(all_results)
        result_df = result_df.sort_values(by=['model_count', 'question_count'], ascending=[False, False])

        if i == 0:
            result_df.to_csv(output_path, index=False, mode='w')
        else:
            result_df.to_csv(output_path, index=False, mode='a', header=False)