import argparse
import os
import json
import re
import pandas as pd
import wandb
from dataset_info_stats import delete_model_name

def extract_model_name(filename):
    match = re.search(r'model=([^,]*)', filename)
    return match.group(1)

def get_bool_answers(data):
    bool_answers = []
    for question in data['request_states']:
        if 'output_mapping' in question:
            # Step 1: Get the predicted answer from the model output, e.g., "B"
            predicted_answer = question['result']['completions'][0]['text'].strip()
            # Step 2: Get the corresponding text for the predicted answer, Maps "B" to the actual text answer
            try:
                predicted_text = question['output_mapping'][predicted_answer]
            except KeyError:
                bool_answers.append(0)
                continue
            # Step 3: Loop through all choices
            for ref in question['instance']['references']:
                if ref['output']['text'] == predicted_text:
                    matching_ref = ref
                    break
            # Step 4: If a matching reference is found, check if it is marked as correct
            if 'correct' in matching_ref['tags']:
                bool_answers.append(1)
            else:
                bool_answers.append(0)
        else:
            predicted_answer = question['result']['completions'][0]['text'].strip()
            true_answer = question['instance']['references'][0]['output']['text'].strip()
            bool_answers.append(int(predicted_answer==true_answer))
    return bool_answers

if __name__ == "__main__":
    wandb.init(mode = "offline")
    parser = argparse.ArgumentParser()
    parser.add_argument('--leaderboard', type=str, default="classic") # classic, mmlu
    parser.add_argument('--start_string', type=str, required=True) # use wandb sweep, mmlu
    args = parser.parse_args()
  
    input_dir = f'../../../data/gather_data/crawl_real/{args.start_string}_json'
    output_dir = f'../../../data/pre_calibration/{args.start_string}'
    os.makedirs(output_dir, exist_ok=True)
    
    full_strings_all = pd.read_csv(f'../../../data/gather_data/crawl_real/crawl_dataset_name_{args.leaderboard}.csv')['Run'].tolist()
    full_strings = [f for f in full_strings_all if f.startswith(args.start_string)]
    all_model_names = list(set([extract_model_name(f) for f in full_strings]))
    all_model_names = sorted(all_model_names, key=lambda x: x[0])
    
    non_model_strings = list(set([delete_model_name(f) for f in full_strings]))
    
    # response matrix
    max_lens = []
    max_len_file_names = []
    for i, non_model_string in enumerate(non_model_strings):
        max_len = 0
        max_len_file_name = ""
        single_matrix = {name: [] for name in all_model_names}
        
        for filename in os.listdir(input_dir):
            if filename.endswith('.json') and (delete_model_name(filename) == f"{non_model_string}.json"):
                model_name = extract_model_name(filename)
                with open(f"{input_dir}/{filename}", 'r') as f:
                    data = json.load(f)
                    
                len_q = len(data['request_states'])
                if len_q > max_len:
                    max_len = len_q
                    max_len_file_name = filename
                    
                bool_answers = get_bool_answers(data)
                single_matrix[model_name] = bool_answers
        for model_name, bool_answers in single_matrix.items():
            single_matrix[model_name] += [-1] * (max_len - len(single_matrix[model_name]))
        
        max_lens.append(max_len)
        max_len_file_names.append(max_len_file_name)
        
        single_matrix_df = pd.DataFrame(single_matrix).T
        single_matrix_df.columns = [f"{j}_{non_model_string}" for j in range(max_len)]
        
        if i == 0:
            all_matrix_df = single_matrix_df
        else:
            assert (all_matrix_df.index == single_matrix_df.index).all()
            all_matrix_df = pd.concat([all_matrix_df, single_matrix_df], axis=1)
    
    bool_delete_list = []
    for col_name, col_data in all_matrix_df.items():
        if set(col_data.unique()).issubset({0, -1}) or set(col_data.unique()).issubset({1, -1}):
            all_responses_df = all_responses_df.drop(columns=[col_name])
            bool_delete_list.append(1)
        else:
            bool_delete_list.append(0)
    all_responses_df.to_csv(f'{output_dir}/matrix.csv', index_label=None)

    # index search
    search_list = []
    base_idx = 0
    for i, non_model_string in enumerate(non_model_strings):
        with open(f"{input_dir}/{max_len_file_names[i]}", 'r') as f:
            data = json.load(f)
        for j, question in enumerate(data['request_states']):
            text = question['instance']['input']['text']
            search_list.append([base_idx+j, text, bool_delete_list[base_idx+j]])
        base_idx += max_lens[i]

    search_df = pd.DataFrame(search_list, columns=["idx", "text", "is_deleted"])
    search_df.to_csv(f"{output_dir}/search.csv", index=False)

