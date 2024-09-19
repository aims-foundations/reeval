import argparse
import os
import json
import pandas as pd

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--leaderboard', type=str, required=True) # classic, mmlu
    parser.add_argument('--start_string', type=str, required=True) # use wandb sweep, mmlu
    args = parser.parse_args()
  
    full_strings_all = pd.read_csv(f'../../../data/real/crawl/crawl_dataset_name_{args.leaderboard}.csv')['Run'].tolist()
    
    save_dir = f'../../../data/real/crawl/{args.start_string}_json'
    os.makedirs(save_dir, exist_ok=True)
    full_strings = [f for f in full_strings_all if f.startswith(args.start_string)]
    for full_string in full_strings:
        save_path = f'{save_dir}/{full_string}.json'
        
    if args.leaderboard == "mmlu":
        json_dir = '../../../data/real/crawl/mmlu_json'
        output_dir = "../../../data/real/response_matrix/normal_mmlu"
        start_strings = pd.read_csv('../../../data/real/crawl/dataset_info_stats_mmlu.csv')['dataset_name'].tolist()
        start_strings = [s.split(",eval_split")[0] for s in start_strings]

    # mask matrix
    all_responses_df = pd.DataFrame()
    max_lens = []
    max_len_file_names = []
    
    for start_string in start_strings:
        max_length = 0
        max_file_names = []
        for json_file in os.listdir(json_dir):
            if json_file.endswith('.json') and json_file.startswith(start_string):
                with open(f"{json_dir}/{json_file}", 'r') as f:
                    data = json.load(f)
                len_q = len(data['request_states'])
                max_file_names.append((json_file, len_q))
                if len_q > max_length:
                    max_length = len_q
        print(f"dataset {start_string}, max length {max_length}")
        max_lens.append(max_length)
        max_len_file_names.append(max(max_file_names, key=lambda x: x[1])[0])
        
        response_matrix = {}
        for json_file in os.listdir(json_dir):
            if json_file.endswith('.json') and json_file.startswith(start_string):
                with open(f"{json_dir}/{json_file}", 'r') as f:
                    data = json.load(f)
                
                model_name = data['adapter_spec']['model']
                bool_answers = get_bool_answers(data)

                if len(bool_answers) < max_length:
                    bool_answers.extend([-1] * (max_length - len(bool_answers)))
                response_matrix[model_name] = bool_answers
           
        response_df = pd.DataFrame(response_matrix).T
        response_df = response_df.sort_index(axis=0)
        
        if all_responses_df.empty:
            all_responses_df = response_df
        else:
            assert (all_responses_df.index == response_df.index).all(), "Model names do not match!"
            all_responses_df = pd.concat([all_responses_df, response_df], axis=1)
            
    all_responses_df.columns = [f'{i}' for i in range(all_responses_df.shape[1])]
    
    bool_delete_list = []
    for col_name, col_data in all_responses_df.items():
        if set(col_data.unique()).issubset({0, -1}) or set(col_data.unique()).issubset({1, -1}):
            all_responses_df = all_responses_df.drop(columns=[col_name])
            bool_delete_list.append(1)
        else:
            bool_delete_list.append(0)

    # index search file
    output_file = f"{output_dir}/mask_index_search.csv"
    output_data = []
    base_idx = 0
    for i, start_string in enumerate(start_strings):
        input_file = f"{json_dir}/{max_len_file_names[i]}"
        with open(input_file, 'r') as f:
            data = json.load(f)
        for j, question in enumerate(data['request_states']):
            text = question['instance']['input']['text']
            output_data.append([base_idx+j, text, bool_delete_list[base_idx+j]])
        base_idx += max_lens[i]

    output_df = pd.DataFrame(output_data, columns=["idx", "text", "is_deleted"])
    output_df.to_csv(output_file, index=False)

    all_responses_df.columns = [f'{i}' for i in range(all_responses_df.shape[1])]
    all_responses_df.to_csv(f'{output_dir}/mask_matrix.csv', index_label=None)

    if min_lens == max_lens:
        print("min_lens == max_lens, non_mask_matrix and mask_matrix are the same")
