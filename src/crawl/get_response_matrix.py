import argparse
import os
import json
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, required=True)
    args = parser.parse_args()
    exp = args.exp
    
    if exp == "synthetic_reasoning":
        # we delete the model T0pp (11B)
        json_dir = '../../data/real/crawl/synthetic_reasoning_json'
        output_dir = "../../data/real/response_matrix/normal_syn_reason"
        start_strings = [
            "synthetic_reasoning:mode=induction",
            "synthetic_reasoning:mode=pattern_match", 
            "synthetic_reasoning:mode=variable_substitution", 
            "synthetic_reasoning_natural:difficulty=easy", 
            "synthetic_reasoning_natural:difficulty=hard"
        ]
        
    elif exp == "mmlu":
        json_dir = '../../data/real/crawl/mmlu_json'
        output_dir = "../../data/real/response_matrix/normal_mmlu"
        start_strings = pd.read_csv('../../data/real/crawl/dataset_info_stats_mmlu.csv')['dataset_name'].tolist()
        start_strings = [s.split(",eval_split")[0] for s in start_strings]
        # delete mmlu:subject=miscellaneous
        start_strings = start_strings[:-1]
    
    elif exp == "civil_comment":
        # we delete Palmyra X (43B)
        json_dir = '../../data/real/crawl/civil_comment_json'
        output_dir = "../../data/real/response_matrix/normal_civil_comment"
        start_strings = [
            "civil_comments:demographic=LGBTQ",
            "civil_comments:demographic=all",
            "civil_comments:demographic=black",
            "civil_comments:demographic=christian",
            "civil_comments:demographic=female",
            "civil_comments:demographic=male",
            "civil_comments:demographic=muslim",
            "civil_comments:demographic=other_religions",
            "civil_comments:demographic=white"
        ]
        start_strings = [s.split(",data_augmentation")[0] for s in start_strings]
    
    # non mask matrix
    all_responses_df = pd.DataFrame()
    min_lens = []
    min_len_file_names = []
    
    for start_string in start_strings:
        min_length = float('inf')
        min_file_names = []
        for json_file in os.listdir(json_dir):
            if json_file.endswith('.json') and json_file.startswith(start_string):
                with open(f"{json_dir}/{json_file}", 'r') as f:
                    data = json.load(f)
                len_q = len(data['request_states'])
                min_file_names.append((json_file, len_q))
                if len_q < min_length:
                    min_length = len_q
        print(f"dataset {start_string}, min length {min_length}")
        min_lens.append(min_length)
        min_len_file_names.append(min(min_file_names, key=lambda x: x[1])[0])
        
        response_matrix = {}
        for json_file in os.listdir(json_dir):
            if json_file.endswith('.json') and json_file.startswith(start_string):
                with open(f"{json_dir}/{json_file}", 'r') as f:
                    data = json.load(f)
                
                model_name = data['adapter_spec']['model']
                correct_answers = []
                
                for idx, question in enumerate(data['request_states']):
                    if exp == "synthetic_reasoning" or exp == "civil_comment":
                        predicted_answer = question['result']['completions'][0]['text'].strip()
                        true_answer = question['instance']['references'][0]['output']['text'].strip()
                        correct_answers.append(int(predicted_answer==true_answer))
                        
                    elif exp == "mmlu":
                         # Step 1: Get the predicted answer from the model output, e.g., "B"
                        predicted_answer = question['result']['completions'][0]['text'].strip()  # e.g., "B"
                        # Step 2: Get the corresponding text for the predicted answer, Maps "B" to the actual text answer
                        try:
                            predicted_text = question['output_mapping'][predicted_answer]
                        except KeyError:
                            # print(f"Warning: predicted answer '{predicted_answer}' not found in output_mapping for model '{model_name}', question index: {idx}")
                            correct_answers.append(0)
                            continue
                        # Step 3: Loop through all choices
                        for ref in question['instance']['references']:
                            if ref['output']['text'] == predicted_text:
                                matching_ref = ref
                                break
                        # Step 4: If a matching reference is found, check if it is marked as correct
                        if 'correct' in matching_ref['tags']:
                            correct_answers.append(1)
                        else:
                            correct_answers.append(0)

                correct_answers = correct_answers[:min_length]
                response_matrix[model_name] = correct_answers
           
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
        if col_data.nunique() == 1:
            all_responses_df = all_responses_df.drop(columns=[col_name])
            bool_delete_list.append(1)
        else:
            bool_delete_list.append(0)

    # index search file
    output_file = f"{output_dir}/non_mask_index_search.csv"
    output_data = []
    base_idx = 0
    for i, start_string in enumerate(start_strings):
        start_string_2 = start_string.replace(":","-")
        input_file = f"{json_dir}/{min_len_file_names[i]}"
        with open(input_file, 'r') as f:
            data = json.load(f)
        for j, question in enumerate(data['request_states']):
            if j > min_lens[i]:
                break
            text = question['instance']['input']['text']
            output_data.append([base_idx+j, text, bool_delete_list[base_idx+j]])
        base_idx += min_lens[i]

    output_df = pd.DataFrame(output_data, columns=["idx", "text", "is_deleted"])
    output_df.to_csv(output_file, index=False)

    all_responses_df.columns = [f'{i}' for i in range(all_responses_df.shape[1])]
    all_responses_df.to_csv(f'{output_dir}/non_mask_matrix.csv', index_label=None)



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
                correct_answers = []
                
                for idx, question in enumerate(data['request_states']):
                    if exp == "synthetic_reasoning" or exp == "civil_comment":
                        predicted_answer = question['result']['completions'][0]['text'].strip()
                        true_answer = question['instance']['references'][0]['output']['text'].strip()
                        correct_answers.append(int(predicted_answer==true_answer))
                        
                    elif exp == "mmlu":
                         # Step 1: Get the predicted answer from the model output, e.g., "B"
                        predicted_answer = question['result']['completions'][0]['text'].strip()  # e.g., "B"
                        # Step 2: Get the corresponding text for the predicted answer, Maps "B" to the actual text answer
                        try:
                            predicted_text = question['output_mapping'][predicted_answer]
                        except KeyError:
                            # print(f"Warning: predicted answer '{predicted_answer}' not found in output_mapping for model '{model_name}', question index: {idx}")
                            correct_answers.append(0)
                            continue
                        # Step 3: Loop through all choices
                        for ref in question['instance']['references']:
                            if ref['output']['text'] == predicted_text:
                                matching_ref = ref
                                break
                        # Step 4: If a matching reference is found, check if it is marked as correct
                        if 'correct' in matching_ref['tags']:
                            correct_answers.append(1)
                        else:
                            correct_answers.append(0)

                if len(correct_answers) < max_length:
                    correct_answers.extend([-1] * (max_length - len(correct_answers)))
                response_matrix[model_name] = correct_answers
           
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
