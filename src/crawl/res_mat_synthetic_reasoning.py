import os
import json
import pandas as pd

json_dir = '../../data/real/crawl/synthetic_reasoning_json/'
start_strings = [
    "synthetic_reasoning:mode=induction", 
    "synthetic_reasoning:mode=pattern_match", 
    "synthetic_reasoning:mode=variable_substitution", 
    "synthetic_reasoning_natural:difficulty=easy", 
    "synthetic_reasoning_natural:difficulty=hard"
]

all_responses_df = pd.DataFrame()
for start_string in start_strings:
    print(f"Processing dataset: {start_string}")
    
    response_matrix = {}
    for json_file in os.listdir(json_dir):
        if json_file.endswith('.json') and json_file.startswith(start_string):
            with open(os.path.join(json_dir, json_file), 'r') as f:
                data = json.load(f)
            
            model_name = data['adapter_spec']['model']
            correct_answers = []
            
            for idx, question in enumerate(data['request_states']):
                predicted_answer = question['result']['completions'][0]['text'].strip()
                true_answer = question['instance']['references'][0]['output']['text'].strip()
                correct_answers.append(int(predicted_answer==true_answer))

            if len(correct_answers) == 3000:
                correct_answers = correct_answers[:1000]
            response_matrix[model_name] = correct_answers
            
    # for model_name, correct_answers in response_matrix.items():
    #     print(f"Model: {model_name}, Number of answers: {len(correct_answers)}")
    
    response_df = pd.DataFrame(response_matrix).T
    response_df = response_df.sort_index(axis=0)
    
    if all_responses_df.empty:
        all_responses_df = response_df
    else:
        assert (all_responses_df.index == response_df.index).all(), "Model names do not match!"
        all_responses_df = pd.concat([all_responses_df, response_df], axis=1)
        
all_responses_df.columns = [f'{i}' for i in range(all_responses_df.shape[1])]
all_responses_df.to_csv('../../data/real/response_matrix/normal_syn_reason/all_matrix.csv', index_label=None)


