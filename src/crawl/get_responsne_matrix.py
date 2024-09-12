import os
import json
import pandas as pd

json_dir = '../../data/real/crawl/mmlu_chem_json/'
response_matrix = {}
for json_file in os.listdir(json_dir):
    if json_file.endswith('.json'):
        with open(os.path.join(json_dir, json_file), 'r') as f:
            data = json.load(f)
        
        model_name = data['adapter_spec']['model']
        correct_answers = []
        
        for idx, question in enumerate(data['request_states']):
            # Step 1: Get the predicted answer from the model output, e.g., "B"
            predicted_answer = question['result']['completions'][0]['text'].strip()  # e.g., "B"
            
            # Step 2: Initialize correct to 0
            correct = 0
            
            # Step 3: Get the corresponding text for the predicted answer, Maps "B" to the actual text answer
            try:
                predicted_text = question['output_mapping'][predicted_answer]
            except KeyError:
                print(f"Warning: predicted answer '{predicted_answer}' not found in output_mapping for model '{model_name}', question index: {idx}")
                correct_answers.append(correct)
                continue  # Skip to the next question
            
            # Step 4: Loop through all choices
            for ref in question['instance']['references']:
                if ref['output']['text'] == predicted_text:
                    matching_ref = ref
                    break
            
            # Step 5: If a matching reference is found, check if it is marked as correct
            if 'correct' in matching_ref['tags']:
                correct = 1  # Mark as correct if the matching reference is correct

            correct_answers.append(correct)
        
        response_matrix[model_name] = correct_answers

response_df = pd.DataFrame(response_matrix).T
response_df.columns = [f'{i}' for i in range(response_df.shape[1])]

response_df.to_csv('../../data/real/crawl/response_matrix.csv', index_label=None)
