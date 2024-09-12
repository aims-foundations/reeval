import pandas as pd
import requests

file_path = f'../../data/real/crawl/crawl_dataset_name_lite.csv'
df = pd.read_csv(file_path)

filtered_df = df[df['Run'].str.startswith('mmlu:subject=college_chemistry')]
for i, row in filtered_df.iterrows():
    exp_string = row['Run']
    
    link = 'https://storage.googleapis.com/crfm-helm-public/lite/benchmark_output/runs/'
    if 480 <= i < 510:
        v = 0
    elif 872 <= i < 874:
        v = 1
    elif 1040 <= i < 1049:
        v = 2
    elif 1260 <= i < 1267:
        v = 3 
    elif 1472 <= i < 1480:
        v = 4
    elif 1648 <= i < 1653:
        v = 5
    elif 1835 <= i < 1843:
        v = 6
    elif 1978 <= i < 1981:
        v = 7
        
    url = f'{link}/v1.{v}.0/{exp_string}/scenario_state.json'
   
    response = requests.get(url)
    if response.status_code == 200:
        file_name = f'../../data/real/crawl/mmlu_chem_json/{exp_string}.json'
        with open(file_name, 'wb') as file:
            file.write(response.content)
        print(f'File {file_name} downloaded successfully')
    else:
        print(f'Failed to download the file for {exp_string}. Status code:', response.status_code)
