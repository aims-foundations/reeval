import argparse
import pandas as pd
import yaml

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--leaderboard', type=str, default="classic") # classic
    args = parser.parse_args()
  
    stats_strings = pd.read_csv(f'../../../data/gather_data/crawl_real/dataset_info_stats_{args.leaderboard}.csv')['dataset_name'].tolist()
    start_strings = list(set([s.split(":")[0].split(",")[0] for s in stats_strings]))
    start_strings = [s for s in start_strings if s != "mmlu"]
    
    yaml_content = {
        'program': 'get_response_matrix.py',
        'project': 'get_response_matrix',
        'method': 'grid',
        'parameters': {
            'leaderboard': {
                'values': [args.leaderboard]
            },
            'start_string': {
                'values': start_strings
            }
        }
    }

    with open('get_response_matrix.yaml', 'w') as yaml_file:
        yaml.dump(yaml_content, yaml_file, default_flow_style=False)