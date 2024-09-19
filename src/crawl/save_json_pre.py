import argparse
import pandas as pd
import yaml

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, required=True)
    args = parser.parse_args()
  
    stats_strings = pd.read_csv(f'../../data/real/crawl/dataset_info_stats_{args.exp}.csv')['dataset_name'].tolist()
    start_strings = list(set([s.split(":")[0].split(",")[0] for s in stats_strings]))
    
    yaml_content = {
        'program': 'save_json.py',
        'project': 'save_json',
        'method': 'grid',
        'parameters': {
            'exp': {
                'values': [args.exp]
            },
            'start_string': {
                'values': start_strings
            }
        }
    }

    with open('save_json.yaml', 'w') as yaml_file:
        yaml.dump(yaml_content, yaml_file, default_flow_style=False)