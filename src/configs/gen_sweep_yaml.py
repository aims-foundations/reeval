import argparse
import yaml
import sys
sys.path.append('..')
from utils import DATASETS

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=str, required=True)
    args = parser.parse_args()
    
    if args.project == "save_json_classic":
        datasets = [d for d in DATASETS if d != 'mmlu' and d != 'airbench']
        program = 'save_json.py'
    elif args.project == "get_response_matrix_classic":
        datasets = [d for d in DATASETS if d != 'mmlu' and d != 'airbench']
        program = 'get_response_matrix.py'
    else:
        datasets = DATASETS
        program = f'{args.project}.py'
        
    project = f'{args.project}'
    save_path = f'{args.project}.yaml'
    
    yaml_content = {
        'program': program,
        'project': project,
        'method': 'grid',
        'parameters': {
            'dataset': {
                'values': datasets
            },
        }
    }

    with open(save_path, 'w') as yaml_file:
        yaml.dump(yaml_content, yaml_file, default_flow_style=False)
        