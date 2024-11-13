import argparse
import yaml
import sys
sys.path.append('..')
from utils import DATASETS

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=str, required=True)
    args = parser.parse_args()
    
    save_path = f'{args.project}.yaml'
    
    if args.project == "plugin_regression" or args.project == "amor_calibration":
        datasets = DATASETS
        program = f'{args.project}.py'
        project = f'{args.project}'
        
        yaml_content = {
            'program': program,
            'project': project,
            'method': 'grid',
            'parameters': {
                'dataset': {
                    'values': datasets
                },
                'seed': {
                    'values': [i for i in range(10)]
                },
            }
        }
    
    elif args.project == "plugin_regression_aggregate":
        program = 'plugin_regression.py'
        project = 'plugin_regression'
        
        yaml_content = {
            'program': program,
            'project': project,
            'method': 'grid',
            'parameters': {
                'dataset': {
                    'values': ['aggregate']
                },
                'seed': {
                    'values': [i for i in range(10)]
                },
                'task': {
                    'values': ['byrandom', 'bydataset']
                },
            }
        }
    
    elif args.project == "agg_embed":
        datasets = DATASETS
        program = 'embed.py'
        project = 'embed'
        
        yaml_content = {
            'program': program,
            'project': project,
            'method': 'grid',
            'parameters': {
                'dataset': {
                    'values': datasets
                },
                'agg_tag': {
                    'values': [True]
                },
            }
        }
        
    else:
        if args.project == "save_json_classic":
            datasets = [d for d in DATASETS if d != 'mmlu' and d != 'airbench']
            program = 'save_json.py'
            project = 'save_json'
            
        elif args.project == "get_response_matrix_classic":
            datasets = [d for d in DATASETS if d != 'mmlu' and d != 'airbench']
            program = 'get_response_matrix.py'
            project = 'get_response_matrix'
            
        else:
            datasets = DATASETS
            program = f'{args.project}.py'
            project = f'{args.project}'
        
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
        
        