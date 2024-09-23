import argparse
import yaml
from utils import DATASETS

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True)
    args = parser.parse_args()
    
    yaml_content = {
        'program': f'{args.task}.py',
        'project': f'{args.task}',
        'method': 'grid',
        'parameters': {
            'dataset': {
                'values': DATASETS
            },
        }
    }

    with open(f'{args.task}.yaml', 'w') as yaml_file:
        yaml.dump(yaml_content, yaml_file, default_flow_style=False)
        