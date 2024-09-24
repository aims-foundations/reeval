import argparse
import yaml
import sys
sys.path.append('..')
from utils import DATASETS

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=str, required=True)
    args = parser.parse_args()
    
    yaml_content = {
        'program': f'{args.project}.py',
        'project': f'{args.project}',
        'method': 'grid',
        'parameters': {
            'dataset': {
                'values': DATASETS
            },
        }
    }

    with open(f'{args.project}.yaml', 'w') as yaml_file:
        yaml.dump(yaml_content, yaml_file, default_flow_style=False)
        