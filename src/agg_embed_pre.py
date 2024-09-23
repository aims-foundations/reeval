import yaml
from utils import DESCRIPTION_MAP


if __name__ == "__main__":
    datasets = list(DESCRIPTION_MAP.keys())
    
    yaml_content = {
        'program': 'agg_embed.py',
        'project': 'agg_embed',
        'method': 'grid',
        'parameters': {
            'dataset': {
                'values': datasets
            },
        }
    }

    with open('agg_embed.yaml', 'w') as yaml_file:
        yaml.dump(yaml_content, yaml_file, default_flow_style=False)
        