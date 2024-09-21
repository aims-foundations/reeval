import os
import yaml

if __name__ == "__main__":
    input_dir = '../data/pre_calibration/'
    folder_list = [f for f in os.listdir(input_dir) if f != "synthetic"]
    
    yaml_content = {
        'program': 'get_embed.py',
        'project': 'get_embed',
        'method': 'grid',
        'parameters': {
            'dataset': {
                'values': folder_list
            },
        }
    }

    with open('get_embed.yaml', 'w') as yaml_file:
        yaml.dump(yaml_content, yaml_file, default_flow_style=False)
        