import os
import yaml

if __name__ == "__main__":
    input_dir = '../data/get_embed/'
    folder_list = [f for f in os.listdir(input_dir)]
    
    yaml_content = {
        'program': 'amor_calibration.py',
        'project': 'amor_calibration',
        'method': 'grid',
        'parameters': {
            'dataset': {
                'values': folder_list
            },
        }
    }

    with open('amor_calibration.yaml', 'w') as yaml_file:
        yaml.dump(yaml_content, yaml_file, default_flow_style=False)
        