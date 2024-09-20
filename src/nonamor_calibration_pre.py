import os
import yaml

if __name__ == "__main__":
    input_dir = '../data/pre_calibration/'
    folder_list = [f for f in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, f))]
    
    yaml_content = {
        'program': 'nonamor_calibration.py',
        'project': 'nonamor_calibration',
        'method': 'grid',
        'parameters': {
            'dataset': {
                'values': folder_list
            },
        }
    }

    with open('nonamor_calibration.yaml', 'w') as yaml_file:
        yaml.dump(yaml_content, yaml_file, default_flow_style=False)
        