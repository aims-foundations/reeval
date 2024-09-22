import os
import yaml

if __name__ == "__main__":
    input_dir = '../data/nonamor_calibration/'
    folder_list = [f for f in os.listdir(input_dir) if f != "synthetic"]
    
    yaml_content = {
        'program': 'plugin_regression.py',
        'project': 'plugin_regression',
        'method': 'grid',
        'parameters': {
            'dataset': {
                'values': folder_list
            },
        }
    }

    with open('plugin_regression.yaml', 'w') as yaml_file:
        yaml.dump(yaml_content, yaml_file, default_flow_style=False)
        