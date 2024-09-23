import os
import yaml

if __name__ == "__main__":
    input_dir = '../data/nonamor_calibration/'
    folder_list = [f for f in os.listdir(input_dir)]
    
    yaml_content = {
        'program': 'nonid_test.py',
        'project': 'nonid_test',
        'method': 'grid',
        'parameters': {
            'dataset': {
                'values': folder_list
            },
        }
    }

    with open('nonid_test.yaml', 'w') as yaml_file:
        yaml.dump(yaml_content, yaml_file, default_flow_style=False)
        