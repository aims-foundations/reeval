import json
import os
import re
from collections import defaultdict

def parse_string(input_str):
    # Split the string by commas and colons
    pairs = re.split(r'[,:]', input_str)
    return pairs

def sort_dict(d):
    """ Recursively sort a dictionary by its keys at each level. """
    if isinstance(d, dict):
        return {key: sort_dict(value) for key, value in sorted(d.items())}
    return d

root_dir_1 = '/dfs/scratch0/nqduc/helm_jsons/classic/runs/v0.3.0'
root_dir_2 = '/dfs/scratch0/nqduc/helm_jsons/classic/runs/v0.4.0'
all_paths = [
        dir_name for dir_name in os.listdir(root_dir_1)
    ] + [
        dir_name for dir_name in os.listdir(root_dir_2)
    ]

parsed_all_paths = [parse_string(dir) for dir in all_paths]
scenario_names = sorted(list(set([dir[0] for dir in parsed_all_paths])))
print(scenario_names) # there is a folder called data_overlap?
print(len(scenario_names))

parsed_all_paths.remove(["data_overlap"])
subscenario_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
for parsed_all_path in parsed_all_paths:
    scenario = parsed_all_path[0]
    for entry in parsed_all_path[1:]:
        subscenario_name, subscenario_content = entry.split('=')
        if subscenario_name != "model":
            subscenario_stats[scenario][subscenario_name][subscenario_content] += 1

subscenario_stats = sort_dict(subscenario_stats)
subscenario_stats = {key: {subkey: dict(value) for subkey, value in value.items()} for key, value in subscenario_stats.items()}
output_file = 'subscenario_stats.json'
with open(output_file, 'w') as f:
    json.dump(subscenario_stats, f, indent=2)
