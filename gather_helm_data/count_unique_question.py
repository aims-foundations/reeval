import json
import os

root_dir = '/dfs/scratch0/nqduc/helm_jsons/classic/runs/v0.3.0'
all_scenario_dirs = [
        dir_name for dir_name in os.listdir(root_dir) if "model=cohere_medium-20220720" in dir_name
    ] + [
        "code:dataset=apps,model=openai_code-cushman-001"
    ] + [
        "code:dataset=humaneval,model=openai_code-cushman-001"
    ]

scenario_names = sorted(list(set([path.split(":")[0].split(",")[0] for path in all_scenario_dirs])))
print(scenario_names)
print(len(scenario_names))

for scenario_name in scenario_names:
    # all_scenario_dirs that start with scenario_name
    scenario_dirs = [path for path in all_scenario_dirs if path.startswith(scenario_name)]
    instance_paths = [f"{root_dir}/{scenario_dir}/instances.json" for scenario_dir in scenario_dirs]
    texts = []
    for instance_path in instance_paths:
        with open(instance_path, 'r') as file:
            data = json.load(file)
            for entry in data:
                text = entry["input"]["text"]
                texts.append(text)
    unique_texts = set(texts)
    print(f"scenario: {scenario_name}, questions: {len(texts)}, unique: {len(unique_texts)}")