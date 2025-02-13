import os
import json
import pickle
import pandas as pd
from tqdm import tqdm
from configs import RUNNING_ENVS, PROMPT_ENVS

BENCHMARKS = [
    # "air-bench",
    # "classic",
    # "cleva",
    # "decodingtrust",
    # # "heim",
    # "lite",
    "mmlu",
    # "instruct",
    # "image2structure",
    # "safety",
    # "thaiexam",
    # "vhelm",
]
VLM_BENCHMARKS = [
    # "heim",
    "image2structure",
    "vhelm",
]


def get_list_models(benchmark):
    schema_folder = "releases"
    if benchmark in VLM_BENCHMARKS:
        schema_folder = "runs"
    if os.path.exists(
        os.path.join(benchmark, schema_folder)
    ):
        latest_release = sorted(
            os.listdir(os.path.join(benchmark, schema_folder))
        )[-1]
        
        if os.path.exists(
            os.path.join(benchmark, schema_folder, latest_release, "schema.json")
        ):
            model_df = json.load(
                open(
                    os.path.join(
                        benchmark, schema_folder, latest_release, "schema.json"
                    )
                )
            )
            return pd.DataFrame(model_df["models"])

        
    # In case the model schema is not found
    assert False, f"Model schema not found for {benchmark}"
    
    
def merge_model_df(model_dfs):
    model_df = pd.concat(model_dfs)
    model_df = model_df.drop_duplicates(subset=["name"])
    
    # Reset the index
    model_df = model_df.reset_index(drop=True)
    
    return model_df

def get_run_path(benchmark):
    if os.path.exists(
        os.path.join(benchmark, "releases")
    ):
        latest_release = sorted(
            os.listdir(os.path.join(benchmark, "releases"))
        )[-1]
        
        if os.path.exists(
            os.path.join(benchmark, "releases", latest_release, "runs_to_run_suites.json")
        ):
            folder_dict = json.load(
                open(
                    os.path.join(
                        benchmark, "releases", latest_release, "runs_to_run_suites.json"
                    )
                )
            )
            
            run_paths = []
            for run, suite in folder_dict.items():
                run_paths.append(os.path.join(benchmark, "runs", suite, run))
                
            return run_paths
        
    elif os.path.exists(
        os.path.join(benchmark, "runs")
    ):
        all_releases = sorted(
            os.listdir(os.path.join(benchmark, "runs"))
        )
        
        run_paths = []
        for suite in all_releases:
            if os.path.exists(
                os.path.join(benchmark, "runs", suite, "runs_to_run_suites.json")
            ):
                folder_dict = json.load(
                    open(
                        os.path.join(
                            benchmark, "runs", suite, "runs_to_run_suites.json"
                        )
                    )
                )
                
                for run, _ in folder_dict.items():
                    run_paths.append(os.path.join(benchmark, "runs", suite, run))
                
        return run_paths
        
    # In case the run paths are not found
    all_suites = os.listdir(os.path.join(benchmark, "runs"))
    # Filter out the suites that is folder and starting with letter "v"
    run_paths = [
        os.path.join(benchmark, "runs", suite)
        for suite in all_suites
        if os.path.isdir(os.path.join(benchmark, "runs", suite))
        and suite[0] != "v"
    ]
    
    return run_paths

def process_run(run_path, MODEL_NAME2ID, scenarios, instances, instance2index):
    responses = {
        "model_id": [],
        "scenarios_id": [],
        "instance_id": [],
        "temperature": [],
        "num_completions": [],
        "top_k_per_token": [],
        "max_tokens": [],
        "stop_sequences": [],
        "echo_prompt": [],
        "top_p": [],
        "presence_penalty": [],
        "frequency_penalty": [],
        "predicted_text": [],
        "base64_images": [],
    }
    # Get model and scenario info
    run_specs = json.load(open(os.path.join(run_path, "run_spec.json")))
    model_name = run_specs["adapter_spec"]["model"]
    model_id = MODEL_NAME2ID[model_name]
    
    if not run_specs["name"].startswith("legal_support"):
        scenario_name = run_specs["name"].split(":")[0]
    else:
        scenario_name = run_specs["name"].split(",")[0]
    if scenario_name not in scenarios["name"]:
        scenario_id = len(scenarios["name"])
        scenarios["scenarios_id"].append(scenario_id)
        scenarios["name"].append(scenario_name)
        
    else:
        scenario_id = scenarios["name"].index(scenario_name)
        # classic have mmlu, mmlu also have mmlu
    
    if os.path.exists(os.path.join(run_path, "instances.json")):
        if os.path.exists(os.path.join(run_path, "display_requests.json")) and os.path.exists(os.path.join(run_path, "display_predictions.json")):
            return process_response_1(responses, model_id, scenario_id, scenario_name, run_path, instances, instance2index)
        elif os.path.exists(os.path.join(run_path, "scenario_state.json")) and os.path.exists(os.path.join(run_path, "per_instance_stats.json")):
            return process_response_2(responses, model_id, scenario_id, scenario_name, run_path, instances, instance2index)
        return pd.DataFrame(responses)
    else:
        return pd.DataFrame(responses)

def process_response_1(responses, model_id, scenario_id, scenario_name, run_path, instances, instance2index):
    # Read the prompt and prediction info
    request_info = json.load(open(os.path.join(run_path, "display_requests.json")))
    prediction_info = json.load(open(os.path.join(run_path, "display_predictions.json")))
    instances_info = json.load(open(os.path.join(run_path, "instances.json")))
    stats = json.load(open(os.path.join(run_path, "stats.json")))
    
    # Get the list of metrics
    list_metrics = set([s["name"]["name"] for s in stats])
    list_metrics = list(list_metrics)
    if "num_completions" in list_metrics:
        list_metrics.remove("num_completions")
    responses.update({metric: [] for metric in list_metrics})
    
    # Sort the request and prediction info
    request_info = sorted(request_info, key=lambda x: x["instance_id"])
    prediction_info = sorted(prediction_info, key=lambda x: x["instance_id"])
    instances_info = sorted(instances_info, key=lambda x: x["id"])
    
    for request, prediction, per_inst in zip(request_info, prediction_info, instances_info):
        if request["instance_id"] != prediction["instance_id"]:
            continue
        
        if per_inst["id"] != prediction["instance_id"]:
            continue
        
        instance_index = f"{scenario_id}_{request['instance_id']}"
        if "data_augmentation=" in run_path:
            aug_type = run_path[run_path.find("data_augmentation=") + 18:].split(",")[0]
            instance_index += "_" + aug_type
        else:
            aug_type = ""
            
        if "mode=" in run_path:
            mode = run_path[run_path.find("mode=") + 5:].split(",")[0]
            instance_index += "_" + mode
        else:
            mode = ""
            
        if "num_prompt_tokens=" in run_path:
            num_prompt_tokens = run_path[run_path.find("num_prompt_tokens=") + 18:].split(",")[0]
            instance_index += "_" + num_prompt_tokens
        else:
            num_prompt_tokens = ""
            
        if "subject=" in run_path:
            subject = run_path[run_path.find("subject=") + 8:].split(",")[0]
            instance_index += "_" + subject
        else:
            subject = ""
            
        if "perturbation" in request:
            perturbation_type = request["perturbation"]["name"]
            instance_index += "_" + perturbation_type
        else:
            perturbation_type = ""
        
        if instance_index not in instance2index:
            instance_id = len(instances["prompt"])
            instances["instance_id"].append(instance_id)
            instances["scenarios_id"].append(scenario_id)
            instances["instance_sid"].append(prediction['instance_id'])
            instances["data_augmentation"].append(aug_type)
            instances["mode"].append(mode)
            instances["num_prompt_tokens"].append(num_prompt_tokens)
            instances["perturbation"].append(perturbation_type)
            instances["subject"].append(subject)
            instances["prompt"].append(request["request"]["prompt"]) # have few shot, have choices
            if per_inst["input"]["text"] == "":
                per_inst["input"]["text"] = request["request"]["prompt"]
            instances["raw_question"].append(per_inst["input"]["text"]) # no few shot, no choices
            instance2index[instance_index] = instance_id
        else:
            instance_id = instance2index[instance_index]
            
        responses["model_id"].append(model_id) # global model id
        responses["scenarios_id"].append(scenario_id)
        responses["instance_id"].append(instance_id)
        responses["temperature"].append(request["request"]["temperature"])
        responses["num_completions"].append(request["request"]["num_completions"])
        responses["top_k_per_token"].append(request["request"]["top_k_per_token"])
        responses["max_tokens"].append(request["request"]["max_tokens"])
        responses["stop_sequences"].append(request["request"]["stop_sequences"])
        responses["echo_prompt"].append(request["request"]["echo_prompt"])
        responses["top_p"].append(request["request"]["top_p"])
        responses["presence_penalty"].append(request["request"]["presence_penalty"])
        responses["frequency_penalty"].append(request["request"]["frequency_penalty"])
        
        responses["predicted_text"].append(prediction["predicted_text"])
        if "base64_images" not in prediction:
            responses["base64_images"].append("")
        else:
            responses["base64_images"].append(prediction["base64_images"])
        
        for stat_name in list_metrics:
            if stat_name not in prediction["stats"]:
                responses[stat_name].append(None)
                continue
            
            responses[stat_name].append(prediction["stats"][stat_name])
            
    # Set the ablation keys, e.g. fewshots, chain-of-thought, etc.
    if scenario_name in RUNNING_ENVS:
        folder_name = run_path.split("/")[-1]
        for key, list_possible_value in RUNNING_ENVS[scenario_name].items():
            success = False
            for value in list_possible_value:
                if f"{key}={value}" in folder_name:
                    responses["ablation_" + key] = [value] * len(responses["model_id"])
                    success = True
                    break
            if not success:
                responses["ablation_" + key] = [""] * len(responses["model_id"])
                    
    return pd.DataFrame(responses)
    

def process_response_2(responses, model_id, scenario_id, scenario_name, run_path, instances, instance2index):
    # Read the prompt and prediction info
    request_info = json.load(open(os.path.join(run_path, "scenario_state.json")))
    request_info = request_info["request_states"]
    prediction_info = json.load(open(os.path.join(run_path, "per_instance_stats.json")))
    instances_info = json.load(open(os.path.join(run_path, "instances.json")))
    stats = json.load(open(os.path.join(run_path, "stats.json")))
    
    # Get the list of metrics
    list_metrics = set([s["name"]["name"] for s in stats])
    list_metrics = list(list_metrics)
    if "num_completions" in list_metrics:
        list_metrics.remove("num_completions")
    responses.update({metric: [] for metric in list_metrics})
    
    # Sort the request and prediction info
    request_info = sorted(request_info, key=lambda x: x["instance"]["id"])
    instances_info = sorted(instances_info, key=lambda x: x["id"])
    if len(request_info) != len(prediction_info):
        prediction_info = join_prediction_info(prediction_info)
    prediction_info = sorted(prediction_info, key=lambda x: x["instance_id"])

    for request, prediction, per_inst in zip(request_info, prediction_info, instances_info):
        if request["instance"]["id"] != prediction["instance_id"]:
            continue
        
        instance_index = f"{scenario_id}_{prediction['instance_id']}"
        if "data_augmentation=" in run_path:
            aug_type = run_path[run_path.find("data_augmentation=") + 18:].split(",")[0]
            instance_index += "_" + aug_type
        else:
            aug_type = ""
            
        if "mode=" in run_path:
            mode = run_path[run_path.find("mode=") + 5:].split(",")[0]
            instance_index += "_" + mode
        else:
            mode = ""
            
        if "num_prompt_tokens=" in run_path:
            num_prompt_tokens = run_path[run_path.find("num_prompt_tokens=") + 18:].split(",")[0]
            instance_index += "_" + num_prompt_tokens
        else:
            num_prompt_tokens = ""
            
        if "subject=" in run_path:
            subject = run_path[run_path.find("subject=") + 8:].split(",")[0]
            instance_index += "_" + subject
        else:
            subject = ""
            
        if "perturbation" in request:
            perturbation_type = request["perturbation"]["name"]
            instance_index += "_" + perturbation_type
        else:
            perturbation_type = ""
        
        if instance_index not in instance2index:
            instance_id = len(instances["prompt"])
            instances["instance_id"].append(instance_id)
            instances["scenarios_id"].append(scenario_id)
            instances["instance_sid"].append(prediction['instance_id'])
            instances["data_augmentation"].append(aug_type)
            instances["mode"].append(mode)
            instances["num_prompt_tokens"].append(num_prompt_tokens)
            instances["perturbation"].append(perturbation_type)
            instances["subject"].append(subject)
            instances["prompt"].append(request["request"]["prompt"])
            if per_inst["input"]["text"] == "":
                per_inst["input"]["text"] = request["request"]["prompt"]
            instances["raw_question"].append(per_inst["input"]["text"])
            instance2index[instance_index] = instance_id
        else:
            instance_id = instance2index[instance_index]
            
        responses["model_id"].append(model_id)
        responses["scenarios_id"].append(scenario_id)
        responses["instance_id"].append(instance_id)
        responses["temperature"].append(request["request"]["temperature"])
        responses["num_completions"].append(request["request"]["num_completions"])
        responses["top_k_per_token"].append(request["request"]["top_k_per_token"])
        responses["max_tokens"].append(request["request"]["max_tokens"])
        responses["stop_sequences"].append(request["request"]["stop_sequences"])
        responses["echo_prompt"].append(request["request"]["echo_prompt"])
        responses["top_p"].append(request["request"]["top_p"])
        responses["presence_penalty"].append(request["request"]["presence_penalty"])
        responses["frequency_penalty"].append(request["request"]["frequency_penalty"])
        
        responses["predicted_text"].append(request["result"]["completions"][0]["text"])
        if "base64_images" not in prediction:
            responses["base64_images"].append("")
        else:
            responses["base64_images"].append(prediction["base64_images"])
        
        all_avai_stats = {stat["name"]["name"]: stat_idx for stat_idx, stat in enumerate(prediction["stats"])}
        for stat_name in list_metrics:
            if stat_name not in all_avai_stats:
                responses[stat_name].append(None)
                continue

            responses[stat_name].append(
                prediction["stats"][all_avai_stats[stat_name]]["sum"]
            )
            
    # Set the ablation keys, e.g. fewshots, chain-of-thought, etc.
    if scenario_name in RUNNING_ENVS:
        folder_name = run_path.split("/")[-1]
        for key, list_possible_value in RUNNING_ENVS[scenario_name].items():
            success = False
            for value in list_possible_value:
                if f"{key}={value}" in folder_name:
                    responses["ablation_" + key] = [value] * len(responses["model_id"])
                    success = True
                    break
            if not success:
                responses["ablation_" + key] = [""] * len(responses["model_id"])
                    
    return pd.DataFrame(responses)


def join_prediction_info(prediction_info):
    unique_instance_ids = [x["instance_id"] for x in prediction_info]
    unique_instance_ids = list(set(unique_instance_ids))
    new_prediction_info = []
    for instance_id in unique_instance_ids:
        instance_info = [x for x in prediction_info if x["instance_id"] == instance_id]
        if len(instance_info) == 1:
            new_prediction_info.append(instance_info[0])
            continue
        first_instance = instance_info[0]
        for inst in instance_info[1:]:
            first_instance["stats"] += inst["stats"]
        new_prediction_info.append(first_instance)
    return new_prediction_info
    
    
def merge_results(results):
    return pd.concat(results)

        
if __name__ == "__main__":
    if os.path.exists("CSV/model_df.csv"):
        print("Model df already exists")
        model_df = pd.read_csv("CSV/model_df.csv")
    else:
        model_dfs = []
        for benchmark in BENCHMARKS:
            model_df = get_list_models(benchmark)
            model_dfs.append(model_df)
        
        # Merge the model dfs
        model_df = merge_model_df(model_dfs)
        model_df['model_id'] = model_df.index
        
        # Save the model df
        model_df.to_csv("CSV/model_df.csv", index=False)
    
    # Create a dictionary of model name to model id
    MODEL_NAME2ID = dict(zip(model_df['name'], model_df['model_id']))
    
    for benchmark in BENCHMARKS: # BENCHMARKS are HELM Classic, Lite, etc.
        # Skip the benchmarks that are already processed
        if os.path.exists(f"CSV/{benchmark}/responses.pkl"):
            print(f"Skipping {benchmark}")
            continue
        
        # Get the run paths
        run_paths = get_run_path(benchmark)
        
        if run_paths is None or len(run_paths) == 0:
            continue
        
        # Sort the run paths
        run_paths = sorted(run_paths)
        
        # Process the runs
        run_results = []
        scenarios = {
            "scenarios_id": [],
            "name": [],
        } # scenarios are babi_qa, bbq, etc. under HELM Classic
        instances = {
            "instance_id": [],
            "scenarios_id": [],
            "instance_sid": [],
            "data_augmentation": [],
            "perturbation": [],
            "subject": [],
            "mode": [],
            "num_prompt_tokens": [],
            "prompt": [],
            "raw_question": [],
        }
        instance2index = {} 
        # for mmlu under HELM MMLU, instance2index = {
        #     "mmlu-scenario-id_id16_abstract-algebra": instance-id
        #     ...
        # }
        
        for run_path in tqdm(run_paths, desc=f"Processing {benchmark}"):
            if "instructions=none" in run_path:
                # Skip duplicate runs
                continue
            results = process_run(run_path, MODEL_NAME2ID, scenarios, instances, instance2index)
            run_results.append(results)
            
        # Merge and save the results
        run_results = merge_results(run_results)
        
        # Save the results, scenarios and instances
        os.makedirs(f"CSV/{benchmark}", exist_ok=True)
        # run_results.to_csv(f"CSV/{benchmark}/responses.csv", index=False)
        pickle.dump(run_results, open(f"CSV/{benchmark}/responses.pkl", "wb"))
        pd.DataFrame(scenarios).to_csv(f"CSV/{benchmark}/scenarios.csv", index=False)
        pd.DataFrame(instances).to_csv(f"CSV/{benchmark}/instances.csv", index=False)
            