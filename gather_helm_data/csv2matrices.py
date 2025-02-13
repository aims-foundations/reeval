import os
import pandas as pd
import pickle
import torch
import argparse
from tqdm import tqdm
from configs import TASK2METRICS, THRESHOLDS
    
def process_joint_matrices(csv_dir, output_dir):
    all_model_infos = pd.read_csv(f"{csv_dir}/model_df.csv")
    total_models = len(all_model_infos)
    matrices = None
    combined_instance_info = None
    for benchmark, scenario_metrics in TASK2METRICS.items():
        print(f"Processing {benchmark}")
        instance_info = pd.read_csv(f"{csv_dir}/{benchmark}/instances.csv")
        if os.path.exists(f"{csv_dir}/{benchmark}/responses.csv"):
            responses = pd.read_csv(f"{csv_dir}/{benchmark}/responses.csv")
        else:
            responses = pickle.load(open(f"{csv_dir}/{benchmark}/responses.pkl", "rb"))
        scenario_info = pd.read_csv(f"{csv_dir}/{benchmark}/scenarios.csv")
        sname2sid = {row["name"]:row["scenarios_id"] for _, row in scenario_info.iterrows()}
        
        for scenario_name, metric_name in scenario_metrics.items():
            print(f"  + Processing {scenario_name}")
            
            # Filter instances row in instance_info
            scenario_instance = instance_info[instance_info["scenarios_id"] == sname2sid[scenario_name]]
            iid2index = {iid: index for index, iid in enumerate(scenario_instance["instance_id"])}
            
            # Filter responses row in responses
            scenario_responses = responses[responses["scenarios_id"] == sname2sid[scenario_name]]
            
            bm_matrix = torch.empty(total_models, len(scenario_instance), dtype=torch.int8)
            bm_matrix.fill_(-1)

            if isinstance(metric_name, str):
                for i, row in tqdm(scenario_responses.iterrows(), total=len(scenario_responses)):
                    if pd.isna(row[metric_name]):
                        continue
                    bm_matrix[int(row["model_id"]), iid2index[int(row["instance_id"])]] = int(
                        float(row[metric_name]) >= THRESHOLDS[metric_name]
                    )
            else:
                for i, row in tqdm(scenario_responses.iterrows(), total=len(scenario_responses)):
                    for metric in metric_name:
                        if metric in row and row[metric]:
                            bm_matrix[int(row["model_id"]), iid2index[int(row["instance_id"])]] = int(
                                float(row[metric]) >= THRESHOLDS[metric]
                            )
                            break
                            
            if matrices is None:
                matrices = bm_matrix
            else:
                matrices = torch.cat([matrices, bm_matrix], dim=1)

            if combined_instance_info is None:
                combined_instance_info = scenario_instance
            else:
                combined_instance_info = pd.concat([combined_instance_info, scenario_instance])
                
    os.makedirs(f"{output_dir}/combined_data", exist_ok=True)
    torch.save(matrices, f"{output_dir}/combined_data/response_matrix.pt")
    combined_instance_info.to_csv(f"{output_dir}/combined_data/question_keys.csv", index=False)
    all_model_infos.to_csv(f"{output_dir}/combined_data/model_keys.csv", index=False)
        
def process_split_matrices(csv_dir, output_dir):
    all_model_infos = pd.read_csv(f"{csv_dir}/model_df.csv")
    total_models = len(all_model_infos)
    for benchmark, scenario_metrics in TASK2METRICS.items():
        print(f"Processing {benchmark}") # HELM Classic, Lite, etc.
        instance_info = pd.read_csv(f"{csv_dir}/{benchmark}/instances.csv")
        all_scenario_info = pd.read_csv(f"{csv_dir}/{benchmark}/scenarios.csv")
        if os.path.exists(f"{csv_dir}/{benchmark}/responses.csv"):
            responses = pd.read_csv(f"{csv_dir}/{benchmark}/responses.csv")
        else:
            responses = pickle.load(open(f"{csv_dir}/{benchmark}/responses.pkl", "rb"))
        
        for _, scenario_row in all_scenario_info.iterrows():
            print(f"  + Processing {scenario_row['name']}") # babi_qa, bbq, etc. under HELM Classic
            scenario_name = scenario_row["name"]
            if scenario_name not in scenario_metrics:
                continue
            result_folder = f"{output_dir}/{benchmark}/{scenario_name}"
            if not os.path.exists(result_folder):
                os.makedirs(result_folder, exist_ok=True)
                
            if os.path.exists(f"{result_folder}/response_matrix.pt"):
                print(f"  + Skipping {scenario_name}")
                continue
            metric_name = scenario_metrics[scenario_name]
                
            # Filter instances row in instance_info
            scenario_instance = instance_info[instance_info["scenarios_id"] == scenario_row["scenarios_id"]]
            iid2index = {iid: index for index, iid in enumerate(scenario_instance["instance_id"])}
            # instance_id -> index (start from 0 for a single scenario)
            
            # Filter responses row in responses
            scenario_responses = responses[responses["scenarios_id"] == scenario_row["scenarios_id"]]
            
            bm_matrix = torch.empty(total_models, len(scenario_instance), dtype=torch.int8)
            bm_matrix.fill_(-1)

            if isinstance(metric_name, str):
                for i, row in tqdm(scenario_responses.iterrows(), total=len(scenario_responses)):
                    if pd.isna(row[metric_name]):
                        continue
                    bm_matrix[int(row["model_id"]), iid2index[int(row["instance_id"])]] = int(
                        float(row[metric_name]) >= THRESHOLDS[metric_name]
                    )
            else:
                for i, row in tqdm(scenario_responses.iterrows(), total=len(scenario_responses)):
                    for metric in metric_name:
                        if metric in row and row[metric]:
                            bm_matrix[int(row["model_id"]), iid2index[int(row["instance_id"])]] = int(
                                float(row[metric]) >= THRESHOLDS[metric]
                            )
                            break
                
            # Remove models with no responses
            valid_models = torch.any(bm_matrix != -1, dim=1)
            bm_matrix = bm_matrix[valid_models]
            
            valid_model_indices = torch.where(valid_models)[0].tolist()
            scenario_model_infos = all_model_infos.iloc[valid_model_indices]

            torch.save(bm_matrix, f"{result_folder}/response_matrix.pt")
            scenario_instance.to_csv(f"{result_folder}/question_keys.csv", index=False)
            scenario_model_infos.to_csv(f"{result_folder}/model_keys.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_dir", type=str, default="CSV")
    parser.add_argument("--output_dir", type=str, default="matrices")
    parser.add_argument("--mode", type=str, required=True, choices=["joint", "split"])
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
        
    if args.mode == "joint":
        process_joint_matrices(args.csv_dir, args.output_dir)
        
    elif args.mode == "split":
        process_split_matrices(args.csv_dir, args.output_dir)