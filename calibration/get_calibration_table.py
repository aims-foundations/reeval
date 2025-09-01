import torch
import numpy as np
import scipy.stats as st
import pandas as pd
def summarize_distr(distr):
    if  distr is None or len(distr) == 0:
        return {"mean": None, "std": None, "ci95": (None, None)}
    arr = np.array(distr, dtype=float)
    n = len(arr)
    mean = np.mean(arr)
    std = np.std(arr, ddof=1)  # sample std (unbiased)
    
    # t critical value for 95% CI
    ci_low, ci_high = st.t.interval(
        0.95, df=n-1, loc=mean, scale=std/np.sqrt(n)
    )
    
    return {"mean": mean, "std": std, "ci95": (ci_low, ci_high)}


    # mask_list = ["random_mask","random_row","date","size"]
    # dataset_list = ["official_provider","HELM"]
    # factor_list = [1, 2, 3, 4, 6, 8, 10, 15, 20, 25, 30, 35, 40, 45, 50]


metrics = ['auc','corr']
datasets = ["HELM","official_provider","everything"]
masking_methods = ["random_mask","random_row","date","size"]
factors = [i for i in range(1,51)]
potential_trial = [i for i in range(100)]

data_list = []

for metric in metrics:
    for dataset in datasets:
        for masking_method in masking_methods:
            for K_fit in factors:
                distr = {
                    "train_dist":[],
                    "test_dist":[],
                }
                
                for i in potential_trial:
                    try:
                        config_name = f"{dataset}_{masking_method}_k{K_fit}_i{i}"

                        train_perf_path = f"results/{metric}/train_{metric}_{config_name}.pt"
                        test_perf_path = f"results/{metric}/test_{metric}_{config_name}.pt"
                        train_perf = torch.load(train_perf_path)
                        test_perf = torch.load(test_perf_path)
                        if train_perf is not None:
                            distr['train_dist'].append(train_perf)
                        if test_perf is not None:
                            distr['test_dist'].append(test_perf)
                    except:
                        print(f"missing {config_name}")
                distr_summary = {
                    "train_dist":summarize_distr(distr['train_dist']),
                    "test_dist":summarize_distr(distr['test_dist'])
                }
                run_data = {
                    "metric":metric,
                    "dataset":dataset,
                    "masking_method":masking_method,
                    "K_fit":K_fit,
                    "distr_summary":distr_summary,
                }
                data_list.append(run_data)

                
flat_records = []
for entry in data_list:
    for split in ["train_dist", "test_dist"]:
        summary = entry["distr_summary"][split]
        flat_records.append({
            "metric": entry["metric"],
            "dataset": entry["dataset"],
            "masking_method": entry["masking_method"],
            "K_fit": entry["K_fit"],
            "split": split,  # train/test
            "mean": summary["mean"],
            "std": summary["std"],
            "ci95_low": summary["ci95"][0],
            "ci95_high": summary["ci95"][1],
        })

df = pd.DataFrame(flat_records)

# if you want to set a useful index:
# df.set_index(["metric", "dataset", "masking_method", "K_fit", "split"], inplace=True)

print(df.head())

pd.DataFrame(flat_records).to_csv("results/summary/partial_results_auc_corr.csv", index=False)

