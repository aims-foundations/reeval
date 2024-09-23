import os
import pandas as pd
from utils import DATASETS

if __name__ == "__main__":
    output_dir = "../plot/nonid_test"
    os.makedirs(output_dir, exist_ok=True)
    
    ctt_tags, ctt_t_stats, irt_tags, irt_t_stats = [], [], [], []
    for dataset in DATASETS:
        input_df = pd.read_csv(f"../data/nonid_test/{dataset}/nonid_test.csv")
        ctt_tag = input_df['ctt_tag'].values
        ctt_t_stat = input_df['ctt_t_stat'].values
        irt_tag = input_df['irt_tag'].values
        irt_t_stat = input_df['irt_t_stat'].values

        ctt_tags.append(ctt_tag)
        ctt_t_stats.append(ctt_t_stat)
        irt_tags.append(irt_tag)
        irt_t_stats.append(irt_t_stat)
    
    assert len(DATASETS) == len(ctt_tags) == len(irt_tags) == len(ctt_t_stats) == len(irt_t_stats)
    output_df = pd.DataFrame({
        'Dataset': DATASETS,
        'CTT Tag': ctt_tags,
        'CTT T Stats': ctt_t_stats,
        'IRT Tag': irt_tags,
        'IRT T Stats': irt_t_stats
    })
    output_df.to_latex(f"{output_dir}/nonid_test.tex", index=False)