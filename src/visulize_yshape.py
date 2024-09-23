import os
import pandas as pd
from utils import DATASETS
import matplotlib.pyplot as plt
from tueplots import bundles
plt.rcParams.update(bundles.icml2022())
plt.style.use('seaborn-v0_8-paper')

if __name__ == "__main__":
    plot_dir = f'../plot/others'
    os.makedirs(plot_dir, exist_ok=True)
    
    question_nums, testtaker_nums = [], []
    for dataset in DATASETS:
        y = pd.read_csv(f'../data/pre_calibration/{dataset}/matrix.csv', index_col=0).values
        testtaker_nums.append(y.shape[0])
        question_nums.append(y.shape[1])
    
    sorted_by_question = sorted(zip(DATASETS, question_nums), key=lambda x: x[1])
    sorted_datasets, sorted_question_nums = zip(*sorted_by_question)
    plt.figure(figsize=(20, 6))
    plt.bar(sorted_datasets, sorted_question_nums)
    plt.xticks(rotation=30, ha='right', fontsize=45)
    plt.tick_params(axis='both', labelsize=35)
    plt.ylabel(r'Number of Questions', fontsize=35)
    plt.savefig(f"{plot_dir}/question_nums.png", dpi = 300, bbox_inches='tight')
    plt.close()

    sorted_by_testtaker = sorted(zip(DATASETS, testtaker_nums), key=lambda x: x[1])
    sorted_datasets, sorted_testtaker_nums = zip(*sorted_by_testtaker)
    plt.figure(figsize=(20, 6))
    plt.bar(sorted_datasets, sorted_testtaker_nums)
    plt.xticks(rotation=30, ha='right', fontsize=45)
    plt.tick_params(axis='both', labelsize=35)
    plt.ylabel(r'Number of Test Takers', fontsize=35)
    plt.savefig(f"{plot_dir}/testtaker_nums.png", dpi = 300, bbox_inches='tight')
    plt.close()