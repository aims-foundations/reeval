import pandas as pd
import torch
import numpy as np
import random
import warnings
from embed_text_package.embed_text import Embedder
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tueplots import bundles
plt.rcParams.update(bundles.icml2022())
plt.style.use('seaborn-v0_8-paper')
import seaborn as sns

class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ELU(),
            nn.Linear(input_dim, input_dim),
            nn.ELU(),
            nn.Linear(input_dim, 2048),
            nn.ELU(),
            nn.Linear(2048, 1024),
            nn.ELU(),
            nn.Linear(1024, 1)
        )
    def forward(self, x):
        return self.model(x)
    
DESCRIPTION_MAP = {
    # 'synthetic_efficiency': '### DATASET: Synthetic efficiency, ### PUBLISH TIME: unknown, ### CONTENT: to better understand inference runtime performance of various models',
    'airbench': '### DATASET: AirBench, ### PUBLISH TIME: 2024, ### CONTENT: AI safety benchmark that aligns with emerging government regulations and company policies',
    'math': '### DATASET: MATH, ### PUBLISH TIME: 2021, ### CONTENT: for measuring mathematical problem solving on competition math problems with or without with chain-of-thought style reasoning',
    'mmlu': '### DATASET: MMLU (Massive Multitask Language Understanding), ### PUBLISH TIME: 2021, ### CONTENT: for knowledge-intensive question answering across 57 domains',
    'wikifact': '### DATASET: WikiFact, ### PUBLISH TIME: 2019, ### CONTENT: knowledge base completion, entity-relation-entity triples in natural language form, to more extensively test factual knowledge',
    'entity_data_imputation': '### DATASET: Data imputation, ### PUBLISH TIME: 2021, ### CONTENT: tests the ability to impute missing entities in a data table',
    'commonsense': '### DATASET: HellaSwag, ### PUBLISH TIME: 2019, ### CONTENT: commonsense reasoning in question answering',
    'quac': '### DATASET: QuAC (Question Answering in Context), ### PUBLISH TIME: 2018, ### CONTENT: question answering in the context of dialogues',
    'imdb': '### DATASET: IMDB, ### PUBLISH TIME: 2011, ### CONTENT: sentiment analysis in movie review',
    'bbq': '### DATASET: BBQ (Bias Benchmark for Question Answering), ### PUBLISH TIME: 2022, ### CONTENT: for measuring social bias in question answering in ambiguous and unambigous context',
    'twitter_aae': '### DATASET: TwitterAAE, ### PUBLISH TIME: 2016, ### CONTENT: for measuring language model performance in tweets as a function of speaker dialect, on African-American-aligned Tweets, on White-aligned Tweets',
    'truthful_qa': '### DATASET: TruthfulQA, ### PUBLISH TIME: 2022, ### CONTENT: for measuring model truthfulness and commonsense knowledge in question answering',
    # 'msmarco': '### DATASET: MSMARCO, ### PUBLISH TIME: 2016, ### CONTENT: for passage retrieval in information retrieval',
    'legal_support': '### DATASET: LegalSupport, ### PUBLISH TIME: unknown, ### CONTENT: measure fine-grained legal reasoning through reverse entailment.',
    'boolq': '### DATASET: boolq, ### PUBLISH TIME: 2019, ### CONTENT: binary (yes/no) question answering, passages from Wikipedia, questions from search queries',
    'narrative_qa': '### DATASET: NarrativeQA, ### PUBLISH TIME: 2017, ### CONTENT: for reading comprehension over narratives, passages are books and movie scripts',
    'real_toxicity_prompts': '### DATASET: RealToxicityPrompts, ### PUBLISH TIME: 2020, ### CONTENT: for measuring toxicity in prompted model generations',
    'bold': '### DATASET: BOLD (Bias in Open-Ended Language Generation Dataset), ### PUBLISH TIME: 2021, ### CONTENT: for measuring biases and toxicity in open-ended language generation',
    # 'gsm': '### DATASET: GSM8K (Grade school math word problems), ### PUBLISH TIME: 2021, ### CONTENT: for testing mathematical reasoning on grade-school math problems',
    'babi_qa': '### DATASET: bAbI, ### PUBLISH TIME: 2015, ### CONTENT: for measuring understanding and reasoning',
    # 'summarization_xsum': '### DATASET: XSUM, ### PUBLISH TIME: 2018, ### CONTENT: for text summarization of BBC news articles',
    'synthetic_reasoning_natural': '### DATASET: Synthetic reasoning (natural language), ### PUBLISH TIME: 2021, ### CONTENT: Synthetic reasoning tasks defined using simple natural language based on LIME',
    'dyck_language_np3': '### DATASET: Dyck, ### PUBLISH TIME: 2019, ### CONTENT: Scenario testing hierarchical reasoning through the Dyck formal languages',
    'civil_comments': '### DATASET: CivilComments, ### PUBLISH TIME: 2019, ### CONTENT: for toxicity detection',
    'lsat_qa': '### DATASET: LSAT, ### PUBLISH TIME: 2021, ### CONTENT: for measuring analytical reasoning on the Law School Admission Test',
    'raft': '### DATASET: RAFT (Real-world Annotated Few-Shot), ### PUBLISH TIME: 2021, ### CONTENT: meta-benchmark of 11 real-world text classification tasks',
    # 'code': '### DATASET: Code, ### PUBLISH TIME: 2021, ### CONTENT: for measuring competence on code challenges, for measuring functional correctness for synthesizing programs from docstrings',
    'entity_matching': '### DATASET: Entity matching, ### PUBLISH TIME: 2016, ### CONTENT: tests the ability to determine if two entities match',
    'synthetic_reasoning': '### DATASET: Synthetic reasoning, ### PUBLISH TIME: 2021, ### CONTENT: defined using abstract symbols based on LIME and simple natural language based on LIME',
}
DATASETS = list(DESCRIPTION_MAP.keys())

def item_response_fn_1PL(z3, theta):
    return 1 / (1 + torch.exp(-(theta + z3)))

def item_response_fn_1PL_np(z3, theta):
    return 1 / (1 + np.exp(-(theta + z3)))

def sample_mean_std(data: np.array):
    masked_data = data[data != -1]
    mean = np.mean(masked_data)
    sample_means = []
    for _ in range(100):
        indices = np.random.choice(
            len(masked_data), int(0.8 * masked_data.shape[0]), replace=False
        )
        sample_mean = np.mean(masked_data[indices])
        sample_means.append(sample_mean)
    sample_std = np.std(sample_means)
    return mean, sample_std

def set_seed(seed):
    random.seed(seed)
    # torch.backends.cudnn.deterministic=True
    # torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def split_indices(length):
    indices = np.arange(length)
    np.random.shuffle(indices)
    train_size = int(0.8 * len(indices))
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    return train_indices.tolist(), test_indices.tolist()

def get_embed(
    dataset,
    cols_to_be_embded = ['text'],
    bs = 1024,
    model_name="meta-llama/Meta-Llama-3-8B",
):
    embdr = Embedder()
    embdr.load(model_name)
    dataloader = DataLoader(dataset, batch_size=bs)
    emb = embdr.get_embeddings(
        dataloader, model_name, cols_to_be_embded
    )
    return emb['text']
    
def goodness_of_fit_1PL(
    z: torch.Tensor,
    theta: torch.Tensor,
    y: torch.Tensor,
    bin_size: int=6,
):
    assert y.shape[1] == z.shape[0], f'{y.shape[1]} != {z.shape[0]}'
    assert y.shape[0] == theta.shape[0], f'{y.shape[0]} != {theta.shape[0]}'

    bin_start, bin_end = torch.min(theta), torch.max(theta)
    bins = torch.linspace(bin_start, bin_end, bin_size+1)
    # print(bins) # [-3. -2. -1.  0.  1.  2.  3.]

    diff_list = []
    for i in range(z.shape[0]):
        single_z = z[i]
        y_col = y[:, i]

        for j in range(bins.shape[0] - 1):
            bin_mask = (theta >= bins[j]) & (theta < bins[j + 1]) & (y_col != -1)
            if bin_mask.sum() > 0: # bin not empty
                y_empirical = y_col[bin_mask].mean()

                theta_mid = (bins[j] + bins[j + 1]) / 2
                y_theoretical = item_response_fn_1PL(theta_mid, single_z).item()

                diff = 1 - abs(y_empirical - y_theoretical)
                diff_list.append(diff)

    diff_array = np.array(diff_list)
    mean_diff = np.mean(diff_array)
    return mean_diff, diff_array

def goodness_of_fit_1PL_plot(
    z: torch.Tensor,
    theta: torch.Tensor,
    y: torch.Tensor,
    plot_path: str,
    bin_size: int=6,
):
    mean_diff, diff_array = goodness_of_fit_1PL(z, theta, y, bin_size)
    
    sample_means = []
    for _ in range(100):
        indices = np.random.choice(
            len(diff_array), int(0.8 * len(diff_array)), replace=False
        )
        sample_mean = np.mean(diff_array[indices])
        sample_means.append(sample_mean)
    std_diff = np.std(sample_means)
    
    plt.figure(figsize=(10, 6))
    plt.hist(diff_array, bins=40, density=True, alpha=0.4)
    plt.xlabel(r'Difference between empirical and theoretical $P(y=1)$', fontsize=30)
    plt.ylabel(r'Goodness of fit', fontsize=30)
    plt.tick_params(axis='both', labelsize=25)
    plt.xlim(0, 1)
    plt.axvline(mean_diff, linestyle='--')
    plt.text(
        mean_diff, 
        plt.gca().get_ylim()[1], 
        f'{mean_diff:.2f} $\\pm$ {3 * std_diff:.2f}', 
        ha='center', 
        va='bottom', 
        fontsize=25
    )
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return mean_diff, std_diff

def theta_corr_ctt(
    theta: np.array,
    y: np.array,
):
    assert y.shape[0] == theta.shape[0], f'{y.shape[1]} != {theta.shape[0]}'
    ctt_scores = []
    for row in y:
        valid_values = row[row != -1]
        if len(valid_values) > 0:
            ctt_scores.append(np.mean(valid_values))
        else:
            ctt_scores.append(np.nan)
    ctt_scores = np.array(ctt_scores)
    
    if np.isnan(ctt_scores).any():
        warnings.warn("ctt_scores contains nan", UserWarning)
    mask = ~np.isnan(ctt_scores)
    theta_masked, ctt_scores_masked = theta[mask], ctt_scores[mask]
    
    if np.unique(ctt_scores_masked).size <= 3:
        warnings.warn(f"ctt_scores_masked has little value: {ctt_scores_masked}", UserWarning)
    corr = np.corrcoef(theta_masked, ctt_scores_masked)[0, 1]
    return corr, theta_masked, ctt_scores_masked

def theta_corr_ctt_plot(
    theta: np.array,
    y: np.array,
    plot_path: str,
):
    corr, theta_masked, ctt_scores_masked = theta_corr_ctt(theta, y)
    
    sample_corrs = []
    for _ in range(100):
        indices = np.random.choice(
            len(theta_masked), int(0.8 * len(theta_masked)), replace=False
        )
        sample_corr = np.corrcoef(theta_masked[indices], ctt_scores_masked[indices])[0, 1]
        sample_corrs.append(sample_corr)
    sample_std = np.std(sample_corrs)
    
    plt.figure(figsize=(18, 10))
    plt.scatter(theta_masked, ctt_scores_masked)
    plt.xlabel(r'$\theta$ from calibration', fontsize=45)
    plt.ylabel(r'CTT score', fontsize=45)
    plt.title(f'Correlation: {corr:.2f} $\\pm$ {3 * sample_std:.2f}', fontsize=45)
    plt.tick_params(axis='both', labelsize=35)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return corr, sample_std
    
def theta_corr_helm(
    theta: np.array,
    dataset: str,
):
    y_model_names = pd.read_csv(
        f'../data/pre_calibration/{dataset}/matrix.csv', 
        index_col=0
    ).index.tolist()
    
    helm_df = pd.read_csv(f'../data/gather_data/crawl_real/helm_score/{dataset}.csv')
    helm_models = helm_df['model_name'].tolist()
    helm_models = [HELM_MODEL_MAP[m] if m in HELM_MODEL_MAP else m for m in helm_models]
    helm_scores = helm_df['score'].values
    
    assert helm_scores.shape[0] == theta.shape[0]
    assert set(helm_models) == set(y_model_names)
    
    helm_df_aligned = pd.DataFrame({'model_name': helm_models, 'score': helm_scores})
    theta_df_aligned = pd.DataFrame({'model_name': y_model_names, 'theta': theta})
    merged_df = pd.merge(helm_df_aligned, theta_df_aligned, on='model_name', how='inner')

    aligned_helm_scores = merged_df['score'].values
    aligned_theta = merged_df['theta'].values
    
    corr = np.corrcoef(aligned_theta, aligned_helm_scores)[0, 1]
    return corr, aligned_theta, aligned_helm_scores

def theta_corr_helm_plot(
    theta: np.array,
    dataset: np.array,
    plot_path: str,
):
    corr, theta, helm_scores = theta_corr_helm(theta, dataset)
    
    sample_corrs = []
    for _ in range(100):
        indices = np.random.choice(
            len(theta), int(0.8 * len(theta)), replace=False
        )
        sample_corr = np.corrcoef(theta[indices], helm_scores[indices])[0, 1]
        sample_corrs.append(sample_corr)
    sample_std = np.std(sample_corrs)
    
    plt.figure(figsize=(18, 10))
    plt.scatter(theta, helm_scores)
    plt.xlabel(r'$\theta$ from calibration', fontsize=45)
    plt.ylabel(r'HELM score', fontsize=45)
    plt.title(f'Correlation: {corr:.2f} $\\pm$ {3 * sample_std:.2f}', fontsize=45)
    plt.tick_params(axis='both', labelsize=35)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return corr, sample_std

def error_bar_plot_single(
    datasets, 
    means,
    stds, 
    plot_path,
    xlabel,
    xlim_upper=1.1
):
    datasets = [PLOT_NAME_MAP[dataset] for dataset in datasets]
    sorted_data = sorted(zip(datasets, means, stds), key=lambda x: x[1])
    datasets, means, stds = zip(*sorted_data)
    stds_mul3 = [s*3 for s in stds]
   
    fig, ax = plt.subplots(figsize=(8, 18))
    ax.barh(
        datasets, means, xerr=[np.zeros(len(datasets)), stds_mul3],
        capsize=5, color='blue', alpha=0.4,
        error_kw={'elinewidth': 1, 'capthick': 1, 'ecolor': 'blue'}
    )
    
    ax.set_xlabel(xlabel, fontsize=35)
    ax.tick_params(axis='both', labelsize=25)
    ax.set_xlim(0, xlim_upper)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
def error_bar_plot_double(
    datasets, 
    means_train, stds_train, 
    means_test, stds_test,
    plot_path,
    xlabel,
    xlim_upper=1.1,
    plot_std=True,
    average_line=False,
):  
    datasets = [PLOT_NAME_MAP[dataset] for dataset in datasets]
    sorted_data = sorted(
        zip(datasets, means_train, stds_train, means_test, stds_test),
        key=lambda x: x[3]
    )
    datasets, means_train, stds_train, means_test, stds_test = zip(*sorted_data)
    fig, ax = plt.subplots(figsize=(8, 18))

    if plot_std:
        stds_train_mul3 = [s*3 for s in stds_train]
        stds_test_mul3 = [s*3 for s in stds_test]
        ax.barh(
            datasets, means_train, xerr=[np.zeros(len(datasets)), stds_train_mul3],
            capsize=5, color='blue', alpha=0.4,
            error_kw={'elinewidth': 1, 'capthick': 1, 'ecolor': 'blue'}
        )
        ax.barh(
            datasets, means_test, xerr=[np.zeros(len(datasets)), stds_test_mul3],
            capsize=5, color='orange', alpha=0.4,
            error_kw={'elinewidth': 2, 'capthick': 2, 'ecolor': 'orange'}
        )
    else:
        ax.barh(
            datasets, means_train,
            color='blue', alpha=0.4
        )
        ax.barh(
            datasets, means_test,
            color='orange', alpha=0.4
        )
        print("")
        print(xlabel)
        improvements = []
        for dataset, mse_train, mse_test in zip(datasets, means_train, means_test):
            improvement = (mse_train - mse_test) / mse_train
            improvements.append((dataset, improvement))
        
        improvements.sort(key=lambda x: x[1], reverse=True)
        # print mean improvement
        print(f'Mean improvement: {np.mean([improvement for _, improvement in improvements])}')
        for dataset, improvement in improvements:
            print(f'{dataset}: {improvement}')

    if average_line:
        avg_train = np.mean(means_train)
        avg_test = np.mean(means_test)
        ax.axvline(avg_train, color='blue', linestyle='--', linewidth=2)
        ax.axvline(avg_test, color='orange', linestyle='--', linewidth=2)
        max_y = len(datasets)-1 
        ax.text(avg_train-0.5, max_y, f'{avg_train:.2f}', color='blue', fontsize=25, ha='center')
        ax.text(avg_test+0.5, max_y, f'{avg_test:.2f}', color='orange', fontsize=25, ha='center')

    ax.set_xlabel(xlabel, fontsize=35)
    ax.tick_params(axis='both', labelsize=25)
    ax.set_xlim(0, xlim_upper)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
def amorz_corr_nonamorz(
    z_amor: np.array,
    z_nonamor: np.array,
):
    assert z_amor.shape == z_nonamor.shape, f'{z_amor.shape} != {z_nonamor.shape}'
    z_corr = np.corrcoef(z_amor, z_nonamor)[0, 1]
    return z_corr

def plot_corr(
    data1,
    data2,
    plot_path,
    title,
    xlabel,
    ylabel,
):
    corr = np.corrcoef(data1, data2)[0, 1]
    plt.figure(figsize=(6, 6))
    plt.scatter(data1, data2, color='blue')
    plt.xlabel(xlabel, fontsize=25)
    plt.ylabel(ylabel, fontsize=25)
    plt.title(
        # title.format(corr),
        title,
        fontsize=25
    )
    plt.plot([0, 1], [0, 1], linestyle='--', color='black')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tick_params(axis='both', labelsize=16)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_corr_double(
    data1_train,
    data1_test,
    data2_train,
    data2_test,
    plot_path,
    xlabel,
    ylabel,
):
    corr_train = np.corrcoef(data1_train, data2_train)[0, 1]
    corr_test = np.corrcoef(data1_test, data2_test)[0, 1]
    plt.figure(figsize=(6, 6))
    plt.scatter(data1_train, data2_train, color='blue', label='Train')
    plt.scatter(data1_test, data2_test, color='red', label='Test')
    plt.plot([0, 1], [0, 1], linestyle='--', color='black')
    plt.xlabel(xlabel, fontsize=25)
    plt.ylabel(ylabel, fontsize=25)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title(
        r'Goodness of Fit',
        # r'Goodness of Fit. $\rho_\mathrm{{train}}$ = {:.2f}, $\rho_\mathrm{{test}}$ = {:.2f}'.format(corr_train, corr_test),
        fontsize=25
    )
    plt.tick_params(axis='both', labelsize=16)
    plt.legend(fontsize=16)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_bar(
    datasets,
    nums,
    plot_path,
    ylabel,
    exp_axis=False
):
    datasets = [PLOT_NAME_MAP[dataset] for dataset in datasets]
    sorted_by_nums = sorted(zip(datasets, nums), key=lambda x: x[1])
    sorted_datasets, sorted_nums = zip(*sorted_by_nums)
    plt.figure(figsize=(25, 10))
    bars = plt.bar(sorted_datasets, sorted_nums)
    plt.xticks(rotation=30, ha='right', fontsize=35)
    plt.tick_params(axis='both', labelsize=35)
    plt.ylabel(ylabel, fontsize=35)
    for bar, num in zip(bars, sorted_nums):
        height = bar.get_height()
        if height >= 1000:
            plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height/1000:.1f}k', 
                     ha='center', va='bottom', fontsize=20)
        else:
            plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height}', 
                     ha='center', va='bottom', fontsize=20)
    if exp_axis:
        plt.yscale('log')
    plt.savefig(plot_path, dpi = 300, bbox_inches='tight')
    plt.close()

def plot_hist(
    data,
    plot_path,
    ylabel,
):
    plt.figure(figsize=(6, 6))
    plt.hist(data, bins=30, density=True, alpha=0.4)
    mean_value = np.mean(data)
    plt.axvline(mean_value, linestyle='--', linewidth=2)
    plt.text(
        mean_value, 
        plt.gca().get_ylim()[1], 
        f'{mean_value:.2f}',
        fontsize=16, 
        ha='center'
    )
    plt.ylabel(ylabel, fontsize=25)
    plt.tick_params(axis='both', labelsize=16)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_rewards(rewards, plot_path):
    plt.figure(figsize=(6, 6))
    steps = range(0, 100, 10) 
    for i in range(len(rewards[0])):
        prompt_rewards = [reward[i] for reward in rewards]
        plt.plot(steps, prompt_rewards, marker='o')
    plt.ylabel(r'Reward', fontsize=25)
    plt.tick_params(axis='both', labelsize=16)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
def plot_loss(
    losses,
    plot_path,
    ylabel,
):
    plt.figure(figsize=(6, 6))
    plt.plot(losses)
    plt.tick_params(axis='both', labelsize=16)
    plt.ylabel(ylabel, fontsize=25)
    plt.ylim(0, 10)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_cat(
    randoms,
    cats,
    plot_path,
    ylabel,
):
    plt.figure(figsize=(6, 6))
    plt.plot(randoms, label='Random', color='red', linewidth=3)
    plt.plot(cats, label='Fisher', color='blue', linewidth=3)
    plt.tick_params(axis='both', labelsize=25)
    plt.ylabel(ylabel, fontsize=25)
    plt.legend(fontsize=25)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_hard_easy(theta_hats_all, y_means_all, theta, y, plot_path):
    plt.figure(figsize=(8, 6))
    plt.hist(theta_hats_all, bins=40, color='red', alpha=0.2, label='IRT Estimation', density=True)
    plt.hist(y_means_all, bins=40, color='blue', alpha=0.2, label='CTT Estimation', density=True)
    plt.axvline(x=theta, color='red', linestyle='-', linewidth=2)
    plt.axvline(x=y.mean().item() * 6 - 3, color='blue', linewidth=2)
    sns.kdeplot(theta_hats_all, color='red', linewidth=2, bw_adjust=2)
    plt.xlabel(r'Ability', fontsize=25)
    plt.ylabel(r'Density', fontsize=25)
    plt.legend(fontsize=20)
    plt.tick_params(axis='both', labelsize=20)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    
PLOT_NAME_MAP = {
    'wikifact': 'wikifact',
    'entity_data_imputation': 'ent_data_imp',
    'commonsense': 'commonsense',
    'quac': 'quac',
    'imdb': 'imdb',
    'bbq': 'bbq',
    'math': 'math',
    'twitter_aae': 'twitter_aae',
    'truthful_qa': 'truthful_qa',
    'legal_support': 'legal_support',
    'boolq': 'boolq',
    'narrative_qa': 'narrative_qa',
    'real_toxicity_prompts': 'real_toxicity',
    'bold': 'bold',
    'babi_qa': 'babi_qa',
    'synthetic_reasoning_natural': 'syn_reason_nat',
    'dyck_language_np3': 'dyck',
    'civil_comments': 'civil_comments',
    'lsat_qa': 'lsat_qa',
    'raft': 'raft',
    'entity_matching': 'entity_match',
    'synthetic_reasoning': 'syn_reason',
    'mmlu': 'mmlu',
    'airbench': 'airbench',
}

HELM_MODEL_MAP = {
    'text-davinci-002': 'openai_text-davinci-002',
    'text-babbage-001': 'openai_text-babbage-001',
    'ada (350M)': 'openai_ada',
    'text-ada-001': 'openai_text-ada-001',
    'babbage (1.3B)': 'openai_babbage',
    'T0pp (11B)☠': 'together_t0pp',
    'text-davinci-003': 'openai_text-davinci-003',
    'text-curie-001': 'openai_text-curie-001',
    'davinci (175B)': 'openai_davinci',
    'curie (6.7B)': 'openai_curie',
}


