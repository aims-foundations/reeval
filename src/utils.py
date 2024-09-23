from matplotlib import gridspec
import torch
import numpy as np
import random
import jax.numpy as jnp
import warnings
from scipy.stats import ttest_ind
from embed_text_package.embed_text import Embedder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tueplots import bundles
plt.rcParams.update(bundles.icml2022())
plt.style.use('seaborn-v0_8-paper')

def item_response_fn_1PL(z3, theta):
    return 1 / (1 + torch.exp(-(theta + z3)))

def item_response_fn_1PL_jnp(z3, theta):
    return 1 / (1 + jnp.exp(-(theta + z3)))
    
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
    train_indices, test_indices = indices[:train_size], indices[train_size:]
    return train_indices, test_indices

def get_embed(
    dataset,
    cols_to_be_embded = ['text'],
    bs = 1024,
    model_name="meta-llama/Meta-Llama-3-8B",
):
    embdr = Embedder()
    embdr.load(model_name)
    dataloader = DataLoader(dataset, batch_size=bs)
    emb = embdr.get_embeddings(dataloader, model_name, cols_to_be_embded)
    return emb['text']

def bootstrap_mean_std(data: np.array):
    mean = np.mean(data)
    bootstrap_means = []
    for _ in range(1000):
        bootstrap_sample = np.random.choice(data, size=int(0.8 * data.shape[0]), replace=True)
        bootstrap_means.append(np.mean(bootstrap_sample))
    std_bootstrap = np.std(bootstrap_means)
    return mean, std_bootstrap
    
def perform_t_test(sample_1, sample_2, label=""):
    print(f"{label} T-test:")
    print(f"Null Hypothesis (H0): The means of the two samples are equal.")
    print(f"Alternative Hypothesis (H1): The means of the two samples are not equal.")
    t_stat, p_value = ttest_ind(sample_1, sample_2)
    print(f"t_stat = {t_stat}, p_value = {p_value}")
    if p_value < 0.05:
        print(f"Reject the null hypothesis for {label}.")
    else:
        print(f"Fail to reject the null hypothesis for {label}.")

def goodness_of_fit_1PL(
    z: torch.Tensor,
    theta: torch.Tensor,
    y: torch.Tensor,
    bin_size: int=7,
):
    assert y.shape[1] == z.shape[0], f'{y.shape[1]} != {z.shape[0]}'
    assert y.shape[0] == theta.shape[0], f'{y.shape[0]} != {theta.shape[0]}'

    bin_start, bin_end = torch.min(theta), torch.max(theta)
    bins = torch.linspace(bin_start, bin_end, bin_size)
    print(bins) # [-3. -2. -1.  0.  1.  2.  3.]

    diff_list = []
    for i in range(z.shape[0]):
        single_z = z[i]
        y_col = y[:, i]

        for j in range(bins.shape[0] - 1):
            bin_mask = (theta >= bins[j]) & (theta < bins[j + 1])
            if bin_mask.sum() > 0: # bin not empty
                y_empirical = y_col[(bin_mask) & (y_col != -1)].mean()

                theta_mid = (bins[j] + bins[j + 1]) / 2
                y_theoretical = item_response_fn_1PL(theta_mid, single_z).item()

                diff = abs(y_empirical - y_theoretical)
                diff_list.append(diff)

    diff_array = np.array(diff_list)
    mean_diff = np.mean(diff_array)
    return mean_diff, diff_list

def goodness_of_fit_1PL_plot(
    z: torch.Tensor,
    theta: torch.Tensor,
    y: torch.Tensor,
    plot_path: str,
    bin_size: int=7,
):
    mean_diff, diff_list = goodness_of_fit_1PL(z, theta, y, bin_size)
    std_diff = np.std(diff_list)
    plt.figure(figsize=(10, 6))
    plt.hist(diff_list, bins=40, density=True, alpha=0.4)
    plt.xlabel(r'Difference between empirical and theoretical $P(y=1)$', fontsize=30)
    plt.tick_params(axis='both', labelsize=25)
    plt.xlim(0, 1)
    plt.axvline(mean_diff, linestyle='--')
    plt.text(mean_diff, plt.gca().get_ylim()[1], f'{mean_diff:.2f} $\\pm$ {3 * std_diff:.2f}', ha='center', va='bottom', fontsize=25)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
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
    assert ctt_scores.shape[0] == theta.shape[0]
    
    if np.isnan(ctt_scores).any():
        warnings.warn("ctt_scores contains nan", UserWarning)
    mask = ~np.isnan(ctt_scores)
    theta_masked, ctt_scores_masked = theta[mask], ctt_scores[mask]
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
        indices = np.random.choice(len(theta_masked), int(0.8 * len(theta_masked)), replace=False)
        sample_corr = np.corrcoef(theta_masked[indices], ctt_scores_masked[indices])[0, 1]
        sample_corrs.append(sample_corr)
    sample_std = np.std(sample_corrs)
    
    plt.figure(figsize=(10, 10))
    plt.scatter(theta_masked, ctt_scores_masked)
    plt.xlabel(r'$\theta$ from calibration', fontsize=45)
    plt.ylabel(r'CTT score', fontsize=45)
    plt.title(f'Correlation: {corr:.2f} $\\pm$ {3 * sample_std:.2f}', fontsize=45)
    plt.tick_params(axis='both', labelsize=35)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    
def error_bar_plot(datasets, means, stds, plot_path):
    plt.figure(figsize=(20, 6))
    plt.errorbar(datasets, means, yerr=stds, elinewidth=1, fmt="o", ms=5, capsize=8, capthick=1)
    plt.xticks(rotation=30, ha='right', fontsize=20)
    plt.tick_params(axis='both', labelsize=20)
    plt.ylim(0, 1)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')

def amorz_corr_nonamorz(
    z_amor: np.array,
    z_nonamor: np.array,
):
    assert z_amor.shape == z_nonamor.shape, f'{z_amor.shape} != {z_nonamor.shape}'
    z_corr = np.corrcoef(z_amor, z_nonamor)[0, 1]
    return z_corr


    
    
    
    
    
def z_corr_plot(
    x,
    y,
    plot_path,
):
    corr = np.corrcoef(x, y)[0, 1]
    mse = np.mean((x - y) ** 2)
    plt.figure(figsize=(10, 10))
    plt.scatter(x, y)
    plt.xlabel(r'$z$ from amortized IRT calibration', fontsize=45)
    plt.ylabel(r'$z$ from non-amortized IRT calibration', fontsize=45)
    plt.title(f'Correlation: {corr:.2f}, MSE: {mse:.2f}', fontsize=45)
    plt.tick_params(axis='both', labelsize=35)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    
def theta_corr_plot(
    x,
    y,
    plot_path,
):
    corr = np.corrcoef(x, y)[0, 1]
    plt.figure(figsize=(10, 10))
    plt.scatter(x, y)
    plt.xlabel(r'$\theta$ from calibration', fontsize=45)
    plt.ylabel(r'CTT score', fontsize=45)
    plt.title(f'Correlation: {corr:.2f}', fontsize=45)
    plt.tick_params(axis='both', labelsize=35)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    
def plot_scatter_with_histograms(z3_py, z3_r, save_path, x_label=r'Our $z_3$', y_label=r'mirt $z_3$'):
    plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4], wspace=0.05, hspace=0.05)

    # Scatter plot between z3_py and z3_r
    ax_main = plt.subplot(gs[1, 0])
    ax_main.scatter(z3_py, z3_r)
    ax_main.set_xlabel(x_label)
    ax_main.set_ylabel(y_label)

    # Calculate correlation and add title at the bottom
    corr_np = np.corrcoef(z3_py, z3_r)[0, 1]
    plt.figtext(0.5, 0.02, f'Correlation: {corr_np:.2f}', ha='center')

    # Histogram for z3_py (top)
    ax_xhist = plt.subplot(gs[0, 0], sharex=ax_main)
    ax_xhist.hist(z3_py, bins=30, color='gray', alpha=0.7)
    ax_xhist.axis('off')

    # Histogram for z3_r (right)
    ax_yhist = plt.subplot(gs[1, 1], sharey=ax_main)
    ax_yhist.hist(z3_r, bins=30, color='gray', alpha=0.7, orientation='horizontal')
    ax_yhist.axis('off')

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


DESCRIPTION_MAP = {
    'synthetic_efficiency': '### DATASET: Synthetic efficiency, ### PUBLISH TIME: unknown, ### CONTENT: to better understand inference runtime performance of various models',
    'wikifact': '### DATASET: WikiFact, ### PUBLISH TIME: 2019, ### CONTENT: knowledge base completion, entity-relation-entity triples in natural language form, to more extensively test factual knowledge',
    'entity_data_imputation': '### DATASET: Data imputation, ### PUBLISH TIME: 2021, ### CONTENT: tests the ability to impute missing entities in a data table',
    'commonsense': '### DATASET: HellaSwag, ### PUBLISH TIME: 2019, ### CONTENT: commonsense reasoning in question answering',
    'quac': '### DATASET: QuAC (Question Answering in Context), ### PUBLISH TIME: 2018, ### CONTENT: question answering in the context of dialogues',
    'imdb': '### DATASET: IMDB, ### PUBLISH TIME: 2011, ### CONTENT: sentiment analysis in movie review',
    'bbq': '### DATASET: BBQ (Bias Benchmark for Question Answering), ### PUBLISH TIME: 2022, ### CONTENT: for measuring social bias in question answering in ambiguous and unambigous context',
    'math': '### DATASET: MATH, ### PUBLISH TIME: 2021, ### CONTENT: for measuring mathematical problem solving on competition math problems with or without with chain-of-thought style reasoning',
    'twitter_aae': '### DATASET: TwitterAAE, ### PUBLISH TIME: 2016, ### CONTENT: for measuring language model performance in tweets as a function of speaker dialect, on African-American-aligned Tweets, on White-aligned Tweets',
    'truthful_qa': '### DATASET: TruthfulQA, ### PUBLISH TIME: 2022, ### CONTENT: for measuring model truthfulness and commonsense knowledge in question answering',
    # 'msmarco': '### DATASET: MSMARCO, ### PUBLISH TIME: 2016, ### CONTENT: for passage retrieval in information retrieval',
    'legal_support': '### DATASET: LegalSupport, ### PUBLISH TIME: unknown, ### CONTENT: measure fine-grained legal reasoning through reverse entailment.',
    'boolq': '### DATASET: boolq, ### PUBLISH TIME: 2019, ### CONTENT: binary (yes/no) question answering, passages from Wikipedia, questions from search queries',
    'narrative_qa': '### DATASET: NarrativeQA, ### PUBLISH TIME: 2017, ### CONTENT: for reading comprehension over narratives, passages are books and movie scripts',
    'real_toxicity_prompts': '### DATASET: RealToxicityPrompts, ### PUBLISH TIME: 2020, ### CONTENT: for measuring toxicity in prompted model generations',
    'bold': '### DATASET: BOLD (Bias in Open-Ended Language Generation Dataset), ### PUBLISH TIME: 2021, ### CONTENT: for measuring biases and toxicity in open-ended language generation',
    'gsm': '### DATASET: GSM8K (Grade school math word problems), ### PUBLISH TIME: 2021, ### CONTENT: for testing mathematical reasoning on grade-school math problems',
    'babi_qa': '### DATASET: bAbI, ### PUBLISH TIME: 2015, ### CONTENT: for measuring understanding and reasoning',
    'summarization_xsum': '### DATASET: XSUM, ### PUBLISH TIME: 2018, ### CONTENT: for text summarization of BBC news articles',
    'synthetic_reasoning_natural': '### DATASET: Synthetic reasoning (natural language), ### PUBLISH TIME: 2021, ### CONTENT: Synthetic reasoning tasks defined using simple natural language based on LIME',
    'dyck_language_np=3': '### DATASET: Dyck, ### PUBLISH TIME: 2019, ### CONTENT: Scenario testing hierarchical reasoning through the Dyck formal languages',
    'civil_comments': '### DATASET: CivilComments, ### PUBLISH TIME: 2019, ### CONTENT: for toxicity detection',
    'lsat_qa': '### DATASET: LSAT, ### PUBLISH TIME: 2021, ### CONTENT: for measuring analytical reasoning on the Law School Admission Test',
    'raft': '### DATASET: RAFT (Real-world Annotated Few-Shot), ### PUBLISH TIME: 2021, ### CONTENT: meta-benchmark of 11 real-world text classification tasks',
    'code': '### DATASET: Code, ### PUBLISH TIME: 2021, ### CONTENT: for measuring competence on code challenges, for measuring functional correctness for synthesizing programs from docstrings',
    'entity_matching': '### DATASET: Entity matching, ### PUBLISH TIME: 2016, ### CONTENT: tests the ability to determine if two entities match',
    'synthetic_reasoning': '### DATASET: Synthetic reasoning, ### PUBLISH TIME: 2021, ### CONTENT: defined using abstract symbols based on LIME and simple natural language based on LIME',
    'mmlu': '### DATASET: MMLU (Massive Multitask Language Understanding), ### PUBLISH TIME: 2021, ### CONTENT: for knowledge-intensive question answering across 57 domains',
    'airbench': '### DATASET: AirBench, ### PUBLISH TIME: 2024, ### CONTENT: AI safety benchmark that aligns with emerging government regulations and company policies',
}
