import argparse
import os
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import jax.numpy as jnp
import jax.random as random
from tqdm import tqdm
from utils import item_response_fn_1PL_jnp, set_seed
from datasets import load_dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tueplots import bundles
plt.rcParams.update(bundles.icml2022())
plt.style.use('seaborn-v0_8-paper')
 
def ucb(
    means: np.array, # (1.2, 1.5)
    stds: np.array, # (10, 20)
    c: float=1.0
):
    ucb_values = means + c * stds
    return np.argmax(ucb_values)

def thompson_sampling(list_of_samples):
    selected_elements = [np.random.choice(arr) for arr in list_of_samples]
    return np.argmax(selected_elements)

def model(asked_zs, asked_ys):
    theta_hat = numpyro.sample("theta_hat", dist.Normal(0.0, 1.0)) # prior
    probs = item_response_fn_1PL_jnp(asked_zs, theta_hat)
    numpyro.sample("obs", dist.Bernoulli(probs), obs=asked_ys)

def fit_theta_mcmc(asked_zs, asked_ys, num_samples=2000, num_warmup=1000):
    rng_key = random.PRNGKey(0)
    rng_key, rng_key_ = random.split(rng_key)
    
    nuts_kernel = NUTS(model)
    mcmc = MCMC(nuts_kernel, num_samples=num_samples, num_warmup=num_warmup)
    mcmc.run(
        rng_key_,
        asked_zs=asked_zs,
        asked_ys=asked_ys,
    )
    mcmc.print_summary()
    
    theta_samples = mcmc.get_samples()["theta_hat"]
    mean_theta = jnp.mean(theta_samples)
    std_theta = jnp.std(theta_samples)
    return mean_theta, std_theta, theta_samples

def plot_theta_estimates(theta_estimates, true_theta, dataset_name, criteria):
    plt.figure()
    plt.plot(theta_estimates[dataset_name]['llama2_7b']['theta_mean'], label="llama2_7b theta", color='red')
    plt.plot(theta_estimates[dataset_name]['llama2_13b']['theta_mean'], label="llama2_13b theta", color='blue')
    
    plt.axhline(y=true_theta['llama2_7b'], linestyle='--', color='red', label='llama2_7b true theta')
    plt.axhline(y=true_theta['llama2_13b'], linestyle='--', color='blue', label='llama2_13b true theta')

    plt.ylabel('Theta Estimates', fontsize=20)
    plt.title(f'Dataset: {dataset_name}', fontsize=20)
    plt.legend(fontsize=16)
    plt.tick_params(axis='both', labelsize=16)
    plt.savefig(f'../plot/bandit_demo_{criteria}/theta_estimates_{dataset_name}.png')

def plot_cumulative_frequency(model_counts, dataset_name, criteria):
    plt.figure()
    plt.plot(model_counts[dataset_name]['llama2_7b'], label="llama2_7b freq", color='red')
    plt.plot(model_counts[dataset_name]['llama2_13b'], label="llama2_13b freq", color='blue')

    plt.ylabel('Cumulative Frequency', fontsize=20)
    plt.title(f'Dataset: {dataset_name}', fontsize=20)
    plt.legend(fontsize=16)
    plt.tick_params(axis='both', labelsize=16)
    plt.savefig(f'../plot/bandit_demo_{criteria}/cumulative_frequency_{dataset_name}.png')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--criteria', type=str, required=True, choices=['ucb', 'thompson'])
    args = parser.parse_args()
    
    set_seed(42)
    plot_dir = f'../plot/bandit_demo_{args.criteria}'
    os.makedirs(plot_dir, exist_ok=True)
    
    # z_hf = load_dataset("stair-lab/reeval_individual-embed")
    # airbench_zs = z_hf['airbench']['z']
    # mmlu_zs = z_hf['mmlu']['z']
    airbench_zs = pd.read_csv("../data/nonamor_calibration/airbench/nonamor_z.csv")['z'].values
    mmlu_zs = pd.read_csv("../data/nonamor_calibration/mmlu/nonamor_z.csv")['z'].values

    airbench_thetas = pd.read_csv("../data/nonamor_calibration/airbench/nonamor_theta.csv")['theta'].values
    airbench_llama2_7b_theta_true = airbench_thetas[42]
    airbench_llama2_13b_theta_true = airbench_thetas[44]
    print(f"Airbench llama2_7b theta: {airbench_llama2_7b_theta_true}, llama2_13b theta: {airbench_llama2_13b_theta_true}")
    airbench_y_full = pd.read_csv(f'../data/pre_calibration/airbench/matrix.csv', index_col=0).values
    airbench_y = {
        'llama2_7b': airbench_y_full[42],
        'llama2_13b': airbench_y_full[44],
    }
    
    mmlu_thetas = pd.read_csv("../data/nonamor_calibration/mmlu/nonamor_theta.csv")['theta'].values
    mmlu_llama2_7b_theta_true = mmlu_thetas[28]
    mmlu_llama2_13b_theta_true = mmlu_thetas[30]
    print(f"MMLU llama2_7b theta: {mmlu_llama2_7b_theta_true}, llama2_13b theta: {mmlu_llama2_13b_theta_true}")
    mmlu_y_full = pd.read_csv(f'../data/pre_calibration/mmlu/matrix.csv', index_col=0).values
    mmlu_y = {
        'llama2_7b': mmlu_y_full[28],
        'llama2_13b': mmlu_y_full[30],
    }
    
    rounds = 300
    datasets = ['airbench', 'mmlu']
    models = ['llama2_7b', 'llama2_13b']
    
    remaining_questions = {
        'airbench': list(range(len(airbench_zs))),
        'mmlu': list(range(len(mmlu_zs)))
    }
    asked_zs = {
        'airbench': {'llama2_7b': [], 'llama2_13b': []}, 
        'mmlu': {'llama2_7b': [], 'llama2_13b': []}
    }
    asked_ys = {
        'airbench': {'llama2_7b': [], 'llama2_13b': []}, 
        'mmlu': {'llama2_7b': [], 'llama2_13b': []}
    }
    A   
    theta_estimates = {
        'airbench': {
            'llama2_7b': {
                'theta_mean': [],
                'theta_std': [],
                'theta_samples': [],
            },
            'llama2_13b': {
                'theta_mean': [],
                'theta_std': [],
                'theta_samples': [],
            },
        }, 
        'mmlu': {
            'llama2_7b': {
                'theta_mean': [],
                'theta_std': [],
                'theta_samples': [],
            },
            'llama2_13b': {
                'theta_mean': [],
                'theta_std': [],
                'theta_samples': [],
            },
        },
    }
    model_counts = {
        'airbench': {
            'llama2_7b': [0,], 
            'llama2_13b': [0,]
        }, 
        'mmlu': {
            'llama2_7b': [0,], 
            'llama2_13b': [0,]
        }
    }
    
    for i in tqdm(range(rounds)):
        selected_dataset = np.random.choice(datasets)
        unselected_dataset = datasets[0] if selected_dataset == datasets[1] else datasets[1]
        selected_idx = np.random.choice(remaining_questions[selected_dataset])
        remaining_questions[selected_dataset].remove(selected_idx)

        if not theta_estimates[selected_dataset]['llama2_7b']['theta_mean'] or not theta_estimates[selected_dataset]['llama2_13b']['theta_mean']:
            selected_model = np.random.choice(models)
            unselected_model = models[0] if selected_model == models[1] else models[1]
        else:
            if args.criteria == 'ucb':
                selected_model = models[
                    ucb(
                        means=np.array([theta_estimates[selected_dataset][model]['theta_mean'][-1] for model in models]),
                        stds=np.array([theta_estimates[selected_dataset][model]['theta_std'][-1] for model in models])
                    )
                ]
                unselected_model = models[0] if selected_model == models[1] else models[1]
                
            elif args.criteria == 'thompson':
                selected_model = models[
                    thompson_sampling([np.array(theta_estimates[selected_dataset][model]['theta_samples'][-1]) for model in models])
                ]
                unselected_model = models[0] if selected_model == models[1] else models[1]
        
        if selected_dataset == 'airbench':
            z, y = airbench_zs[selected_idx], airbench_y[selected_model][selected_idx]
        else:
            z, y = mmlu_zs[selected_idx], mmlu_y[selected_model][selected_idx]
            
        asked_zs[selected_dataset][selected_model].append(z)
        asked_ys[selected_dataset][selected_model].append(y)
        
        theta_mean, theta_std, theta_samples = fit_theta_mcmc(
            jnp.array(asked_zs[selected_dataset][selected_model]),
            jnp.array(asked_ys[selected_dataset][selected_model])
        )

        theta_estimates[selected_dataset][selected_model]['theta_mean'].append(theta_mean)
        theta_estimates[selected_dataset][selected_model]['theta_std'].append(theta_std)
        theta_estimates[selected_dataset][selected_model]['theta_samples'].append(theta_samples)
        model_counts[selected_dataset][selected_model].append(model_counts[selected_dataset][selected_model][-1] + 1)
        model_counts[unselected_dataset][selected_model].append(model_counts[unselected_dataset][selected_model][-1])
        model_counts[selected_dataset][unselected_model].append(model_counts[selected_dataset][unselected_model][-1])
        model_counts[unselected_dataset][unselected_model].append(model_counts[unselected_dataset][unselected_model][-1])

    true_theta_airbench = {
        'llama2_7b': airbench_llama2_7b_theta_true,
        'llama2_13b': airbench_llama2_13b_theta_true
    }
    true_theta_mmlu = {
        'llama2_7b': mmlu_llama2_7b_theta_true,
        'llama2_13b': mmlu_llama2_13b_theta_true
    }

    plot_theta_estimates(theta_estimates, true_theta_airbench, 'airbench', args.criteria)
    plot_theta_estimates(theta_estimates, true_theta_mmlu, 'mmlu', args.criteria)
    plot_cumulative_frequency(model_counts, 'airbench', args.criteria)
    plot_cumulative_frequency(model_counts, 'mmlu', args.criteria)