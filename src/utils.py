import torch
import numpy as np
from scipy.stats import wasserstein_distance, wasserstein_distance_nd
import jax.numpy as jnp
import random
import gc
import sys
from scipy.stats import ttest_ind
import os

# overleaf/R_library: z1/g, z2/a1, z3/d
def item_response_fn_3PL(z1, z2, z3, theta, datatype="torch"):
    if datatype == "torch":
        return z1 + (1 - z1) / (1 + torch.exp(-(z2 * theta + z3)))
    elif datatype == "numpy":
        return z1 + (1 - z1) / (1 + np.exp(-(z2 * theta + z3)))
    elif datatype == "jnp":
        return z1 + (1 - z1) / (1 + jnp.exp(-(z2 * theta + z3)))

def item_response_fn_2PL(z2, z3, theta):
    return 1 / (1 + torch.exp(-(z2 * theta + z3)))

def item_response_fn_1PL(z3, theta, datatype="torch"):
    if datatype == "torch":
        return 1 / (1 + torch.exp(-(theta + z3)))
    elif datatype == "numpy":
        return 1 / (1 + np.exp(-(theta + z3)))
    elif datatype == "jnp":
        return 1 / (1 + jnp.exp(-(theta + z3)))
    else:
        raise ValueError("datatype should be 'torch' or 'numpy' or 'jnp'")
    
def item_response_fn_1PL_cheat(z3, contamination, theta_true, theta_cheat, datatype="torch"):
    # z3, contamination: vector/scalar; theta_true, theta_cheat: scalar
    bool_cheat = (theta_true < theta_cheat)
    
    if datatype == "torch":
        return item_response_fn_1PL(z3, theta_true)**(1-bool_cheat) \
            * ((1-contamination) * item_response_fn_1PL(z3, theta_true) \
            + contamination * item_response_fn_1PL(z3, theta_cheat))**bool_cheat
    elif datatype == "jnp":
        return item_response_fn_1PL(z3, theta_true, datatype="jnp")**(1-bool_cheat) \
            * ((1-contamination) * item_response_fn_1PL(z3, theta_true, datatype="jnp") \
            + contamination * item_response_fn_1PL(z3, theta_cheat, datatype="jnp"))**bool_cheat        

def calculate_1d_wasserstein_distance(vector1, vector2):
    return wasserstein_distance(vector1, vector2)

def calculate_3d_wasserstein_distance(matrix1, matrix2):
    return wasserstein_distance_nd(matrix1, matrix2)

def clear_caches():
    modules = list(sys.modules.items())  # Create a list of items to avoid runtime errors
    for module_name, module in modules:
        if module_name.startswith("jax"):
            if module_name not in ["jax.interpreters.partial_eval"]:
                for obj_name in dir(module):
                    obj = getattr(module, obj_name)
                    if hasattr(obj, "cache_clear"):
                        try:
                            obj.cache_clear()
                        except:
                            pass
    gc.collect()
    print("cache cleared")
    
def set_seed(seed):
    random.seed(seed)
    # torch.backends.cudnn.deterministic=True
    # torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed_all(seed)
    
def save_state(filepath, **kwargs):
    torch.save(kwargs, filepath)
    print(f"State saved to {filepath}")

def load_state(filepath):
    if os.path.exists(filepath):
        state = torch.load(filepath)
        print(f"State loaded from {filepath}")
        return state
    else:
        print(f"No previous state found at {filepath}")
        return None

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

if __name__ == "__main__":
    print(calculate_1d_wasserstein_distance([0, 1, 3], [5, 6, 8]))
    print(calculate_1d_wasserstein_distance([1, 1, 3], [5, 6, 8]))
    
    mean_1 = 1
    mean_2 = 3
    size = 10000
    diff = mean_2 - mean_1
    
    samples_1 = np.random.normal(loc=mean_1, scale=1, size=size)
    samples_2 = np.random.normal(loc=mean_2, scale=1, size=size)
    
    wd = calculate_1d_wasserstein_distance(samples_1, samples_2)
    print(f'Wasserstein Distance: {wd}')
    print(f'error rate: {(wd - diff)/diff}')
    
    mean_1 = 0.7
    mean_2 = 0.3
    samples_1 = np.random.binomial(n=1, p=mean_1, size=size)
    samples_2 = np.random.binomial(n=1, p=mean_2, size=size)
    wd = calculate_1d_wasserstein_distance(samples_1, samples_2)
    diff = samples_1.mean() - samples_2.mean()
    print(wd)
    print(diff)
    