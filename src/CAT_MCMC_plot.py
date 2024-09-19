import matplotlib.pyplot as plt
import jax.numpy as jnp
from utils import load_state
import os

if __name__ == '__main__':
    state_dir = "../data/synthetic/CAT"
    question_num = 10000
    theta_star = 0
    strategy_list = ['random', 'fisher', 'owen', 'efisher']

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    axs = axs.flatten()
    
    for i, strategy in enumerate(strategy_list):
        state_path = os.path.join(state_dir, f"{strategy}_{question_num}_{theta_star}.pt")
        state = load_state(state_path)
        
        if state:
            z3 = state['z3']
            theta_means = state['theta_means']
            theta_stds = state['theta_stds']
            
            total_question_nums = range(question_num)
            subset_question_num = len(theta_means)
            subset_question_nums = range(subset_question_num)

            axs[i].plot(
                subset_question_nums, 
                [theta_star] * subset_question_num, 
                label='True Theta', 
                color='black', 
                linestyle='--'
                )
            axs[i].plot(subset_question_nums, theta_means, label=f'{strategy}')
            theta_means = jnp.array(theta_means)
            theta_stds = jnp.array(theta_stds)
            axs[i].fill_between(subset_question_nums, 
                                theta_means - 3 * theta_stds, 
                                theta_means + 3 * theta_stds, 
                                alpha=0.2)
            axs[i].set_title(f'{strategy}')
            axs[i].set_xlabel('Number of Questions')
            axs[i].set_ylabel('Theta')
            axs[i].set_ylim([-4, 4])
            axs[i].grid(True)
            axs[i].legend()

    plt.tight_layout()
    plt.savefig('../plot/synthetic/random_adaptive_test_subplot.png')
