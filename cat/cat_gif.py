import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from tueplots import bundles
plt.rcParams.update(bundles.icml2022())
plt.style.use('seaborn-v0_8-paper')
from PIL import Image

if __name__ == "__main__":
    dataset = 'twitter_aae'
    
    plot_dir = '../plot/cat_gif'
    os.makedirs(plot_dir, exist_ok=True)
    
    df_path = f'../data/cat/{dataset}/cat_gif.csv'
    data = pd.read_csv(df_path)
    cat_theta_estimate = data[data['variant'] == 'CAT']['thetaEstimate'].tolist()
    random_theta_estimate = data[data['variant'] == 'Random']['thetaEstimate'].tolist()
    
    num_iter = len(cat_theta_estimate)
    true_theta = 0.5
    img_paths = []
    for i in tqdm(range(num_iter)):
        plt.figure(figsize=(8, 6))
        plt.axvline(true_theta, color='black', linestyle='--', label=r'True $\theta$', linewidth=2)
        
        plt.scatter(cat_theta_estimate[:i+1], [0] * (i+1), label=r'CAT', color='blue', marker='o', s=100)
        plt.scatter(random_theta_estimate[:i+1], [0] * (i+1), label=r'Random', color='red', marker='x', s=100)
        
        # if i > 1:
        #     cat_mean, cat_std = norm.fit(cat_theta_estimate[:i+1])
        #     random_mean, random_std = norm.fit(random_theta_estimate[:i+1])
        #     x = np.linspace(-5, 5, 100)
        #     plt.plot(x, norm.pdf(x, cat_mean, cat_std), color='blue', lw=2)
        #     plt.plot(x, norm.pdf(x, random_mean, random_std), color='red', lw=2)
        plt.axvline(cat_theta_estimate[i], color='blue', linestyle='-', linewidth=2)
        plt.axvline(random_theta_estimate[i], color='red', linestyle='-', linewidth=2)
        
        plt.xlim(-5, 5)
        # plt.ylim(0, 1)
        plt.gca().get_yaxis().set_visible(False)
        plt.xlabel(r'$\theta$', fontsize=25)
        plt.title(f'Iteration {i+1}/{num_iter}', fontsize=25)
        plt.legend(fontsize=15)
        plt.tick_params(axis='both', labelsize=25)
        
        img_path = f'{plot_dir}/frame_{i}.png'
        plt.savefig(img_path, dpi=300, bbox_inches='tight')
        img_paths.append(img_path)
        plt.close()
    
    frames = [Image.open(img_path) for img_path in img_paths]
    frames[0].save(f"{plot_dir}/cat.gif", format='GIF', append_images=frames[1:], save_all=True, duration=200, loop=0)