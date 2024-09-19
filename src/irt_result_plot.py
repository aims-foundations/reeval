import os
import pandas as pd
import matplotlib.pyplot as plt
import subprocess
import torch
import numpy as np
from utils import item_response_fn_1PL, calculate_3d_wasserstein_distance, calculate_1d_wasserstein_distance
import argparse
from sklearn.metrics import mean_squared_error
# from tueplots import bundles
# bundles.icml2022()
# bundles.icml2022(family="sans-serif", usetex=False, column="full", nrows=2)
# plt.rcParams.update(bundles.icml2022())

def plot_hist(serial, data, color, para, perturb):
    plt.subplot(3, 3, serial)
    plt.hist(data, bins=30, color=color, density=True)
    plt.xlabel('value')
    plt.ylabel('frequency')
    plt.title(f'{para} of {perturb} coef')
    plt.grid(True)
            
if __name__ == "__main__":
    plt.rcParams.update({'font.size': 20})
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, help="synthetic or real_normal or real_appendix1")
    args = parser.parse_args()
    
    if args.exp == "synthetic":
        true_Z_path = f'../data/synthetic/response_matrix/true_Z.csv'
        true_theta_path = f'../data/synthetic/response_matrix/true_theta.csv'
        Z_dir = f'../data/synthetic/irt_result/Z'
        theta_dir = f'../data/synthetic/irt_result/theta'
        output_dir = '../plot/synthetic'

    elif args.exp == "real_normal":
        Z_dir = f'../data/real/irt_result/normal/Z'
        theta_dir = f'../data/real/irt_result/normal/theta'
        output_dir = '../plot/real'
        response_matrix_dir = '../data/real/response_matrix/normal'
    
    elif args.exp == "real_appendix1":
        Z_dir = f'../data/real/irt_result/appendix1/Z'
        theta_dir = f'../data/real/irt_result/appendix1/theta'
        output_dir = '../plot/real'
        response_matrix_dir = '../data/real/response_matrix/appendix1'
        
    os.makedirs(Z_dir, exist_ok=True)
    os.makedirs(theta_dir, exist_ok=True)
    
    model_list = ["1PL", "2PL", "3PL"]



    # run mirt.R
    print("running mirt.R")
    subprocess.run(f"conda run -n R Rscript fit_irt.R {args.exp}", shell=True, check=True)
    
    # clean up item parameters inferred from IRT
    print("cleaning up item parameters inferred from IRT")
    for filename in os.listdir(Z_dir):
        if filename.endswith('_Z.csv'):
            file_path = os.path.join(Z_dir, filename)
            df = pd.read_csv(file_path)

            # Delete columns
            df = df.iloc[:, 1:-2]

            # Define new columns and prepare data
            new_columns = ['z2', 'z3', 'z1', 'u']
            data = {col: [] for col in new_columns}
            for i in range(0, len(df.columns), 4):
                for col, new_col in zip(df.columns[i:i+4], new_columns):
                    data[new_col].append(df[col].values[0])

            # Create a new DataFrame with the cleaned data
            new_df = pd.DataFrame(data)
            new_df = new_df[['z1', 'z2', 'z3']]

            # Save the cleaned data to a new CSV file
            clean_file_path = os.path.join(Z_dir, filename.replace('.csv', '_clean.csv'))
            new_df.to_csv(clean_file_path, index=False)



    if args.exp == 'real_appendix1':
        all_z_df = pd.read_csv(f'{Z_dir}/all_1PL_Z_clean.csv')
        index_search_df = pd.read_csv(f'{response_matrix_dir}/index_search.csv')

        z3_values = all_z_df['z3']
        index_search_df = index_search_df[index_search_df['is_deleted'] != 1]

        assert len(index_search_df) == len(all_z_df)

        index_search_df['z3'] = z3_values.values

        z3_dict = {
            'base': index_search_df[index_search_df['perturb'] == 'base']['z3'].tolist(),
            'perturb1': index_search_df[index_search_df['perturb'] == 'perturb1']['z3'].tolist(),
            'perturb2': index_search_df[index_search_df['perturb'] == 'perturb2']['z3'].tolist()
        }

        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    axes[0].hist(z3_dict['base'], bins=20, alpha=0.7, density=True)
    axes[0].set_title('Distribution of base')
    axes[0].set_xlabel('z3 Values')
    axes[0].set_ylabel('Density')

    axes[1].hist(z3_dict['perturb1'], bins=20, alpha=0.7, density=True)
    axes[1].set_title('Distribution of perturb1')
    axes[1].set_xlabel('z3 Values')

    axes[2].hist(z3_dict['perturb2'], bins=20, alpha=0.7, density=True)
    axes[2].set_title('Distribution of perturb2')
    axes[2].set_xlabel('z3 Values')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/appendix1_three_z_set_separate.png')
        
        
        
    if args.exp == 'synthetic':
        # synthetic only, mse of Z and theta
        print("synthetic only, mse of Z and theta")
        mse_output_path = f'{output_dir}/mse.txt'
        with open(mse_output_path, 'w') as f:
            f.write('MSE of Z\n')
            for filename in os.listdir(Z_dir):
                if filename.endswith('_clean.csv'):
                    synthetic_Z_path = os.path.join(Z_dir, filename)
                    synthetic_Z = pd.read_csv(synthetic_Z_path).values
                    true_Z = pd.read_csv(true_Z_path).values
                    mse_Z = mean_squared_error(true_Z, synthetic_Z)
                    model = filename.split('_Z_clean.csv')[0].split('synthetic_')[1]
                    f.write(f'{model}: {mse_Z}\n')
                        
            f.write('\n\n\nMSE of theta\n')
            for filename in os.listdir(theta_dir):
                if filename.endswith('.csv'):
                    synthetic_theta_path = os.path.join(theta_dir, filename)
                    synthetic_theta = pd.read_csv(synthetic_theta_path).values[:, -1]
                    true_theta = pd.read_csv(true_theta_path).values[:, -1]
                    mse_theta = mean_squared_error(true_theta, synthetic_theta)
                    model = filename.split('_theta.csv')[0].split('synthetic_')[1]
                    f.write(f'{model}: {mse_theta}\n')
    
    
    
    if args.exp == 'real_normal':
        # real_normal only, 3PL only, 3D plot of [z1, z2, z3]
        print("real_normal only, 3PL only, 3D plot of [z1, z2, z3]")
        base_coef = pd.read_csv(f'{Z_dir}/base_3PL_Z_clean.csv')
        perturb1_coef = pd.read_csv(f'{Z_dir}/perturb1_3PL_Z_clean.csv')
        perturb2_coef = pd.read_csv(f'{Z_dir}/perturb2_3PL_Z_clean.csv')

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(base_coef.iloc[:, 0], base_coef.iloc[:, 1], base_coef.iloc[:, 2], c='r', label='Base Z')
        ax.scatter(perturb1_coef.iloc[:, 0], perturb1_coef.iloc[:, 1], perturb1_coef.iloc[:, 2], c='g', label='Perturb1 Z')
        ax.scatter(perturb2_coef.iloc[:, 0], perturb2_coef.iloc[:, 1], perturb2_coef.iloc[:, 2], c='b', label='Perturb2 Z')

        ax.set_xlabel(r'$z_1$')
        ax.xaxis.labelpad = 20
        ax.set_ylabel(r'$z_2$')
        ax.yaxis.labelpad = 20
        ax.set_zlabel(r'$z_3$')
        ax.zaxis.labelpad = 20

        # ax.set_xlim(-3, 3)
        # ax.set_ylim(-3, 3)
        # ax.set_zlim(-3, 3)

        ax.legend()
        plt.savefig(f'{output_dir}/3d3pl.png')


        # real_normal only, plot Z distribution (density histogram)
        print("real_normal only, plot Z distribution (density histogram)")
        for i, model in enumerate(model_list):
            base_coef = pd.read_csv(f'{Z_dir}/base_{model}_Z_clean.csv')
            perturb1_coef = pd.read_csv(f'{Z_dir}/perturb1_{model}_Z_clean.csv')
            perturb2_coef = pd.read_csv(f'{Z_dir}/perturb2_{model}_Z_clean.csv')

            base_value = [base_coef.iloc[:, i] for i in range(3)]
            perturb1_value = [perturb1_coef.iloc[:, i] for i in range(3)]
            perturb2_value = [perturb2_coef.iloc[:, i] for i in range(3)]

            plt.figure(figsize=(25, 25))

            plot_hist(1, base_value[0], 'r', 'z1', 'base')
            plot_hist(2, base_value[1], 'r', 'z2', 'base')
            plot_hist(3, base_value[2], 'r', 'z3', 'base')   

            plot_hist(4, perturb1_value[0], 'g', 'z1', 'perturb1')
            plot_hist(5, perturb1_value[1], 'g', 'z2', 'perturb1')
            plot_hist(6, perturb1_value[2], 'g', 'z3', 'perturb1')   

            plot_hist(7, perturb2_value[0], 'b', 'z1', 'perturb2')
            plot_hist(8, perturb2_value[1], 'b', 'z2', 'perturb2')
            plot_hist(9, perturb2_value[2], 'b', 'z3', 'perturb2')   

            plt.savefig(f'{output_dir}/iparams{i+1}pl.png')
    

        # real_normal only, 1D Wasserstein Distance of Z
        print("real_normal only, 1D Wasserstein Distance of Z")
        with open(f"{output_dir}/1D_Wasserstein_Distance.txt", "w", encoding="utf-8") as f:
            for model in model_list:
                f.write(f"{model}\n")
                base_path = f"{Z_dir}/base_{model}_Z_clean.csv"
                perturb1_path = f"{Z_dir}/perturb1_{model}_Z_clean.csv"
                perturb2_path = f"{Z_dir}/perturb2_{model}_Z_clean.csv"

                para_list =  ['z1', 'z2', 'z3']
                for para in para_list:
                    base_coef = pd.read_csv(base_path, usecols=[para]).values.flatten()
                    perturb1_coef = pd.read_csv(perturb1_path, usecols=[para]).values.flatten()
                    perturb2_coef = pd.read_csv(perturb2_path, usecols=[para]).values.flatten()

                    distance_1_2 = calculate_1d_wasserstein_distance(base_coef, perturb1_coef)
                    distance_1_3 = calculate_1d_wasserstein_distance(base_coef, perturb2_coef)
                    distance_2_3 = calculate_1d_wasserstein_distance(perturb1_coef, perturb2_coef)
                    
                    f.write(f"Parameter {para}\n")
                    f.write(f"Distance between base and perturb1: {distance_1_2}\n")
                    f.write(f"Distance between base and perturb2: {distance_1_3}\n")
                    f.write(f"Distance between perturb1 and perturb2: {distance_2_3}\n")
                    f.write("\n")
        


        # real_normal only, 3D Wasserstein Distance of Z
        print("real_normal only, 3D Wasserstein Distance of Z")
        with open(f"{output_dir}/3D_Wasserstein_Distance.txt", "w", encoding="utf-8") as f:
            for model in model_list:
                f.write(f"{model}\n")

                base_path = f"{Z_dir}/base_{model}_Z_clean.csv"
                perturb1_path = f"{Z_dir}/perturb1_{model}_Z_clean.csv"
                perturb2_path = f"{Z_dir}/perturb2_{model}_Z_clean.csv"

                base_matrix = pd.read_csv(base_path).values
                perturb1_matrix = pd.read_csv(perturb1_path).values
                perturb2_matrix = pd.read_csv(perturb2_path).values
                
                # min_size = min(base_matrix.shape[0], perturb1_matrix.shape[0], perturb2_matrix.shape[0])
                # base_matrix = base_matrix[:min_size, :]
                # perturb1_matrix = perturb1_matrix[:min_size, :]
                # perturb2_matrix = perturb2_matrix[:min_size, :]

                distance_1_2 = calculate_3d_wasserstein_distance(base_matrix, perturb1_matrix)
                distance_1_3 = calculate_3d_wasserstein_distance(base_matrix, perturb2_matrix)
                distance_2_3 = calculate_3d_wasserstein_distance(perturb1_matrix, perturb2_matrix)

                f.write(f"Distance between base_matrix and perturb1_matrix: {distance_1_2}\n")
                f.write(f"Distance between base_matrix and perturb2_matrix: {distance_1_3}\n")
                f.write(f"Distance between perturb1_matrix and perturb2_matrix: {distance_2_3}\n")
                f.write("\n")
        


        # real_normal only, irt curve & scattar plot, only 1PL base
        print("real_normal only, irt curve & scattar plot, only 1PL base")
        base_coef_1PL = pd.read_csv(f'{Z_dir}/base_1PL_Z_clean.csv')
        base_value_1PL = base_coef_1PL.iloc[:, 2].values

        target_points = [-3, -1.5, 0, 1.5, 3]
        
        closest_points = {}
        closest_points_indices = {}
        for target in target_points:
            abs_diff = np.abs(base_value_1PL - target)
            min_index = abs_diff.argmin()
            closest_points_indices[target] = min_index
            closest_points[target] = base_value_1PL[min_index]
            
        theta_df = pd.read_csv(f'{theta_dir}/base_1PL_theta.csv')
        theta = torch.tensor(theta_df.iloc[:, 1].values, dtype=torch.float32)
        
        y_df = pd.read_csv(f"{response_matrix_dir}/base_matrix.csv", index_col=0)
        assert len(base_value_1PL) == len(y_df.columns)

        for i, target in enumerate(closest_points.keys()):
            z3 = torch.tensor(closest_points[target], dtype=torch.float32)
            indice = closest_points_indices[target]

            y = y_df.iloc[:, indice].values

            plt.figure(figsize=(10, 6))

            plt.scatter(theta.numpy(), y)
            
            theta_range = torch.linspace(min(theta).item(), max(theta).item(), 500)
            y_curve = item_response_fn_1PL(theta_range, z3)
            plt.plot(theta_range.numpy(), y_curve.numpy())
            
            plt.xlabel('Theta')
            plt.ylabel('Pr(y=1)')
            plt.title(f'Scatter Plot and 1PL Model Curve for d around {target}')
            plt.savefig(f'{output_dir}/empiricalvsestimated{i+1}.png')
