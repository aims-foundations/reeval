import argparse
import os
import subprocess
import wandb

if __name__ == "__main__":
    wandb.init(project="cat")
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()

    output_dir = f'../data/nonamor_calibration_R/{args.dataset}'
    os.makedirs(output_dir, exist_ok=True)
    
    subprocess.run(f"conda run -n cat Rscript nonamor_calibration.R", shell=True, check=True)
    
    for filename in os.listdir(output_dir):
        if filename.endswith('.csv'):
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

