import argparse
import numpy as np
from embed_text_package.embed_text import Embedder
import pandas as pd
from torch.utils.data import DataLoader
from datasets import Dataset, DatasetDict, load_dataset
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
import pickle
import os
import xgboost as xgb

def main(
    hf_repo,
    save_path,
    regression_model,
    embed_repo=None,
    model_name="meta-llama/Meta-Llama-3-8B",
    bs = 1024
):
    dataset = load_dataset(hf_repo, split="whole")
    
    try:
        emb_hf = load_dataset(embed_repo, split="whole")
        X = emb_hf['embeddings']
        
    except:
        cols_to_be_embded = ['question_text']
        
        embdr = Embedder()
        embdr.load(model_name)
        dataloader = DataLoader(dataset, batch_size=bs)
        emb = embdr.get_embeddings(
            dataloader, model_name, cols_to_be_embded
        )
        
        embed_df = pd.DataFrame({
            'question_text': dataset['question_text'],
            'embeddings': emb['question_text'],
        })
        embed_dataset = Dataset.from_pandas(embed_df)
        embed_dataset_dict = DatasetDict({
            "whole": embed_dataset,
        })
        embed_dataset_dict.push_to_hub(embed_repo)
        
        X = emb['question_text']
        
    y = dataset['z3']

    X = np.array(X)
    print(f'Shape of X: {X.shape}') 
    y = np.array(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if regression_model == "bayridge":
        model = BayesianRidge()
    elif regression_model == "xgboost":
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            max_depth=5,              # Limit depth of trees
            min_child_weight=5,       # Increase the minimum weight to reduce overfitting
            gamma=0.1,                # Minimum loss reduction required to make a split
            subsample=0.8,            # Subsample ratio of the training instances
            colsample_bytree=0.8,     # Subsample ratio of columns when constructing each tree
            # alpha=0.001,              # L1 regularization term
            reg_lambda=1.0,           # L2 regularization term
            n_estimators=1000,        # Number of boosting rounds
            learning_rate=0.01,       # Smaller learning rate to train more slowly
            random_state=42
        )
    elif regression_model == "mlp":
        model = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        alpha=0.01,  # L2 regularization term
        # early_stopping=False,  # Early stopping based on validation score
        # validation_fraction=0.1,  # Fraction of training data to use as validation
        max_iter=500,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    train_error = mean_squared_error(y_train, y_train_pred)
    print(f'Training Set Error (MSE): {train_error}')

    test_error = mean_squared_error(y_test, y_test_pred)
    print(f'Test Set Error (MSE): {test_error}')
    
    y_mean_pred = np.mean(y_train)
    mean_pred_train_error = mean_squared_error(y_train, np.full_like(y_train, y_mean_pred))
    mean_pred_test_error = mean_squared_error(y_test, np.full_like(y_test, y_mean_pred))

    print(f'Mean Prediction Training Set Error (MSE): {mean_pred_train_error}')
    print(f'Mean Prediction Test Set Error (MSE): {mean_pred_test_error}')
    
    with open(save_path, 'wb') as f:
        pickle.dump(model, f)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, required=True)
    parser.add_argument('--regression_model', type=str, default="bayridge")
    args = parser.parse_args()
    os.makedirs(f'../data/real/ppo/{args.exp}', exist_ok=True)
    
    main(
        hf_repo=f'stair-lab/{args.exp}-difficulty',
        save_path=f'../data/real/ppo/{args.exp}/{args.regression_model}_model.pkl',
        embed_repo=f'stair-lab/{args.exp}-embedding',
        regression_model=args.regression_model
    )
