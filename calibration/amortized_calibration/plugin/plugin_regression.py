import argparse
import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from datasets import concatenate_datasets, load_dataset
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from utils import MLP, set_seed, split_indices


def train_mlp(
    emb_train: torch.Tensor,
    emb_test: torch.Tensor,
    z_train: torch.Tensor,
    z_test: torch.Tensor,
    batch_size: int = 4096,
    max_epoch: int = 120,
    lr: float = 0.001,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_dim = emb_train.shape[1]
    model = MLP(input_dim).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_dataset = TensorDataset(emb_train, z_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = TensorDataset(emb_test, z_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    pbar = tqdm(range(max_epoch))
    for _ in pbar:
        model.train()
        for emb_batch, z_batch in train_loader:
            emb_batch, z_batch = emb_batch.to(device), z_batch.to(device)
            optimizer.zero_grad()
            outputs = model(emb_batch)
            loss = criterion(outputs, z_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        total_train_loss = 0
        total_test_loss = 0
        with torch.no_grad():
            for emb_batch, z_batch in train_loader:
                emb_batch, z_batch = emb_batch.to(device), z_batch.to(device)
                outputs = model(emb_batch)
                loss = criterion(outputs, z_batch)
                total_train_loss += loss.item()
            for emb_batch, z_batch in test_loader:
                emb_batch, z_batch = emb_batch.to(device), z_batch.to(device)
                outputs = model(emb_batch)
                loss = criterion(outputs, z_batch)
                total_test_loss += loss.item()

        train_loss = total_train_loss / len(train_loader)
        test_loss = total_test_loss / len(test_loader)
        wandb.log({"train_loss": train_loss, "test_loss": test_loss})
        pbar.set_postfix({"train_loss": train_loss, "test_loss": test_loss})

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    with torch.no_grad():
        z_train_pred = []
        for emb_batch in train_loader:
            emb_batch = emb_batch[0].to(device)
            outputs = model(emb_batch)
            z_train_pred.append(outputs.cpu().numpy())
        z_train_pred = np.concatenate(z_train_pred).flatten()

        z_test_pred = []
        for emb_batch in test_loader:
            emb_batch = emb_batch[0].to(device)
            outputs = model(emb_batch)
            z_test_pred.append(outputs.cpu().numpy())
        z_test_pred = np.concatenate(z_test_pred).flatten()

    return z_train_pred, z_test_pred, model.cpu()


def main(
    train_indices,
    test_indices,
    emb_train,
    z_train,
    emb_test,
    z_test,
    df_train_path,
    df_test_path,
    save_model_path=None,
):
    z_train_pred, z_test_pred, model = train_mlp(
        emb_train=torch.tensor(emb_train, dtype=torch.float32),
        emb_test=torch.tensor(emb_test, dtype=torch.float32),
        z_train=torch.tensor(z_train, dtype=torch.float32).view(-1, 1),
        z_test=torch.tensor(z_test, dtype=torch.float32).view(-1, 1),
    )

    df_train = pd.DataFrame(
        {
            "index": train_indices,
            "z_true": z_train,
            "z_pred": z_train_pred,
        }
    )
    df_train.to_csv(df_train_path, index=False)

    df_test = pd.DataFrame(
        {
            "index": test_indices,
            "z_true": z_test,
            "z_pred": z_test_pred,
        }
    )
    df_test.to_csv(df_test_path, index=False)

    if save_model_path is not None:
        with open(save_model_path, "wb") as f:
            pickle.dump(model, f)


if __name__ == "__main__":
    wandb.init(project="plugin_regression")
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--model", type=str, default="mlp", choices=["mlp"])
    parser.add_argument(
        "--task", type=str, default="byrandom", choices=["byrandom", "bydataset"]
    )
    args = parser.parse_args()

    output_dir = f"../data/plugin_regression/{args.dataset}"
    os.makedirs(output_dir, exist_ok=True)

    set_seed(args.seed)
    if args.dataset != "aggregate":
        assert args.task == "byrandom"
        dataset = load_dataset(f"stair-lab/reeval_individual-embed", split=args.dataset)
        emb = np.array(dataset["embed"])
        z = np.array(dataset["z"])

        train_indices, test_indices = split_indices(z.shape[0])
        emb_train, z_train = emb[train_indices], z[train_indices]
        emb_test, z_test = emb[test_indices], z[test_indices]

        main(
            train_indices,
            test_indices,
            emb_train,
            z_train,
            emb_test,
            z_test,
            df_train_path=f"{output_dir}/train_{args.seed}.csv",
            df_test_path=f"{output_dir}/test_{args.seed}.csv",
            save_model_path=(
                f"{output_dir}/{args.model}.pkl" if args.seed == 0 else None
            ),
        )

    else:  # args.dataset == 'aggregate'
        if args.task == "byrandom":
            dataset_info = load_dataset("stair-lab/reeval_aggregate-embed", split=None)
            splits = dataset_info.keys()
            datasets = [
                load_dataset("stair-lab/reeval_aggregate-embed", split=split)
                for split in splits
            ]
            dataset = concatenate_datasets(datasets)
            emb = np.array(dataset["embed"])
            z = np.array(dataset["z"])

            train_indices, test_indices = split_indices(z.shape[0])
            emb_train, z_train = emb[train_indices], z[train_indices]
            emb_test, z_test = emb[test_indices], z[test_indices]

            main(
                train_indices,
                test_indices,
                emb_train,
                z_train,
                emb_test,
                z_test,
                df_train_path=f"{output_dir}/train_byrandom_{args.seed}.csv",
                df_test_path=f"{output_dir}/test_byrandom_{args.seed}.csv",
                save_model_path=(
                    f"{output_dir}/{args.model}_byrandom.pkl"
                    if args.seed == 0
                    else None
                ),
            )

        else:  # args.task == 'bydataset'
            dataset_info = load_dataset("stair-lab/reeval_aggregate-embed", split=None)
            splits = list(dataset_info.keys())
            train_dataset_indices, test_dataset_indices = split_indices(len(splits))
            train_splits = [splits[i] for i in train_dataset_indices]
            test_splits = [splits[i] for i in test_dataset_indices]
            print(f"Test Splits: {test_splits}")

            train_datasets = [
                load_dataset("stair-lab/reeval_aggregate-embed", split=split)
                for split in train_splits
            ]
            train_dataset = concatenate_datasets(train_datasets)
            emb_train = np.array(train_dataset["embed"])
            z_train = np.array(train_dataset["z"])

            test_datasets = [
                load_dataset("stair-lab/reeval_aggregate-embed", split=split)
                for split in test_splits
            ]
            test_dataset = concatenate_datasets(test_datasets)
            emb_test = np.array(test_dataset["embed"])
            z_test = np.array(test_dataset["z"])

            train_indices = []
            for split, dataset in zip(train_splits, train_datasets):
                train_indices.extend([split] * len(dataset))

            test_indices = []
            for split, dataset in zip(test_splits, test_datasets):
                test_indices.extend([split] * len(dataset))

            main(
                train_indices,
                test_indices,
                emb_train,
                z_train,
                emb_test,
                z_test,
                df_train_path=f"{output_dir}/train_bydataset_{args.seed}.csv",
                df_test_path=f"{output_dir}/test_bydataset_{args.seed}.csv",
                save_model_path=(
                    f"{output_dir}/{args.model}_bydataset.pkl"
                    if args.seed == 0
                    else None
                ),
            )
