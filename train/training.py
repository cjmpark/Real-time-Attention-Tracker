
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.optim as optim

import os
import sys
import random
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from classifiers.attn_cls import AttentionNet


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CustomData(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)


def train_model(x_train, y_train, x_val, y_val, batch_sizes, lrs, num_epoch, results, name, weight_decay, seed):
    set_seed(seed)

    for batch_size in tqdm(batch_sizes):
        for lr in lrs:
            print(f"batch_size: {batch_size}, lr: {lr}")

            train_data = CustomData(x_train, y_train)
            val_data = CustomData(x_val, y_val)

            loader_train = DataLoader(train_data, batch_size=batch_size, shuffle=True)
            loader_val = DataLoader(val_data, batch_size=batch_size, shuffle=False)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = AttentionNet(x_train.shape[1]).to(device)
            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

            best_val_loss = float('inf')
            best_val_acc = 0.0
            best_epoch = -1
            best_model_state = None
            best_train_loss = None
            best_train_acc = None

            train_loss_per_epoch = []
            val_loss_per_epoch = []

            for epoch in range(num_epoch):
                model.train()
                total_train_loss = 0
                correct_train = total_samples_train = 0

                for input_vec, labels in loader_train:
                    input_vec, labels = input_vec.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(input_vec)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    total_train_loss += loss.item() * input_vec.size(0)
                    preds = (torch.sigmoid(outputs) > 0.5)
                    correct_train += (preds == labels.bool()).sum().item()
                    total_samples_train += labels.size(0)

                train_loss = total_train_loss / total_samples_train
                train_acc = correct_train / total_samples_train

                model.eval()
                total_val_loss = 0
                correct_val = total_samples_val = 0

                with torch.no_grad():
                    for input_vec, labels in loader_val:
                        input_vec, labels = input_vec.to(device), labels.to(device)
                        outputs = model(input_vec)
                        loss = criterion(outputs, labels)
                        total_val_loss += loss.item() * input_vec.size(0)
                        preds = (torch.sigmoid(outputs) > 0.5)
                        correct_val += (preds == labels.bool()).sum().item()
                        total_samples_val += labels.size(0)

                val_loss = total_val_loss / total_samples_val
                val_acc = correct_val / total_samples_val

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_acc = val_acc
                    best_epoch = epoch
                    best_model_state = model.state_dict()
                    best_train_loss = train_loss
                    best_train_acc = train_acc

                train_loss_per_epoch.append(train_loss)
                val_loss_per_epoch.append(val_loss)

            results.append({
                "batch_size": batch_size,
                "lr": lr,
                "best_val_loss": best_val_loss,
                "best_val_acc": best_val_acc,
                "best_epoch": best_epoch,
                "train_loss": best_train_loss,
                "train_acc": best_train_acc
            })

            torch.save(best_model_state, f"train/potential_models/{name}_bs{batch_size}_lr{lr}.pth")

            plt.figure()
            plt.plot(train_loss_per_epoch, label="Train Loss")
            plt.plot(val_loss_per_epoch, label="Val Loss")
            plt.title(f"Loss Curve (bs={batch_size}, lr={lr})")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig(f"train/loss_plots/{name}_bs{batch_size}_lr{lr}.png")
            plt.close()

    return results


def load_data(train_path, val_path, header):
    df_train = pd.read_csv(train_path, header=header).dropna()
    print("len of train:", len(df_train))
    x_train = df_train.iloc[:, :-1].values.astype(np.float32)
    y_train = df_train.iloc[:, -1].values.astype(np.float32)

    df_val = pd.read_csv(val_path, header=header).dropna()
    print("len of val:", len(df_val))
    x_val = df_val.iloc[:, :-1].values.astype(np.float32)
    y_val = df_val.iloc[:, -1].values.astype(np.float32)
    return x_train, y_train, x_val, y_val


def run_multiple_seeds(x_train, y_train, x_val, y_val, batch_sizes, lrs, num_epoch, name, weight_decay, seeds):
    all_summaries = []

    for batch_size in batch_sizes:
        for lr in lrs:
            val_accs = []
            val_losses = []
            print(f"\n--- Running batch_size={batch_size}, lr={lr} ---")

            for seed in seeds:
                print(f" → Seed {seed}")
                results = []
                train_model(
                    x_train, y_train, x_val, y_val,
                    [batch_size], [lr], num_epoch, results,
                    name=f"{name}_s{seed}_bs{batch_size}_lr{lr}",
                    weight_decay=weight_decay,
                    seed=seed
                )
                val_accs.append(results[0]["best_val_acc"])
                val_losses.append(results[0]["best_val_loss"])

            all_summaries.append({
                "batch_size": batch_size,
                "lr": lr,
                "val_accs": val_accs,
                "val_losses": val_losses,
                "mean_val_acc": np.mean(val_accs),
                "std_val_acc": np.std(val_accs),
                "mean_val_loss": np.mean(val_losses),
                "std_val_loss": np.std(val_losses)
            })

            print(f"*****Mean Val Acc: {np.mean(val_accs):.4f} ± {np.std(val_accs):.4f}")
            print(f"*****Mean Val Loss: {np.mean(val_losses):.4f} ± {np.std(val_losses):.4f}")

    return all_summaries


if __name__ == "__main__":
    x_train, y_train, x_val, y_val = load_data("data_set/setB_scaled.csv", "data_set/setA_scaled.csv", header=0)

    summary = run_multiple_seeds(
        x_train, y_train, x_val, y_val,
        batch_sizes=[32, 64, 128],
        lrs=[1e-5, 3e-5, 5e-6],
        num_epoch=200,
        name="L2_mm_sc",
        weight_decay=1e-4,
        seeds=[5, 29, 111, 334, 888]
    )

    with open("train/summary_by_seed.json", "w") as f:
        json.dump(summary, f, indent=4)