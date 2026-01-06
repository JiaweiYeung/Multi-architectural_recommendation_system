import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import json
from pathlib import Path
import itertools


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate=0.0, num_classes=0):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.fc(x)


def save_best_params(params: dict, filename: str = "best_hyperparams.json"):
    """Save best hyperparameter dictionary to a JSON file."""
    with open(filename, "w") as f:
        json.dump(params, f, indent=4)
    print(f"Saved best params to {filename}")


def load_best_params(filename: str = "best_hyperparams.json") -> dict:
    """Load best hyperparameter dictionary from a JSON file."""
    if not Path(filename).exists():
        raise FileNotFoundError(f"File '{filename}' not found.")
    with open(filename, "r") as f:
        params = json.load(f)
    print(f"Loaded best params from {filename}")
    return params


def train_with_grid_search():
    df = pd.read_csv("features_50vit.csv")
    labels = df['label'].values
    features = df.drop(columns=['label', 'image_path']).values

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        features, labels, test_size=0.1, random_state=42, stratify=labels)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.1111, random_state=42, stratify=y_train_val)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_val = torch.tensor(y_val, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    param_grid = {
        # 'BATCH_SIZE': [64],
        # 'LEARNING_RATE': [1e-3, 1e-4],
        # 'HIDDEN_DIM': [256],
        # 'DROPOUT': [0.5],
        # 'EPOCHS': [10],
        # 'PATIENCE': [5]

        'BATCH_SIZE': [64,128],
        'LEARNING_RATE': [1e-4, 1e-4, 1e-5],
        'HIDDEN_DIM': [128, 256, 512],
        'DROPOUT': [0.1, 0.3, 0.4, 0.5],
        'EPOCHS': [500],
        'PATIENCE': [10]
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("training on device:", device)
    best_val_acc = 0.0
    best_model_state = None
    best_params = {}

    param_combinations = list(itertools.product(
        param_grid['BATCH_SIZE'],
        param_grid['LEARNING_RATE'],
        param_grid['HIDDEN_DIM'],
        param_grid['DROPOUT'],
        param_grid['EPOCHS'],
        param_grid['PATIENCE']
    ))

    for i, (BATCH_SIZE, LR, HIDDEN_DIM, DROPOUT, EPOCHS, PATIENCE) in enumerate(param_combinations):
        print(
            f"\n🔍 Grid Search {i + 1}/{len(param_combinations)}: BS={BATCH_SIZE}, LR={LR}, HIDDEN={HIDDEN_DIM}, DROPOUT={DROPOUT}")

        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE)

        model = MLPClassifier(input_dim=features.shape[1], hidden_dim=HIDDEN_DIM,
                              dropout_rate=DROPOUT, num_classes=len(np.unique(labels))).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LR)

        early_stop_counter = 0
        best_this_val_acc = 0.0
        best_this_model_state = None

        for epoch in range(EPOCHS):
            model.train()
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

            model.eval()
            val_correct, val_total = 0, 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    outputs = model(X_batch)
                    preds = outputs.argmax(1)
                    val_correct += (preds == y_batch).sum().item()
                    val_total += y_batch.size(0)
            val_acc = val_correct / val_total
            print(f"Epoch {epoch + 1}/{EPOCHS} - Val Acc: {val_acc:.4f}")

            if val_acc > best_this_val_acc:
                best_this_val_acc = val_acc
                best_this_model_state = model.state_dict()
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if early_stop_counter >= PATIENCE:
                    print("Early stopping triggered.")
                    break

        if best_this_val_acc > best_val_acc:
            best_val_acc = best_this_val_acc
            best_model_state = best_this_model_state
            best_params = {
                'BATCH_SIZE': BATCH_SIZE,
                'LEARNING_RATE': LR,
                'HIDDEN_DIM': HIDDEN_DIM,
                'DROPOUT': DROPOUT,
                'EPOCHS': EPOCHS,
                'PATIENCE': PATIENCE
            }
            print("Best model so far updated.")

    if best_model_state:
        torch.save(best_model_state, 'best_grid_50vit.pth')
        print(f"\n Best model saved. Params: {best_params}")
        save_best_params(best_params)

    model = MLPClassifier(input_dim=features.shape[1],
                          hidden_dim=best_params['HIDDEN_DIM'],
                          dropout_rate=best_params['DROPOUT'],
                          num_classes=len(np.unique(labels))).to(device)
    model.load_state_dict(torch.load('best_grid_50vit.pth'))
    model.eval()

    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=best_params['BATCH_SIZE'])
    test_correct, test_total = 0, 0
    with torch.no_grad():
        for X_batch, y_batch in tqdm(test_loader, desc=" Testing"):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            preds = outputs.argmax(1)
            test_correct += (preds == y_batch).sum().item()
            test_total += y_batch.size(0)

    test_acc = test_correct / test_total
    print(f"\n Final Test Accuracy: {test_acc:.4f} with best hyperparams: {best_params}")


if __name__ == "__main__":
    train_with_grid_search()