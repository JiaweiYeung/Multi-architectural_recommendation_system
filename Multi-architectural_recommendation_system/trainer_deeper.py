import json
import random
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import itertools


# ====================== seed ======================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# ====================== constructor ======================
class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout_rate):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(dim, dim)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        out = self.linear1(self.norm1(x))
        out = self.relu(out)
        out = self.dropout1(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.dropout2(out)
        return x + out


class DeepResMLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate=0.5, num_classes=0, num_blocks=1):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout_rate) for _ in range(num_blocks)]
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.res_blocks(x)
        return self.classifier(x)

# ====================== save hyper-parameter ======================
def save_best_params(params: dict, filename: str = "best_hyperparams_deep.json"):
    with open(filename, "w") as f:
        json.dump(params, f, indent=4)
    print(f"Saved best params to {filename}")

def load_best_params(filename: str = "best_hyperparams_deep.json") -> dict:
    if not Path(filename).exists():
        raise FileNotFoundError(f"File '{filename}' not found.")
    with open(filename, "r") as f:
        params = json.load(f)
    print(f"Loaded best params from {filename}")
    return params

# ====================== training flow======================
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
        'BATCH_SIZE': [64, 128, 256],
        'LEARNING_RATE': [1e-4,1e-5],
        'HIDDEN_DIM': [128,256,512],
        'DROPOUT': [0.2, 0.3, 0.4, 0.5],
        # 'DROPOUT': [0.35],
        'EPOCHS': [500],
        'PATIENCE': [20],
        'NUM_BLOCKS': [1,2,3]
    # "BATCH_SIZE": 256,
    # "LEARNING_RATE": 0.0001,
    # "HIDDEN_DIM": 512,
    # "DROPOUT": 0.3,
    # "EPOCHS": 500,
    # "PATIENCE": 10,
    # "NUM_BLOCKS": 2
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Training on device:", device)

    best_val_acc = 0.0
    best_model_bundle = None
    best_params = {}

    param_combinations = list(itertools.product(
        param_grid['BATCH_SIZE'],
        param_grid['LEARNING_RATE'],
        param_grid['HIDDEN_DIM'],
        param_grid['DROPOUT'],
        param_grid['EPOCHS'],
        param_grid['PATIENCE'],
        param_grid['NUM_BLOCKS']
    ))

    for i, (BATCH_SIZE, LR, HIDDEN_DIM, DROPOUT, EPOCHS, PATIENCE, NUM_BLOCKS) in enumerate(param_combinations):
        print(f"\n Grid Search {i + 1}/{len(param_combinations)}: BS={BATCH_SIZE}, LR={LR}, HID={HIDDEN_DIM}, "
              f"DROPOUT={DROPOUT}, BLOCKS={NUM_BLOCKS}")

        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE)

        model = DeepResMLPClassifier(
            input_dim=features.shape[1],
            hidden_dim=HIDDEN_DIM,
            dropout_rate=DROPOUT,
            num_classes=len(np.unique(labels)),
            num_blocks=NUM_BLOCKS
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=LR)
        criterion = nn.CrossEntropyLoss()

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
            val_loss_total = 0.0
            val_correct, val_total = 0, 0

            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    val_loss_total += loss.item() * y_batch.size(0)
                    preds = outputs.argmax(dim=1)
                    val_correct += (preds == y_batch).sum().item()
                    val_total += y_batch.size(0)

            val_acc = val_correct / val_total
            val_loss = val_loss_total / val_total
            print(f"Epoch {epoch+1}/{EPOCHS} - Val Acc: {val_acc:.4f} - Loss: {val_loss:.4f}")

            if val_acc > best_this_val_acc:
                best_this_val_acc = val_acc
                best_this_model_state = model.state_dict()
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if early_stop_counter >= PATIENCE:
                    print(" Early stopping triggered.")
                    break

        if best_this_val_acc > best_val_acc:
            best_val_acc = best_this_val_acc
            best_model_bundle = {
                'model_state_dict': best_this_model_state,
                'model_config': {
                    'input_dim': features.shape[1],
                    'hidden_dim': HIDDEN_DIM,
                    'dropout_rate': DROPOUT,
                    'num_classes': len(np.unique(labels)),
                    'num_blocks': NUM_BLOCKS
                }
            }
            best_params = {
                'BATCH_SIZE': BATCH_SIZE,
                'LEARNING_RATE': LR,
                'HIDDEN_DIM': HIDDEN_DIM,
                'DROPOUT': DROPOUT,
                'EPOCHS': EPOCHS,
                'PATIENCE': PATIENCE,
                'NUM_BLOCKS': NUM_BLOCKS
            }
            print("Best model updated.")

    # 保存最佳模型
    if best_model_bundle:
        torch.save(best_model_bundle, 'best_grid_50vit_deep.pth')
        print(f"\nSaved best model to 'best_grid_50vit_deep.pth'")
        save_best_params(best_params)

    # 测试集评估
    print("\nEvaluating best model on test set...")
    checkpoint = torch.load("best_grid_50vit_deep.pth", map_location=device)
    model = DeepResMLPClassifier(**checkpoint['model_config']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=best_params['BATCH_SIZE'])
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for X_batch, y_batch in tqdm(test_loader, desc="Testing"):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            preds = outputs.argmax(1)
            test_correct += (preds == y_batch).sum().item()
            test_total += y_batch.size(0)

    test_acc = test_correct / test_total
    print(f"\nFinal Test Accuracy: {test_acc:.4f} using best hyperparameters: {best_params}")

if __name__ == "__main__":
    train_with_grid_search()
