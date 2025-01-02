#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# train.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score
from model import BinaryClassificationModel, FinalBinaryClassificationModel

def train_model(
    X_train_resampled, y_train_resampled, 
    X_val_scaled, y_val,
    input_dim,
    lr=0.0073, dr=0.4896, hn1=42, hn2=35, 
    batch_size=512, num_epochs=10, device='cpu'
):
    """
    Train a PyTorch model with given hyperparameters and data.
    Returns the trained model.
    """
    # Convert data to tensors
    X_train_tensor = torch.tensor(X_train_resampled, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train_resampled.values, dtype=torch.float32).unsqueeze(1).to(device)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1).to(device)

    # Create datasets and loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, criterion, optimizer
    model = FinalBinaryClassificationModel(input_dim, hn1, hn2, dr).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * batch_X.size(0)
        
        epoch_train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                preds = torch.sigmoid(outputs).cpu().numpy()
                all_preds.append(preds)
                all_labels.append(batch_y.cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        roc_auc = roc_auc_score(all_labels, all_preds)

        print(f'Epoch [{epoch+1}/{num_epochs}] '
              f'Train Loss: {epoch_train_loss:.4f} '
              f'Val ROC AUC: {roc_auc:.4f}')

    return model

def evaluate_model(model, X_val_scaled, y_val, device='cpu'):
    """
    Evaluate a model on the validation set and print common metrics.
    """
    model.eval()
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1).to(device)

    with torch.no_grad():
        outputs = model(X_val_tensor)
        preds = torch.sigmoid(outputs).cpu().numpy()

    binary_preds = (preds >= 0.5).astype(int)
    accuracy = accuracy_score(y_val, binary_preds)
    roc_auc = roc_auc_score(y_val, preds)
    auprc = average_precision_score(y_val, preds)
    f1 = f1_score(y_val, binary_preds)

    print("\nValidation Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"AUPRC: {auprc:.4f}")
    print(f"F1 Score: {f1:.4f}")

