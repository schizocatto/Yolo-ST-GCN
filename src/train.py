"""
train.py
Training and evaluation loop functions for Model_STGCN.
"""

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Single-epoch helpers
# ---------------------------------------------------------------------------

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Run one training epoch.

    Returns
    -------
    avg_loss : float
    accuracy : float
    """
    model.train()
    total_loss, all_preds, all_labels = 0.0, [], []

    for batch_data, batch_labels in loader:
        batch_data   = batch_data.to(device)
        batch_labels = batch_labels.to(device)

        optimizer.zero_grad()
        outputs = model(batch_data)
        loss    = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch_labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy


def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float, List[int], List[int]]:
    """
    Run one evaluation epoch.

    Returns
    -------
    avg_loss  : float
    accuracy  : float
    macro_f1  : float
    all_preds : list[int]
    all_labels: list[int]
    """
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []

    with torch.no_grad():
        for batch_data, batch_labels in loader:
            batch_data   = batch_data.to(device)
            batch_labels = batch_labels.to(device)

            outputs  = model(batch_data)
            loss     = criterion(outputs, batch_labels)
            total_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())

    if len(loader) == 0 or not all_labels:
        return 0.0, 0.0, 0.0, [], []
    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return avg_loss, accuracy, macro_f1, all_preds, all_labels


# ---------------------------------------------------------------------------
# Full training loop
# ---------------------------------------------------------------------------

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    scheduler_step: int = 30,
    scheduler_gamma: float = 0.1,
) -> Dict[str, List[float]]:
    """
    Train `model` for `num_epochs` and return the history dictionary.

    Returns
    -------
    history : dict with keys train_loss, val_loss, train_acc, val_acc, val_f1
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=scheduler_step, gamma=scheduler_gamma
    )

    history: Dict[str, List[float]] = {
        'train_loss': [], 'val_loss': [],
        'train_acc':  [], 'val_acc':  [], 'val_f1': [],
    }

    for epoch in range(num_epochs):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1, _, _ = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)

        print(
            f"Epoch {epoch+1}/{num_epochs}  "
            f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  val_f1={val_f1:.4f}"
        )

    return history
