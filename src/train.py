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
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Single-epoch helpers
# ---------------------------------------------------------------------------

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    show_progress: bool = False,
    progress_desc: str = 'Train',
) -> Tuple[float, float]:
    """
    Run one training epoch.

    Returns
    -------
    avg_loss : float
    accuracy : float
    """
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    use_non_blocking = device.type == 'cuda'

    iterator = loader
    if show_progress:
        iterator = tqdm(loader, desc=progress_desc, leave=False)

    for batch_data, batch_labels in iterator:
        if isinstance(batch_data, (tuple, list)) and len(batch_data) == 2:
            joint_data = batch_data[0].to(device, non_blocking=use_non_blocking)
            bone_data = batch_data[1].to(device, non_blocking=use_non_blocking)
        else:
            joint_data = batch_data.to(device, non_blocking=use_non_blocking)
            bone_data = None
        batch_labels = batch_labels.to(device, non_blocking=use_non_blocking)

        optimizer.zero_grad()
        outputs = model(joint_data, bone_data) if bone_data is not None else model(joint_data)
        loss    = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        total_correct += (preds == batch_labels).sum().item()
        total_samples += batch_labels.size(0)

    if len(loader) == 0:
        return 0.0, 0.0
    avg_loss = total_loss / len(loader)
    accuracy = (total_correct / total_samples) if total_samples > 0 else 0.0
    return avg_loss, accuracy


def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    show_progress: bool = False,
    progress_desc: str = 'Val',
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
    total_loss = 0.0
    all_preds_tensors: List[torch.Tensor] = []
    all_labels_tensors: List[torch.Tensor] = []
    total_correct = 0
    total_samples = 0
    use_non_blocking = device.type == 'cuda'

    iterator = loader
    if show_progress:
        iterator = tqdm(loader, desc=progress_desc, leave=False)

    with torch.no_grad():
        for batch_data, batch_labels in iterator:
            if isinstance(batch_data, (tuple, list)) and len(batch_data) == 2:
                joint_data = batch_data[0].to(device, non_blocking=use_non_blocking)
                bone_data = batch_data[1].to(device, non_blocking=use_non_blocking)
            else:
                joint_data = batch_data.to(device, non_blocking=use_non_blocking)
                bone_data = None
            batch_labels = batch_labels.to(device, non_blocking=use_non_blocking)

            outputs  = model(joint_data, bone_data) if bone_data is not None else model(joint_data)
            loss     = criterion(outputs, batch_labels)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            total_correct += (preds == batch_labels).sum().item()
            total_samples += batch_labels.size(0)
            all_preds_tensors.append(preds.detach().cpu())
            all_labels_tensors.append(batch_labels.detach().cpu())

    if len(loader) == 0 or total_samples == 0:
        return 0.0, 0.0, 0.0, [], []

    all_preds = torch.cat(all_preds_tensors).tolist() if all_preds_tensors else []
    all_labels = torch.cat(all_labels_tensors).tolist() if all_labels_tensors else []
    avg_loss = total_loss / len(loader)
    accuracy = total_correct / total_samples
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
        tr_loss, tr_acc = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            show_progress=True,
            progress_desc=f'Epoch {epoch+1}/{num_epochs} [train]',
        )
        val_loss, val_acc, val_f1, _, _ = eval_epoch(
            model,
            val_loader,
            criterion,
            device,
            show_progress=True,
            progress_desc=f'Epoch {epoch+1}/{num_epochs} [val]',
        )
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
