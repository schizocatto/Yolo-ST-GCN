"""
train.py
Training and evaluation loop functions for Model_STGCN.
"""

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.losses import build_classification_criterion, compute_smoothed_alpha


# ---------------------------------------------------------------------------
# Optimizer factory
# ---------------------------------------------------------------------------

def _build_optimizer(
    model: nn.Module,
    optimizer_name: str,
    lr: float,
    weight_decay: float,
    sgd_momentum: float = 0.9,
    sgd_nesterov: bool = True,
) -> torch.optim.Optimizer:
    name = optimizer_name.strip().lower()
    if name in ('adam',):
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name in ('adamw',):
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name in ('sgd',):
        return torch.optim.SGD(
            model.parameters(), lr=lr, momentum=sgd_momentum,
            weight_decay=weight_decay, nesterov=sgd_nesterov,
        )
    raise ValueError(f"Unsupported optimizer '{optimizer_name}'. Choose: adam | adamw | sgd")


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
    grad_clip_norm: float = 1.0,
) -> Tuple[float, float]:
    """
    Run one training epoch.

    Parameters
    ----------
    grad_clip_norm : float
        Maximum L2-norm for gradient clipping (applied after backward, before
        optimizer step).  Set to 0 or a negative value to disable clipping.

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
        if grad_clip_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
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
    checkpoint_every: int = 0,
    on_checkpoint: Optional[Callable[[int, nn.Module], None]] = None,
    loss_name: str = 'ce',
    focal_gamma: float = 2.0,
    focal_alpha_mode: str = 'none',
    num_classes: Optional[int] = None,
    train_labels: Optional[torch.Tensor | np.ndarray] = None,
    start_epoch: int = 0,
    warmup_epochs: int = 0,
    optimizer_name: str = 'adam',
    sgd_momentum: float = 0.9,
    sgd_nesterov: bool = True,
    grad_clip_norm: float = 1.0,
) -> Dict[str, List[float]]:
    """
    Train `model` for `num_epochs` and return the history dictionary.

    Returns
    -------
    history : dict with keys train_loss, val_loss, train_acc, val_acc, val_f1
    """
    labels_for_alpha = train_labels
    if labels_for_alpha is None and hasattr(train_loader.dataset, 'labels'):
        labels_for_alpha = getattr(train_loader.dataset, 'labels')

    focal_alpha = None
    if loss_name.strip().lower() == 'focal' and focal_alpha_mode.strip().lower() != 'none':
        inferred_num_classes = num_classes
        if inferred_num_classes is None:
            if labels_for_alpha is None:
                raise ValueError('num_classes or train_labels is required for focal alpha smoothing.')
            labels_tensor = labels_for_alpha if isinstance(labels_for_alpha, torch.Tensor) else torch.tensor(labels_for_alpha)
            inferred_num_classes = int(labels_tensor.max().item()) + 1
        focal_alpha = compute_smoothed_alpha(
            labels=labels_for_alpha,
            num_classes=int(inferred_num_classes),
            mode=focal_alpha_mode,
        )

    criterion = build_classification_criterion(
        loss_name=loss_name,
        device=device,
        focal_gamma=focal_gamma,
        focal_alpha=focal_alpha,
    )
    optimizer = _build_optimizer(model, optimizer_name, lr, weight_decay, sgd_momentum, sgd_nesterov)
    print(f'[train] optimizer={optimizer_name}  lr={lr}  weight_decay={weight_decay}'
          f'  grad_clip_norm={grad_clip_norm if grad_clip_norm > 0 else "disabled"}')
    if warmup_epochs > 0:
        warmup_ep = min(warmup_epochs, num_epochs - 1)
        cosine_ep = num_epochs - warmup_ep
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=1e-2, end_factor=1.0, total_iters=warmup_ep),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=max(1, cosine_ep), eta_min=0.0),
            ],
            milestones=[warmup_ep],
        )
        print(f'[train] LR warmup: {warmup_ep} epochs → cosine decay for {cosine_ep} epochs')
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, num_epochs),
            eta_min=0.0,
        )

    history: Dict[str, List[float]] = {
        'train_loss': [], 'val_loss': [],
        'train_acc':  [], 'val_acc':  [], 'val_f1': [],
    }

    total_epochs = start_epoch + num_epochs
    for epoch in range(num_epochs):
        actual_epoch = start_epoch + epoch + 1
        tr_loss, tr_acc = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            show_progress=True,
            progress_desc=f'Epoch {actual_epoch}/{total_epochs} [train]',
            grad_clip_norm=grad_clip_norm,
        )
        val_loss, val_acc, val_f1, _, _ = eval_epoch(
            model,
            val_loader,
            criterion,
            device,
            show_progress=True,
            progress_desc=f'Epoch {actual_epoch}/{total_epochs} [val]',
        )
        scheduler.step()

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)

        print(
            f"Epoch {actual_epoch}/{total_epochs}  "
            f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  val_f1={val_f1:.4f}"
        )

        if checkpoint_every > 0 and on_checkpoint is not None and (actual_epoch % checkpoint_every == 0):
            on_checkpoint(actual_epoch, model)

    return history


def train_model_preloaded(
    model: nn.Module,
    train_joint_data: torch.Tensor,
    train_labels: torch.Tensor,
    val_loader: DataLoader,
    num_epochs: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    batch_size: int,
    train_bone_data: torch.Tensor | None = None,
    scheduler_step: int = 30,
    scheduler_gamma: float = 0.1,
    checkpoint_every: int = 0,
    on_checkpoint: Optional[Callable[[int, nn.Module], None]] = None,
    loss_name: str = 'ce',
    focal_gamma: float = 2.0,
    focal_alpha_mode: str = 'none',
    num_classes: Optional[int] = None,
    start_epoch: int = 0,
    warmup_epochs: int = 0,
    optimizer_name: str = 'adam',
    sgd_momentum: float = 0.9,
    sgd_nesterov: bool = True,
    grad_clip_norm: float = 1.0,
) -> Dict[str, List[float]]:
    """
    Train when full training tensors are preloaded on the target device.

    Parameters
    ----------
    train_joint_data : Tensor (N, C, T, V, M) already on ``device``
    train_labels     : Tensor (N,) already on ``device``
    train_bone_data  : optional Tensor (N, C, T, V, M) on ``device``
    """
    def _is_same_device_family(t: torch.Tensor, target: torch.device) -> bool:
        # Accept cuda vs cuda:0 as compatible; strict index equality only when both are explicit.
        td = t.device
        if td.type != target.type:
            return False
        if td.type != 'cuda':
            return True
        if target.index is None or td.index is None:
            return True
        return td.index == target.index

    if batch_size <= 0:
        raise ValueError('batch_size must be > 0 for preloaded training mode.')
    if (not _is_same_device_family(train_joint_data, device)) or (not _is_same_device_family(train_labels, device)):
        raise ValueError('Preloaded tensors must be on the same device passed to train_model_preloaded.')
    if train_bone_data is not None and (not _is_same_device_family(train_bone_data, device)):
        raise ValueError('train_bone_data must be on the same device passed to train_model_preloaded.')

    focal_alpha = None
    if loss_name.strip().lower() == 'focal' and focal_alpha_mode.strip().lower() != 'none':
        inferred_num_classes = int(num_classes) if num_classes is not None else int(train_labels.max().item()) + 1
        focal_alpha = compute_smoothed_alpha(
            labels=train_labels,
            num_classes=inferred_num_classes,
            mode=focal_alpha_mode,
        )

    criterion = build_classification_criterion(
        loss_name=loss_name,
        device=device,
        focal_gamma=focal_gamma,
        focal_alpha=focal_alpha,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if warmup_epochs > 0:
        warmup_ep = min(warmup_epochs, num_epochs - 1)
        cosine_ep = num_epochs - warmup_ep
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=1e-2, end_factor=1.0, total_iters=warmup_ep),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=max(1, cosine_ep), eta_min=0.0),
            ],
            milestones=[warmup_ep],
        )
        print(f'[train] LR warmup: {warmup_ep} epochs → cosine decay for {cosine_ep} epochs')
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, num_epochs),
            eta_min=0.0,
        )

    history: Dict[str, List[float]] = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [], 'val_f1': [],
    }

    num_samples = int(train_labels.size(0))
    num_batches = max(1, (num_samples + batch_size - 1) // batch_size)

    total_epochs = start_epoch + num_epochs
    for epoch in range(num_epochs):
        actual_epoch = start_epoch + epoch + 1
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_seen = 0

        perm = torch.randperm(num_samples, device=device)
        progress = tqdm(range(0, num_samples, batch_size), desc=f'Epoch {actual_epoch}/{total_epochs} [train-preload]', leave=False)
        for start in progress:
            idx = perm[start:start + batch_size]
            joint_batch = train_joint_data[idx]
            label_batch = train_labels[idx]
            bone_batch = train_bone_data[idx] if train_bone_data is not None else None

            optimizer.zero_grad()
            outputs = model(joint_batch, bone_batch) if bone_batch is not None else model(joint_batch)
            loss = criterion(outputs, label_batch)
            loss.backward()
            if grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            total_correct += (preds == label_batch).sum().item()
            total_seen += label_batch.size(0)

        train_loss = total_loss / num_batches
        train_acc = (total_correct / total_seen) if total_seen > 0 else 0.0

        val_loss, val_acc, val_f1, _, _ = eval_epoch(
            model,
            val_loader,
            criterion,
            device,
            show_progress=True,
            progress_desc=f'Epoch {actual_epoch}/{total_epochs} [val]',
        )
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)

        print(
            f"Epoch {actual_epoch}/{total_epochs}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  val_f1={val_f1:.4f}"
        )

        if checkpoint_every > 0 and on_checkpoint is not None and (actual_epoch % checkpoint_every == 0):
            on_checkpoint(actual_epoch, model)

    return history
