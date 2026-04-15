"""
scripts/train.py
CLI entry point for training ST-GCN on Penn Action.

Usage
-----
python scripts/train.py \
    --labels_dir /path/to/Penn_Action/labels \
    --out_dir    outputs/ \
    --epochs     50 \
    --batch_size 32 \
    --lr         0.001

Saves
-----
outputs/stgcn_penn_action.pth     — best model weights
outputs/training_curves.png       — loss / accuracy / F1 plots
outputs/confusion_matrix.png
outputs/per_class_f1.png
"""

import argparse
import os
import sys

import numpy as np
import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader

# Allow running from repo root without installing the package
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.config import EXERCISE_CLASSES
from src.dataset import build_data_tensors, PennActionDataset
from src.model import Model_STGCN
from src.train import train_model, eval_epoch
from src.visualize import plot_training_curves, plot_confusion_matrix, plot_per_class_f1


def parse_args():
    p = argparse.ArgumentParser(description='Train ST-GCN on Penn Action')
    p.add_argument('--labels_dir', required=True,
                   help='Path to dataset labels directory (Penn .mat or COCO .npz)')
    p.add_argument('--dataset_format', default='penn', choices=['penn', 'coco'],
                   help='Input dataset format. COCO will be remapped to Penn layout.')
    p.add_argument('--out_dir',    default='outputs',
                   help='Output directory for weights and plots')
    p.add_argument('--epochs',     type=int,   default=50)
    p.add_argument('--batch_size', type=int,   default=32)
    p.add_argument('--lr',         type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--num_workers', '--num_wokers', dest='num_workers', type=int, default=0,
                   help='Number of DataLoader workers for both train/val (supports alias --num_wokers).')
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    os.makedirs(args.out_dir, exist_ok=True)

    # ── Data (official subject-isolated split via Penn Action train flag) ──
    print('Loading data...')
    data, labels, flags, _, _ = build_data_tensors(
        labels_dir=args.labels_dir,
        dataset_format=args.dataset_format,
    )
    print(f'  Loaded {len(data)} samples  '
          f'(train flag=1: {(flags==1).sum()}  test flag=0: {(flags==0).sum()})')

    X_train, y_train = data[flags == 1], labels[flags == 1]
    X_val,   y_val   = data[flags == 0], labels[flags == 0]

    train_loader = DataLoader(
        PennActionDataset(X_train, y_train),
        batch_size=args.batch_size, shuffle=True, drop_last=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        PennActionDataset(X_val, y_val),
        batch_size=args.batch_size, shuffle=False, drop_last=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    print(f'  Train: {len(X_train)}  Val: {len(X_val)}  num_workers={args.num_workers}')

    # ── Model ────────────────────────────────────────────────────────────
    model = Model_STGCN().to(device)

    # ── Train ────────────────────────────────────────────────────────────
    history = train_model(
        model, train_loader, val_loader,
        num_epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=device,
    )

    # ── Save weights ─────────────────────────────────────────────────────
    weights_path = os.path.join(args.out_dir, 'stgcn_penn_action.pth')
    torch.save(model.state_dict(), weights_path)
    print(f'Model saved to {weights_path}')

    # ── Final evaluation ─────────────────────────────────────────────────
    import torch.nn as nn
    _, _, _, preds, gt = eval_epoch(model, val_loader, nn.CrossEntropyLoss(), device)
    print('\n--- Classification Report ---')
    print(classification_report(gt, preds, target_names=EXERCISE_CLASSES, zero_division=0))

    # ── Plots ─────────────────────────────────────────────────────────────
    plot_training_curves(history,    out_dir=args.out_dir)
    plot_confusion_matrix(gt, preds, out_dir=args.out_dir)
    plot_per_class_f1(gt, preds,     out_dir=args.out_dir)


if __name__ == '__main__':
    main()
