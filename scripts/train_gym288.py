"""
scripts/train_gym288.py
Train ST-GCN on Gym288-skeleton dataset.

Usage
-----
python scripts/train_gym288.py \
    --dataset_path /path/to/gym288_skeleton.pkl \
    --out_dir outputs/gym288 \
    --epochs 30 \
    --batch_size 32 \
    --lr 0.001
"""

import argparse
import json
import os
import sys

import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader

# Allow running from repo root without installing the package
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.config import GYM288_NUM_CLASSES
from src.dataset import PennActionDataset
from src.gym288_dataset import build_gym288_data_tensors, infer_num_gym288_classes
from src.model import Model_STGCN
from src.train import eval_epoch, train_model


def parse_args():
    p = argparse.ArgumentParser(description='Train ST-GCN on Gym288-skeleton')
    p.add_argument('--dataset_path', required=True, help='Path to gym288_skeleton.pkl')
    p.add_argument('--out_dir', default='outputs/gym288', help='Output directory')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--num_classes', type=int, default=0,
                   help='Override class count. 0 = infer from dataset labels')
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out_dir, exist_ok=True)

    print(f'Device: {device}')
    print('Loading Gym288-skeleton dataset...')
    data, labels, flags, _, _ = build_gym288_data_tensors(
        dataset_path=args.dataset_path,
        split='all',
        keep_unknown_split=False,
    )

    train_mask = flags == 1
    test_mask = flags == 0
    if train_mask.sum() == 0 or test_mask.sum() == 0:
        raise RuntimeError('Gym288 split is empty for train/test. Check dataset pickle structure.')

    X_train, y_train = data[train_mask], labels[train_mask]
    X_val, y_val = data[test_mask], labels[test_mask]
    print(f'Loaded {len(data)} samples  train={len(X_train)}  test={len(X_val)}')

    inferred_classes = infer_num_gym288_classes(args.dataset_path, fallback=GYM288_NUM_CLASSES)
    num_classes = args.num_classes if args.num_classes > 0 else inferred_classes
    print(f'num_classes={num_classes} (inferred={inferred_classes})')

    train_loader = DataLoader(
        PennActionDataset(X_train, y_train),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        PennActionDataset(X_val, y_val),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    model = Model_STGCN(num_classes=num_classes).to(device)
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=device,
    )

    weights_path = os.path.join(args.out_dir, 'stgcn_gym288.pth')
    torch.save(model.state_dict(), weights_path)
    print(f'Saved weights: {weights_path}')

    import torch.nn as nn
    _, _, _, preds, gt = eval_epoch(model, val_loader, nn.CrossEntropyLoss(), device)

    top1 = accuracy_score(gt, preds)
    macro_f1 = f1_score(gt, preds, average='macro', zero_division=0)
    print(f'Test Top-1 Accuracy: {top1:.4f}')
    print(f'Test Macro-F1     : {macro_f1:.4f}')

    metrics = {
        'num_classes': int(num_classes),
        'num_train': int(len(X_train)),
        'num_test': int(len(X_val)),
        'top1_accuracy': float(top1),
        'macro_f1': float(macro_f1),
    }
    with open(os.path.join(args.out_dir, 'metrics_train_gym288.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    with open(os.path.join(args.out_dir, 'classification_report.txt'), 'w', encoding='utf-8') as f:
        f.write(classification_report(gt, preds, zero_division=0))

    with open(os.path.join(args.out_dir, 'history.json'), 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)


if __name__ == '__main__':
    main()
