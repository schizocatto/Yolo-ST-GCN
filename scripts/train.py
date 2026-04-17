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
from src.checkpointing import save_checkpoint
from src.dataset import build_data_tensors, PennActionDataset
from src.experiment_config import apply_overrides, load_experiment_config
from src.losses import build_classification_criterion, compute_smoothed_alpha
from src.model import Model_STGCN
from src.two_stream_stgcn import TwoStream_STGCN
from src.train import train_model, eval_epoch
from src.visualize import plot_training_curves, plot_confusion_matrix, plot_per_class_f1


def parse_args():
    p = argparse.ArgumentParser(description='Train ST-GCN on Penn Action')
    p.add_argument('--experiment_config', default='',
                   help='Optional JSON config for frequent experiment updates.')
    p.add_argument('--labels_dir', required=True,
                   help='Path to dataset labels directory (Penn .mat or COCO .npz)')
    p.add_argument('--dataset_format', default='penn', choices=['penn', 'coco'],
                   help='Input dataset format. COCO will be remapped to Penn layout.')
    p.add_argument('--joint_spec_name', default='penn14', choices=['penn14', 'coco18'],
                   help='Target joint layout for model input.')
    p.add_argument('--out_dir',    default='outputs',
                   help='Output directory for weights and plots')
    p.add_argument('--epochs',     type=int,   default=50)
    p.add_argument('--batch_size', type=int,   default=32)
    p.add_argument('--lr',         type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--num_workers', '--num_wokers', dest='num_workers', type=int, default=0,
                   help='Number of DataLoader workers for both train/val (supports alias --num_wokers).')
    p.add_argument('--use_two_stream', action='store_true',
                   help='Enable 2s-STGCN (joint stream + bone stream with late fusion).')
    p.add_argument('--save_every_epochs', type=int, default=10,
                   help='Save periodic checkpoints every N epochs (0 to disable).')
    p.add_argument('--loss_name', default='ce', choices=['ce', 'cross_entropy', 'focal'],
                   help='Classification loss type.')
    p.add_argument('--focal_gamma', type=float, default=2.0,
                   help='Focal Loss gamma parameter.')
    p.add_argument('--focal_alpha_mode', default='none', choices=['none', 'inverse', 'sqrt_inverse'],
                   help='Class alpha weighting mode for focal loss.')
    p.add_argument('--grad_clip_norm', type=float, default=1.0,
                   help='Max gradient L2-norm for clipping (applied after backward, before optimizer step). '
                        'Set to 0 or negative to disable gradient clipping.')
    return p.parse_args()


def main():
    args   = parse_args()
    if args.experiment_config:
        cfg = load_experiment_config(args.experiment_config)
        args = apply_overrides(args, cfg, sys.argv[1:])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    os.makedirs(args.out_dir, exist_ok=True)

    # ── Data (official subject-isolated split via Penn Action train flag) ──
    print('Loading data...')
    if args.use_two_stream:
        data, bone_data, labels, flags, _, _ = build_data_tensors(
            labels_dir=args.labels_dir,
            dataset_format=args.dataset_format,
            joint_spec_name=args.joint_spec_name,
            return_bone_data=True,
        )
    else:
        data, labels, flags, _, _ = build_data_tensors(
            labels_dir=args.labels_dir,
            dataset_format=args.dataset_format,
            joint_spec_name=args.joint_spec_name,
        )
    print(f'  Loaded {len(data)} samples  '
          f'(train flag=1: {(flags==1).sum()}  test flag=0: {(flags==0).sum()})')

    X_train, y_train = data[flags == 1], labels[flags == 1]
    X_val,   y_val   = data[flags == 0], labels[flags == 0]
    B_train = B_val = None
    if args.use_two_stream:
        B_train = bone_data[flags == 1]
        B_val = bone_data[flags == 0]

    focal_alpha = None
    if args.loss_name == 'focal' and args.focal_alpha_mode != 'none':
        focal_alpha = compute_smoothed_alpha(y_train, num_classes=len(EXERCISE_CLASSES), mode=args.focal_alpha_mode)

    train_loader = DataLoader(
        PennActionDataset(
            X_train,
            y_train,
            bone_data=B_train,
            include_bone=args.use_two_stream,
            joint_spec_name=args.joint_spec_name,
        ),
        batch_size=args.batch_size, shuffle=True, drop_last=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        PennActionDataset(
            X_val,
            y_val,
            bone_data=B_val,
            include_bone=args.use_two_stream,
            joint_spec_name=args.joint_spec_name,
        ),
        batch_size=args.batch_size, shuffle=False, drop_last=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    print(
        f'  Train: {len(X_train)}  Val: {len(X_val)}  '
        f'num_workers={args.num_workers}  two_stream={args.use_two_stream}'
    )

    # ── Model ────────────────────────────────────────────────────────────
    model = (
        TwoStream_STGCN(num_classes=len(EXERCISE_CLASSES), joint_spec=args.joint_spec_name)
        if args.use_two_stream
        else Model_STGCN(joint_spec=args.joint_spec_name)
    ).to(device)

    def save_periodic_checkpoint(epoch_no: int, model_obj: torch.nn.Module) -> None:
        periodic_name = (
            f'stgcn_penn_action_2s_epoch{epoch_no}.pth'
            if args.use_two_stream
            else f'stgcn_penn_action_epoch{epoch_no}.pth'
        )
        periodic_path = os.path.join(args.out_dir, periodic_name)
        save_checkpoint(
            periodic_path,
            model_obj,
            metadata={
                'joint_spec_name': args.joint_spec_name,
                'use_two_stream': bool(args.use_two_stream),
                'dataset_format': args.dataset_format,
                'num_classes': len(EXERCISE_CLASSES),
                'epoch': int(epoch_no),
                'periodic_checkpoint': True,
            },
        )
        print(f'Periodic checkpoint saved: {periodic_path}')

    # ── Train ────────────────────────────────────────────────────────────
    history = train_model(
        model, train_loader, val_loader,
        num_epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=device,
        checkpoint_every=args.save_every_epochs,
        on_checkpoint=save_periodic_checkpoint,
        loss_name=args.loss_name,
        focal_gamma=args.focal_gamma,
        focal_alpha_mode=args.focal_alpha_mode,
        num_classes=len(EXERCISE_CLASSES),
        train_labels=y_train,
        grad_clip_norm=args.grad_clip_norm,
    )

    # ── Save weights ─────────────────────────────────────────────────────
    weights_name = 'stgcn_penn_action_2s.pth' if args.use_two_stream else 'stgcn_penn_action.pth'
    weights_path = os.path.join(args.out_dir, weights_name)
    save_checkpoint(
        weights_path,
        model,
        metadata={
            'joint_spec_name': args.joint_spec_name,
            'use_two_stream': bool(args.use_two_stream),
            'dataset_format': args.dataset_format,
            'num_classes': len(EXERCISE_CLASSES),
        },
    )
    print(f'Model saved to {weights_path}')

    # ── Final evaluation ─────────────────────────────────────────────────
    eval_criterion = build_classification_criterion(
        loss_name=args.loss_name,
        device=device,
        focal_gamma=args.focal_gamma,
        focal_alpha=focal_alpha,
    )
    _, _, _, preds, gt = eval_epoch(model, val_loader, eval_criterion, device)
    print('\n--- Classification Report ---')
    print(classification_report(gt, preds, target_names=EXERCISE_CLASSES, zero_division=0))

    # ── Plots ─────────────────────────────────────────────────────────────
    plot_training_curves(history,    out_dir=args.out_dir)
    plot_confusion_matrix(gt, preds, out_dir=args.out_dir)
    plot_per_class_f1(gt, preds,     out_dir=args.out_dir)


if __name__ == '__main__':
    main()
