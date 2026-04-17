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
from src.checkpointing import save_checkpoint
from src.dataset import PennActionDataset
from src.experiment_config import apply_overrides, load_experiment_config
from src.gym288_dataset import build_gym288_data_tensors, infer_num_gym288_classes
from src.losses import build_classification_criterion, compute_smoothed_alpha
from src.model import Model_STGCN
from src.train import eval_epoch, train_model
from src.two_stream_stgcn import TwoStream_STGCN
from src.feeder import SkeletonFeeder, build_feeder_pair, make_weighted_sampler


def parse_args():
    p = argparse.ArgumentParser(description='Train ST-GCN on Gym288-skeleton')
    p.add_argument('--experiment_config', default='',
                   help='Optional JSON config for frequent experiment updates.')
    p.add_argument('--dataset_path', required=True, help='Path to gym288_skeleton.pkl')
    p.add_argument('--out_dir', default='outputs/gym288', help='Output directory')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--num_classes', type=int, default=0,
                   help='Override class count. 0 = infer from dataset labels')
    p.add_argument('--joint_spec_name', default='penn14', choices=['penn14', 'coco18'],
                   help='Target joint layout for model input.')
    p.add_argument('--num_workers', '--num_wokers', dest='num_workers', type=int, default=0,
                   help='Number of DataLoader workers for both train/val (supports alias --num_wokers).')
    p.add_argument('--max_train_samples', type=int, default=0,
                   help='For quick smoke runs: keep only first N train samples (0 = all).')
    p.add_argument('--max_val_samples', type=int, default=0,
                   help='For quick smoke runs: keep only first N val samples (0 = all).')
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
    p.add_argument('--use_augment_feeder', action='store_true',
                   help='Use the smart class-imbalance aware SkeletonFeeder instead of basic dataset.')
    p.add_argument('--use_weighted_sampler', action='store_true',
                   help='Use WeightedRandomSampler to oversample minority classes. Useful with --use_augment_feeder.')
    return p.parse_args()


def main():
    args = parse_args()
    if args.experiment_config:
        cfg = load_experiment_config(args.experiment_config)
        args = apply_overrides(args, cfg, sys.argv[1:])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out_dir, exist_ok=True)

    print(f'Device: {device}')
    if args.dataset_path.startswith('/content/drive'):
        print('[warning] Dataset is on Google Drive path; copy to /content for faster I/O.')
    print('Loading Gym288-skeleton dataset...')
    if args.use_two_stream:
        data, bone_data, labels, flags, _, _ = build_gym288_data_tensors(
            dataset_path=args.dataset_path,
            joint_spec_name=args.joint_spec_name,
            split='all',
            keep_unknown_split=False,
            return_bone_data=True,
        )
    else:
        data, labels, flags, _, _ = build_gym288_data_tensors(
            dataset_path=args.dataset_path,
            joint_spec_name=args.joint_spec_name,
            split='all',
            keep_unknown_split=False,
        )

    train_mask = flags == 1
    test_mask = flags == 0
    if train_mask.sum() == 0 or test_mask.sum() == 0:
        raise RuntimeError('Gym288 split is empty for train/test. Check dataset pickle structure.')

    X_train, y_train = data[train_mask], labels[train_mask]
    X_val, y_val = data[test_mask], labels[test_mask]
    B_train = B_val = None
    if args.use_two_stream:
        B_train = bone_data[train_mask]
        B_val = bone_data[test_mask]

    if args.max_train_samples > 0:
        n = min(args.max_train_samples, len(X_train))
        X_train, y_train = X_train[:n], y_train[:n]
        if B_train is not None:
            B_train = B_train[:n]
    if args.max_val_samples > 0:
        n = min(args.max_val_samples, len(X_val))
        X_val, y_val = X_val[:n], y_val[:n]
        if B_val is not None:
            B_val = B_val[:n]
    print(f'Loaded {len(data)} samples  train={len(X_train)}  test={len(X_val)}')

    inferred_classes = infer_num_gym288_classes(args.dataset_path, fallback=GYM288_NUM_CLASSES)
    num_classes = args.num_classes if args.num_classes > 0 else inferred_classes
    print(f'num_classes={num_classes} (inferred={inferred_classes})')

    focal_alpha = None
    if args.loss_name == 'focal' and args.focal_alpha_mode != 'none':
        focal_alpha = compute_smoothed_alpha(y_train, num_classes=num_classes, mode=args.focal_alpha_mode)

    # This dataset is fully materialized in RAM; extra workers often slow Colab due to IPC overhead.
    effective_workers = args.num_workers
    if args.num_workers > 0:
        print('[info] Using in-memory tensors; forcing num_workers=0 to avoid dataloader overhead.')
        effective_workers = 0

    if args.use_augment_feeder:
        flip_pairs = (
            SkeletonFeeder.PENN14_FLIP_PAIRS if args.joint_spec_name == 'penn14'
            else None  # Adjust if COCO18 flip pairs are needed
        )
        train_ds, val_ds = build_feeder_pair(
            train_data=X_train,
            train_labels=y_train,
            val_data=X_val,
            val_labels=y_val,
            train_bone=B_train,
            val_bone=B_val,
            include_bone=args.use_two_stream,
            flip_pairs=flip_pairs,
            verbose=True,
        )
    else:
        train_ds = PennActionDataset(
            X_train, y_train, bone_data=B_train,
            include_bone=args.use_two_stream, joint_spec_name=args.joint_spec_name
        )
        val_ds = PennActionDataset(
            X_val, y_val, bone_data=B_val,
            include_bone=args.use_two_stream, joint_spec_name=args.joint_spec_name
        )

    sampler = make_weighted_sampler(train_ds) if args.use_weighted_sampler else None
    shuffle = (sampler is None)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        drop_last=False,
        num_workers=effective_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=effective_workers,
        pin_memory=torch.cuda.is_available(),
    )
    print(f'DataLoader num_workers={effective_workers}  two_stream={args.use_two_stream}')

    model = (
        TwoStream_STGCN(num_classes=num_classes, joint_spec=args.joint_spec_name)
        if args.use_two_stream
        else Model_STGCN(num_classes=num_classes, joint_spec=args.joint_spec_name)
    ).to(device)

    def save_periodic_checkpoint(epoch_no: int, model_obj: torch.nn.Module) -> None:
        periodic_name = (
            f'stgcn_gym288_2s_epoch{epoch_no}.pth'
            if args.use_two_stream
            else f'stgcn_gym288_epoch{epoch_no}.pth'
        )
        periodic_path = os.path.join(args.out_dir, periodic_name)
        save_checkpoint(
            periodic_path,
            model_obj,
            metadata={
                'joint_spec_name': args.joint_spec_name,
                'use_two_stream': bool(args.use_two_stream),
                'dataset_format': 'gym288',
                'num_classes': int(num_classes),
                'epoch': int(epoch_no),
                'periodic_checkpoint': True,
            },
        )
        print(f'Saved periodic checkpoint: {periodic_path}')

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=device,
        checkpoint_every=args.save_every_epochs,
        on_checkpoint=save_periodic_checkpoint,
        loss_name=args.loss_name,
        focal_gamma=args.focal_gamma,
        focal_alpha_mode=args.focal_alpha_mode,
        num_classes=num_classes,
        train_labels=y_train,
        grad_clip_norm=args.grad_clip_norm,
    )

    weights_name = 'stgcn_gym288_2s.pth' if args.use_two_stream else 'stgcn_gym288.pth'
    weights_path = os.path.join(args.out_dir, weights_name)
    save_checkpoint(
        weights_path,
        model,
        metadata={
            'joint_spec_name': args.joint_spec_name,
            'use_two_stream': bool(args.use_two_stream),
            'dataset_format': 'gym288',
            'num_classes': int(num_classes),
        },
    )
    print(f'Saved weights: {weights_path}')

    eval_criterion = build_classification_criterion(
        loss_name=args.loss_name,
        device=device,
        focal_gamma=args.focal_gamma,
        focal_alpha=focal_alpha,
    )
    _, _, _, preds, gt = eval_epoch(model, val_loader, eval_criterion, device)

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
