"""
scripts/train_gym99.py
Train ST-GCN on Gym99-skeleton dataset.

Usage
-----
python scripts/train_gym99.py \
    --dataset_path /path/to/gym99_skeleton.pkl \
    --out_dir outputs/gym99 \
    --epochs 30 \
    --batch_size 32 \
    --lr 0.001

Or auto-build Gym99 from Gym288 first (no manual notebook preprocessing):

python scripts/train_gym99.py \
    --auto_build_from_gym288 \
    --gym288_dataset_path /path/to/gym288_skeleton.pkl \
    --dataset_path /path/to/gym99_from_gym288.pkl \
    --out_dir outputs/gym99
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

from src.config import GYM99_NUM_CLASSES
from src.checkpointing import load_checkpoint, save_checkpoint
from src.skeleton_utils import bbox_normalize
from src.dataset import PennActionDataset
from src.experiment_config import apply_overrides, load_experiment_config
from src.gym99_builder import build_gym99_from_gym288_pickle
from src.gym99_dataset import build_gym99_data_tensors, infer_num_gym99_classes
from src.losses import build_classification_criterion, compute_smoothed_alpha
from src.model import Model_STGCN
from src.train import eval_epoch, train_model, train_model_preloaded
from src.two_stream_stgcn import TwoStream_STGCN


def parse_args():
    p = argparse.ArgumentParser(description='Train ST-GCN on Gym99-skeleton')
    p.add_argument('--experiment_config', default='',
                   help='Optional JSON config for frequent experiment updates.')
    p.add_argument('--dataset_path', default='', help='Path to gym99_skeleton.pkl')
    p.add_argument('--gym288_dataset_path', default='',
                   help='Optional Gym288 pickle path used to auto-build Gym99 when needed.')
    p.add_argument('--auto_build_from_gym288', action='store_true',
                   help='Auto-build Gym99 pickle from Gym288 before training.')
    p.add_argument('--gym99_generated_path', default='outputs/datasets/gym99_from_gym288.pkl',
                   help='Output path used when auto-building Gym99 from Gym288.')
    p.add_argument('--out_dir', default='outputs/gym99', help='Output directory')
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
    p.add_argument('--train_data_mode', default='standard', choices=['standard', 'preload_vram'],
                   help='standard: DataLoader on host RAM, preload_vram: preload full train tensors to GPU then train.')
    p.add_argument('--save_every_epochs', type=int, default=10,
                   help='Save periodic checkpoints every N epochs (0 to disable).')
    p.add_argument('--loss_name', default='ce', choices=['ce', 'cross_entropy', 'focal'],
                   help='Classification loss type.')
    p.add_argument('--focal_gamma', type=float, default=2.0,
                   help='Focal Loss gamma parameter.')
    p.add_argument('--focal_alpha_mode', default='none', choices=['none', 'inverse', 'sqrt_inverse'],
                   help='Class alpha weighting mode for focal loss.')
    p.add_argument('--bbox_norm', action='store_true',
                   help='Apply per-sample bounding box normalization to skeleton coordinates before training.')
    p.add_argument('--warmup_epochs', type=int, default=0,
                   help='Linear LR warmup epochs before cosine decay (0 = cosine only).')
    return p.parse_args()


def main():
    args = parse_args()
    if args.experiment_config:
        cfg = load_experiment_config(args.experiment_config)
        args = apply_overrides(args, cfg, sys.argv[1:])

    should_auto_build = args.auto_build_from_gym288 or (not args.dataset_path and bool(args.gym288_dataset_path))
    if should_auto_build:
        if not args.gym288_dataset_path:
            raise ValueError('auto_build_from_gym288 requires --gym288_dataset_path (or config key gym288_dataset_path).')
        out_path = args.dataset_path if args.dataset_path else args.gym99_generated_path
        print('Building Gym99-from-Gym288 pickle...')
        stats = build_gym99_from_gym288_pickle(
            gym288_dataset_path=args.gym288_dataset_path,
            gym99_dataset_path=out_path,
        )
        print(
            'Gym99 mapping stats:',
            f"direct={stats['matched_direct']}",
            f"minus1={stats['matched_minus1']}",
            f"plus1={stats['matched_plus1']}",
            f"train={stats['train_count']}",
            f"test={stats['test_count']}",
        )
        if stats['train_count'] == 0 or stats['test_count'] == 0:
            raise RuntimeError('Gym99 split is empty after mapping; abort before training.')
        args.dataset_path = out_path

    if not args.dataset_path:
        raise ValueError(
            'Missing dataset path. Provide --dataset_path, or use --auto_build_from_gym288 with --gym288_dataset_path.'
        )
    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(f'Gym99 dataset not found: {args.dataset_path}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out_dir, exist_ok=True)

    print(f'Device: {device}')
    if args.dataset_path.startswith('/content/drive'):
        print('[warning] Dataset is on Google Drive path; copy to /content for faster I/O.')
    print('Loading Gym99-skeleton dataset...')
    if args.use_two_stream:
        data, bone_data, labels, flags, _, _ = build_gym99_data_tensors(
            dataset_path=args.dataset_path,
            joint_spec_name=args.joint_spec_name,
            split='all',
            keep_unknown_split=False,
            return_bone_data=True,
        )
    else:
        data, labels, flags, _, _ = build_gym99_data_tensors(
            dataset_path=args.dataset_path,
            joint_spec_name=args.joint_spec_name,
            split='all',
            keep_unknown_split=False,
        )

    train_mask = flags == 1
    test_mask = flags == 0
    if train_mask.sum() == 0 or test_mask.sum() == 0:
        raise RuntimeError('Gym99 split is empty for train/test. Check dataset pickle structure.')

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

    if args.bbox_norm:
        print('[info] Applying bbox normalization to train and val tensors...')
        X_train = bbox_normalize(X_train)
        X_val   = bbox_normalize(X_val)
        if B_train is not None:
            B_train = bbox_normalize(B_train)
        if B_val is not None:
            B_val = bbox_normalize(B_val)

    inferred_classes = infer_num_gym99_classes(args.dataset_path, fallback=GYM99_NUM_CLASSES)
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

    train_loader = DataLoader(
        PennActionDataset(
            X_train,
            y_train,
            bone_data=B_train,
            include_bone=args.use_two_stream,
            joint_spec_name=args.joint_spec_name,
        ),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=effective_workers,
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
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=effective_workers,
        pin_memory=torch.cuda.is_available(),
    )
    print(
        f'DataLoader num_workers={effective_workers}  '
        f'two_stream={args.use_two_stream}  train_data_mode={args.train_data_mode}'
    )

    model = (
        TwoStream_STGCN(num_classes=num_classes, joint_spec=args.joint_spec_name)
        if args.use_two_stream
        else Model_STGCN(num_classes=num_classes, joint_spec=args.joint_spec_name)
    ).to(device)

    if torch.cuda.device_count() > 1:
        print(f'[info] Using {torch.cuda.device_count()} GPUs via DataParallel')
        model = torch.nn.DataParallel(model)

    def _unwrap(m: torch.nn.Module) -> torch.nn.Module:
        """Return the underlying module, stripping DataParallel if present."""
        return m.module if isinstance(m, torch.nn.DataParallel) else m

    def save_periodic_checkpoint(epoch_no: int, model_obj: torch.nn.Module) -> None:
        periodic_name = (
            f'stgcn_gym99_coco18_2s_epoch{epoch_no}.pth'
            if args.use_two_stream
            else f'stgcn_gym99_coco18_epoch{epoch_no}.pth'
        )
        periodic_path = os.path.join(args.out_dir, periodic_name)
        save_checkpoint(
            periodic_path,
            _unwrap(model_obj),
            metadata={
                'joint_spec_name': args.joint_spec_name,
                'use_two_stream': bool(args.use_two_stream),
                'dataset_format': 'gym99',
                'num_classes': int(num_classes),
                'epoch': int(epoch_no),
                'periodic_checkpoint': True,
            },
        )
        print(f'Saved periodic checkpoint: {periodic_path}')

    history = None
    if args.train_data_mode == 'preload_vram':
        if device.type != 'cuda':
            print('[warning] preload_vram requested but CUDA is unavailable. Falling back to standard mode.')
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
                warmup_epochs=args.warmup_epochs,
            )
        else:
            print('[info] Preloading full train tensors to VRAM...')
            train_joint_data = torch.as_tensor(X_train, dtype=torch.float32, device=device)
            train_labels = torch.as_tensor(y_train, dtype=torch.long, device=device)
            train_bone_data = (
                torch.as_tensor(B_train, dtype=torch.float32, device=device)
                if (args.use_two_stream and B_train is not None)
                else None
            )
            history = train_model_preloaded(
                model=model,
                train_joint_data=train_joint_data,
                train_labels=train_labels,
                train_bone_data=train_bone_data,
                val_loader=val_loader,
                num_epochs=args.epochs,
                lr=args.lr,
                weight_decay=args.weight_decay,
                device=device,
                batch_size=args.batch_size,
                checkpoint_every=args.save_every_epochs,
                on_checkpoint=save_periodic_checkpoint,
                loss_name=args.loss_name,
                focal_gamma=args.focal_gamma,
                focal_alpha_mode=args.focal_alpha_mode,
                num_classes=num_classes,
                warmup_epochs=args.warmup_epochs,
            )
    else:
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
            warmup_epochs=args.warmup_epochs,
        )

    weights_name = 'stgcn_gym99_coco18_2s.pth' if args.use_two_stream else 'stgcn_gym99_coco18.pth'
    weights_path = os.path.join(args.out_dir, weights_name)
    save_checkpoint(
        weights_path,
        _unwrap(model),
        metadata={
            'joint_spec_name': args.joint_spec_name,
            'use_two_stream': bool(args.use_two_stream),
            'dataset_format': 'gym99',
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
    with open(os.path.join(args.out_dir, 'metrics_train_gym99.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    with open(os.path.join(args.out_dir, 'classification_report.txt'), 'w', encoding='utf-8') as f:
        f.write(classification_report(gt, preds, zero_division=0))

    with open(os.path.join(args.out_dir, 'history.json'), 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)


if __name__ == '__main__':
    main()
