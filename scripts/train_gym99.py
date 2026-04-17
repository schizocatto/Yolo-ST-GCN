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
from src.skeleton_utils import bbox_normalize, center_normalize
from src.dataset import PennActionDataset
from src.experiment_config import apply_overrides, load_experiment_config
from src.gym99_builder import build_gym99_from_gym288_pickle
from src.gym99_dataset import build_gym99_data_tensors, infer_num_gym99_classes
from src.losses import build_classification_criterion, compute_smoothed_alpha
from src.model import Model_STGCN
from src.train import eval_epoch, train_model, train_model_preloaded
from src.two_stream_stgcn import TwoStream_STGCN
from src.feeder import SkeletonFeeder, build_feeder_pair, make_weighted_sampler


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
    p.add_argument('--loss_name', default='focal', choices=['ce', 'focal', 'dice'],
                   help='Loss function (default: focal)')
    p.add_argument('--focal_gamma', type=float, default=2.0,
                   help='Focal Loss gamma parameter.')
    p.add_argument('--focal_alpha_mode', default='sqrt_inverse', choices=['uniform', 'inverse', 'sqrt_inverse'],
                   help='Alpha weighting strategy for Focal Loss (default: sqrt_inverse)')
    p.add_argument('--bbox_norm', action='store_true',
                   help='Normalize skeleton bounding boxes to [0,1] over the entire video')
    p.add_argument('--center_norm', action='store_true',
                   help='(ST-GCN original) Move origin (0,0) to center joint frame-by-frame')
    p.add_argument('--warmup_epochs', type=int, default=0,
                   help='Linear LR warmup epochs before cosine decay (0 = cosine only).')
    p.add_argument('--optimizer', default='adam', choices=['adam', 'adamw', 'sgd'],
                   help='Optimizer: adam | adamw | sgd.')
    p.add_argument('--sgd_momentum', type=float, default=0.9,
                   help='Momentum for SGD optimizer.')
    p.add_argument('--sgd_nesterov', action='store_true', default=True,
                   help='Enable Nesterov momentum for SGD (default: True).')
    p.add_argument('--grad_clip_norm', type=float, default=1.0,
                   help='Max gradient L2-norm for clipping (applied after backward, before optimizer step). '
                        'Set to 0 or negative to disable gradient clipping.')
    p.add_argument('--use_augment_feeder', action='store_true',
                   help='Use the smart class-imbalance aware SkeletonFeeder instead of basic dataset.')
    p.add_argument('--use_weighted_sampler', action='store_true',
                   help='Use WeightedRandomSampler to oversample minority classes. Useful with --use_augment_feeder.')
    p.add_argument('--aug_config_path', default='', type=str,
                   help='Path to a JSON file containing custom augmentation policy overwriting SkeletonFeeder defaults.')
    p.add_argument('--oversample_ratio', type=float, default=1.0,
                   help='Multiplier for samples drawn per epoch. E.g. 2.0 generates twice the augmented variants.')
    p.add_argument('--apparatus', default='all', choices=['all', 'VT', 'FX', 'BB', 'UB'],
                   help=(
                       'Filter dataset to a single apparatus for Expert training. '
                       'Labels are remapped to local indices [0, N-1]. '
                       'VT=0-5 (6 cls), FX=6-40 (35 cls), BB=41-73 (33 cls), UB=74-98 (25 cls), all=99 cls.'
                   ))
    return p.parse_args()


def main():
    args = parse_args()
    if args.experiment_config:
        cfg = load_experiment_config(args.experiment_config)
        args = apply_overrides(args, cfg, sys.argv[1:])

    should_auto_build = args.auto_build_from_gym288 or (not args.dataset_path and bool(args.gym288_dataset_path))
    if args.use_augment_feeder and args.train_data_mode == 'preload_vram':
        # Safety check: preload_vram bypasses the DataLoader __getitem__, so augmentation wouldn't work.
        print('[warning] Cannot use preload_vram with use_augment_feeder. Falling back to standard mode.')
        args.train_data_mode = 'standard'
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

    # ── Apparatus filtering for Expert training ──────────────────────────────
    APPARATUS_RANGES = {
        'VT': (0,   5),
        'FX': (6,  40),
        'BB': (41, 73),
        'UB': (74, 98),
    }
    apparatus_label_offset = 0  # used for weight filename tagging
    if args.apparatus != 'all':
        lo, hi = APPARATUS_RANGES[args.apparatus]
        apparatus_label_offset = lo

        def _filter(X, y, B):
            mask = (y >= lo) & (y <= hi)
            X_f = X[mask]
            y_f = y[mask] - lo  # remap to [0, hi-lo]
            B_f = B[mask] if B is not None else None
            return X_f, y_f, B_f

        X_train, y_train, B_train = _filter(X_train, y_train, B_train)
        X_val,   y_val,   B_val   = _filter(X_val,   y_val,   B_val)
        print(
            f'[apparatus={args.apparatus}] class range [{lo}, {hi}] → '
            f'local classes [0, {hi - lo}]  '
            f'train={len(X_train)}  val={len(X_val)}'
        )
        if len(X_train) == 0 or len(X_val) == 0:
            raise RuntimeError(f'Apparatus filter "{args.apparatus}" produced empty split. Check dataset labels.')
    # ─────────────────────────────────────────────────────────────────────────

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

    if args.center_norm:
        from src.config import JOINT_NAMES, COCO17_JOINT_NAMES
        # Automatically detect center joint index based on joint_spec
        # Penn14: center* is index 13
        # COCO18: center* is index 17
        center_idx = 17 if 'coco' in args.joint_spec_name else 13
        print(f'[info] Applying per-frame center normalization (center joint = {center_idx})...')
        X_train = center_normalize(X_train, center_idx)
        X_val   = center_normalize(X_val, center_idx)
        # We do NOT center normalize bone data because bone data is relative (child - parent)
        # Adding/subtracting an offset to/from both child and parent cancels out


    if args.apparatus != 'all':
        lo, hi = APPARATUS_RANGES[args.apparatus]
        num_classes = hi - lo + 1
        print(f'num_classes={num_classes} (apparatus={args.apparatus}, local labels 0-{num_classes - 1})')
    else:
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

    if args.use_augment_feeder:
        flip_pairs = (
            SkeletonFeeder.PENN14_FLIP_PAIRS if args.joint_spec_name == 'penn14'
            else None  # Adjust if COCO18 flip pairs are needed
        )
        custom_policy = None
        if args.aug_config_path:
            with open(args.aug_config_path, 'r') as f:
                raw_policy = json.load(f)
                custom_policy = {int(k): v for k, v in raw_policy.items()}
            print(f'[info] Loaded custom augmentation policy from: {args.aug_config_path}')

        train_ds, val_ds = build_feeder_pair(
            train_data=X_train,
            train_labels=y_train,
            val_data=X_val,
            val_labels=y_val,
            train_bone=B_train,
            val_bone=B_val,
            include_bone=args.use_two_stream,
            flip_pairs=flip_pairs,
            custom_policy=custom_policy,
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

    sampler = None
    target_samples = int(len(train_ds) * args.oversample_ratio)
    if args.use_weighted_sampler:
        sampler = make_weighted_sampler(train_ds, num_samples=target_samples)
    elif args.oversample_ratio != 1.0:
        # Fallback to standard random sampler with replacement if weighted is not requested
        from torch.utils.data import RandomSampler
        sampler = RandomSampler(train_ds, replacement=True, num_samples=target_samples)

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

    print(
        f'DataLoader num_workers={effective_workers}  '
        f'two_stream={args.use_two_stream}  train_data_mode={args.train_data_mode}'
    )

    model = (
        TwoStream_STGCN(num_classes=num_classes, joint_spec=args.joint_spec_name)
        if args.use_two_stream
        else Model_STGCN(num_classes=num_classes, joint_spec=args.joint_spec_name)
    ).to(device)

    def save_periodic_checkpoint(epoch_no: int, model_obj: torch.nn.Module) -> None:
        apparatus_suffix = f'_expert_{args.apparatus}' if args.apparatus != 'all' else ''
        periodic_name = (
            f'stgcn_gym99_coco18_2s{apparatus_suffix}_epoch{epoch_no}.pth'
            if args.use_two_stream
            else f'stgcn_gym99_coco18{apparatus_suffix}_epoch{epoch_no}.pth'
        )
        periodic_path = os.path.join(args.out_dir, periodic_name)
        save_checkpoint(
            periodic_path,
            model_obj,
            metadata={
                'joint_spec_name': args.joint_spec_name,
                'use_two_stream': bool(args.use_two_stream),
                'dataset_format': 'gym99',
                'num_classes': int(num_classes),
                'apparatus': args.apparatus,
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
                optimizer_name=args.optimizer,
                sgd_momentum=args.sgd_momentum,
                sgd_nesterov=args.sgd_nesterov,
                grad_clip_norm=args.grad_clip_norm,
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
                optimizer_name=args.optimizer,
                sgd_momentum=args.sgd_momentum,
                sgd_nesterov=args.sgd_nesterov,
                grad_clip_norm=args.grad_clip_norm,
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
            optimizer_name=args.optimizer,
            sgd_momentum=args.sgd_momentum,
            sgd_nesterov=args.sgd_nesterov,
            grad_clip_norm=args.grad_clip_norm,
        )

    apparatus_suffix = f'_expert_{args.apparatus}' if args.apparatus != 'all' else ''
    weights_name = (
        f'stgcn_gym99_coco18_2s{apparatus_suffix}.pth'
        if args.use_two_stream
        else f'stgcn_gym99_coco18{apparatus_suffix}.pth'
    )
    weights_path = os.path.join(args.out_dir, weights_name)
    save_checkpoint(
        weights_path,
        model,
        metadata={
            'joint_spec_name': args.joint_spec_name,
            'use_two_stream': bool(args.use_two_stream),
            'dataset_format': 'gym99',
            'num_classes': int(num_classes),
            'apparatus': args.apparatus,
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
