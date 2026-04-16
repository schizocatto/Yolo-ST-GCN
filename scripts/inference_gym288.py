"""
scripts/inference_gym288.py
Evaluate ST-GCN inference on Gym288-skeleton test split.

Usage
-----
python scripts/inference_gym288.py \
    --dataset_path /path/to/gym288_skeleton.pkl \
    --weights outputs/gym288/stgcn_gym288.pth \
    --out_dir outputs/gym288
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

# Allow running from repo root without installing the package
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.config import GYM288_NUM_CLASSES
from src.checkpointing import load_checkpoint
from src.dataset import PennActionDataset
from src.experiment_config import apply_overrides, load_experiment_config
from src.gym288_dataset import build_gym288_data_tensors, infer_num_gym288_classes
from src.losses import build_classification_criterion, compute_smoothed_alpha
from src.model import Model_STGCN
from src.two_stream_stgcn import TwoStream_STGCN


def parse_args():
    p = argparse.ArgumentParser(description='Inference/Evaluation on Gym288-skeleton test split')
    p.add_argument('--experiment_config', default='',
                   help='Optional JSON config for frequent experiment updates.')
    p.add_argument('--dataset_path', required=True, help='Path to gym288_skeleton.pkl')
    p.add_argument('--weights', required=True, help='Path to trained ST-GCN weights (.pth)')
    p.add_argument('--out_dir', default='outputs/gym288', help='Output directory')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--num_classes', type=int, default=0,
                   help='Override class count. 0 = infer from dataset labels')
    p.add_argument('--joint_spec_name', default='penn14', choices=['penn14', 'coco18'],
                   help='Target joint layout for model input.')
    p.add_argument('--topk', type=int, default=5)
    p.add_argument('--num_workers', '--num_wokers', dest='num_workers', type=int, default=0,
                   help='Number of DataLoader workers (supports alias --num_wokers).')
    p.add_argument('--use_two_stream', action='store_true',
                   help='Enable 2s-STGCN inference (requires two-stream-trained weights).')
    p.add_argument('--loss_name', default='ce', choices=['ce', 'cross_entropy', 'focal'],
                   help='Classification loss used for reported loss value.')
    p.add_argument('--focal_gamma', type=float, default=2.0,
                   help='Focal Loss gamma parameter.')
    p.add_argument('--focal_alpha_mode', default='none', choices=['none', 'inverse', 'sqrt_inverse'],
                   help='Class alpha weighting mode for focal loss.')
    return p.parse_args()


def _evaluate_topk(model, loader, criterion, device, topk: int = 5):
    model.eval()
    all_logits = []
    all_labels = []
    total_loss = 0.0

    with torch.no_grad():
        for batch_data, batch_labels in tqdm(loader, desc='Inference [test]', leave=False):
            if isinstance(batch_data, (tuple, list)) and len(batch_data) == 2:
                joint_data = batch_data[0].to(device)
                bone_data = batch_data[1].to(device)
            else:
                joint_data = batch_data.to(device)
                bone_data = None
            batch_labels = batch_labels.to(device)

            logits = model(joint_data, bone_data) if bone_data is not None else model(joint_data)
            loss = criterion(logits, batch_labels)
            total_loss += loss.item()

            all_logits.append(logits.cpu().numpy())
            all_labels.append(batch_labels.cpu().numpy())

    if not all_labels:
        return 0.0, 0.0, 0.0, [], []

    logits_np = np.concatenate(all_logits, axis=0)
    labels_np = np.concatenate(all_labels, axis=0)

    preds_top1 = np.argmax(logits_np, axis=1)
    top1 = accuracy_score(labels_np, preds_top1)
    macro_f1 = f1_score(labels_np, preds_top1, average='macro', zero_division=0)

    k = max(1, int(topk))
    topk_idx = np.argpartition(logits_np, -k, axis=1)[:, -k:]
    topk_hit = np.array([labels_np[i] in topk_idx[i] for i in range(len(labels_np))], dtype=np.float32)
    topk_acc = float(topk_hit.mean()) if len(topk_hit) > 0 else 0.0

    avg_loss = total_loss / len(loader)
    return avg_loss, float(top1), float(topk_acc), float(macro_f1), preds_top1.tolist(), labels_np.tolist()


def main():
    args = parse_args()
    if args.experiment_config:
        cfg = load_experiment_config(args.experiment_config)
        args = apply_overrides(args, cfg, sys.argv[1:])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out_dir, exist_ok=True)

    if args.use_two_stream:
        data, bone_data, labels, flags, _, video_ids = build_gym288_data_tensors(
            dataset_path=args.dataset_path,
            joint_spec_name=args.joint_spec_name,
            split='test',
            keep_unknown_split=False,
            return_bone_data=True,
        )
    else:
        data, labels, flags, _, video_ids = build_gym288_data_tensors(
            dataset_path=args.dataset_path,
            joint_spec_name=args.joint_spec_name,
            split='test',
            keep_unknown_split=False,
        )
    if len(data) == 0:
        raise RuntimeError('No test samples found in Gym288 dataset.')

    inferred_classes = infer_num_gym288_classes(args.dataset_path, fallback=GYM288_NUM_CLASSES)
    num_classes = args.num_classes if args.num_classes > 0 else inferred_classes

    focal_alpha = None
    if args.loss_name == 'focal' and args.focal_alpha_mode != 'none':
        focal_alpha = compute_smoothed_alpha(labels, num_classes=num_classes, mode=args.focal_alpha_mode)

    model = (
        TwoStream_STGCN(num_classes=num_classes, joint_spec=args.joint_spec_name)
        if args.use_two_stream
        else Model_STGCN(num_classes=num_classes, joint_spec=args.joint_spec_name)
    ).to(device)
    state_dict, ckpt_meta = load_checkpoint(args.weights, map_location=device)
    ckpt_spec = ckpt_meta.get('joint_spec_name', '')
    if ckpt_spec and ckpt_spec != args.joint_spec_name:
        print(
            f"[warning] Checkpoint joint_spec={ckpt_spec} but inference joint_spec={args.joint_spec_name}. "
            "This may fail or degrade results."
        )
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    loader = DataLoader(
        PennActionDataset(
            data,
            labels,
            bone_data=bone_data if args.use_two_stream else None,
            include_bone=args.use_two_stream,
            joint_spec_name=args.joint_spec_name,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    print(f'DataLoader num_workers={args.num_workers}  two_stream={args.use_two_stream}')

    criterion = build_classification_criterion(
        loss_name=args.loss_name,
        device=device,
        focal_gamma=args.focal_gamma,
        focal_alpha=focal_alpha,
    )

    loss, top1, topk_acc, macro_f1, preds, gt = _evaluate_topk(
        model,
        loader,
        criterion,
        device,
        topk=args.topk,
    )

    print(f'Test samples    : {len(data)}')
    print(f'Loss            : {loss:.4f}')
    print(f'Top-1 accuracy  : {top1:.4f}')
    print(f'Top-{args.topk} accuracy: {topk_acc:.4f}')
    print(f'Macro-F1        : {macro_f1:.4f}')

    metrics = {
        'num_test': int(len(data)),
        'num_classes': int(num_classes),
        'loss': float(loss),
        'top1_accuracy': float(top1),
        f'top{args.topk}_accuracy': float(topk_acc),
        'macro_f1': float(macro_f1),
    }
    with open(os.path.join(args.out_dir, 'metrics_inference_gym288.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    preds_dump = [
        {'video_id': vid, 'gt': int(t), 'pred_top1': int(p)}
        for vid, t, p in zip(video_ids, gt, preds)
    ]
    with open(os.path.join(args.out_dir, 'predictions_test_top1.json'), 'w', encoding='utf-8') as f:
        json.dump(preds_dump, f, indent=2)


if __name__ == '__main__':
    main()
