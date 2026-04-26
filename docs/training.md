# Training Guide

## Quick Start

**Train single-stream ST-GCN on Gym99 with COCO18 joints:**
```bash
python scripts/train_gym99.py \
  --dataset_path /path/to/gym99_skeleton.pkl \
  --out_dir outputs/gym99_coco18 \
  --joint_spec_name coco18 \
  --epochs 30 \
  --batch_size 256
```

**Train two-stream model via JSON config:**
```bash
python scripts/train_gym99.py \
  --dataset_path /path/to/gym99_skeleton.pkl \
  --experiment_config configs/experiments/gym99_coco18_2s.json
```

**Smoke test (no dataset needed):**
```bash
python scripts/train_gym99.py \
  --dataset_path /path/to/gym99_skeleton.pkl \
  --max_train_samples 100 \
  --max_val_samples 20 \
  --epochs 2
```

---

## Configuration System

### Priority (highest to lowest)

1. **CLI arguments**: `--batch_size 128` overrides everything
2. **JSON experiment config**: `--experiment_config configs/experiments/gym99_coco18_2s.json`
3. **Script defaults**: hardcoded values in the training script

Use JSON configs for reproducible experiments. CLI args for one-off overrides.

### JSON Config Format

```json
{
  "joint_spec_name": "coco18",
  "use_two_stream": true,
  "epochs": 30,
  "batch_size": 256,
  "lr": 0.001,
  "weight_decay": 0.0001,
  "num_workers": 2,
  "loss_name": "focal",
  "focal_gamma": 2.0,
  "focal_alpha_mode": "sqrt_inverse",
  "dropout": 0.3,
  "edge_importance_weighting": true,
  "warmup_epochs": 5
}
```

Config files live in `configs/experiments/`. Create a new `.json` file there for each experiment variant.

---

## CLI Arguments Reference

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset_path` | required | Path to `.pkl` dataset file |
| `--out_dir` | `outputs/` | Directory for checkpoints and logs |
| `--joint_spec_name` | `penn14` | `penn14` or `coco18` |
| `--use_two_stream` | False | Enable joint + bone two-stream model |
| `--epochs` | 30 | Total training epochs |
| `--batch_size` | 256 | Mini-batch size |
| `--lr` | 0.001 | Initial learning rate |
| `--weight_decay` | 1e-4 | AdamW weight decay |
| `--dropout` | 0.3 | Dropout rate inside STGCN blocks |
| `--loss_name` | `focal` | `ce`, `focal`, or `dice` |
| `--focal_gamma` | 2.0 | Focal loss gamma parameter |
| `--focal_alpha_mode` | `sqrt_inverse` | Class weight mode: `uniform`, `inverse`, `sqrt_inverse` |
| `--warmup_epochs` | 5 | Linear LR warmup before cosine decay |
| `--num_workers` | 2 | DataLoader workers |
| `--max_train_samples` | None | Limit training set size (for smoke tests) |
| `--max_val_samples` | None | Limit validation set size |
| `--experiment_config` | None | Path to JSON config |

---

## Training Pipeline

### Step-by-step

```
1. Load dataset
   build_data_tensors(format, joint_spec, path, split='train', return_bone=use_two_stream)

2. Build augmentation feeder
   SkeletonFeeder(data, labels, bone_data)
   make_weighted_sampler(labels)  → for balanced mini-batches

3. Create DataLoader
   DataLoader(feeder, batch_size, sampler=weighted_sampler, num_workers)

4. Build model
   Model_STGCN(num_classes, joint_spec_name, ...)
   or TwoStream_STGCN(num_classes, joint_spec_name, ...)

5. Build loss
   build_classification_criterion(loss_name, labels, focal_gamma, focal_alpha_mode)

6. Build optimizer
   AdamW(model.parameters(), lr, weight_decay)

7. Build scheduler
   CosineAnnealingLR with linear warmup over warmup_epochs

8. Train
   train_model(model, train_loader, val_loader, optimizer, criterion,
               scheduler, device, epochs, out_dir, warmup_epochs)
```

### Checkpoints

`train_model` saves:
- `best_model.pth` — highest validation accuracy seen so far
- `checkpoint_epoch_N.pth` — periodic saves (every `save_every_n_epochs`)

Each checkpoint file contains:
```python
{
  'model_state_dict': ...,
  'metadata': {
    'joint_spec_name': 'coco18',
    'use_two_stream': True,
    'num_classes': 99,
    'epoch': 22,
    'val_acc': 0.73,
  }
}
```

---

## Class Imbalance Handling

Gym99 has severe class imbalance. Use both mechanisms together for best results.

### FocalLoss

Reduces gradient contribution from easy (well-classified majority) examples:
```
FL(p) = -α_t · (1 - p_t)^γ · log(p_t)
```
- `γ = 2.0` is standard; increase to 3.0–5.0 for very imbalanced data
- `focal_alpha_mode='sqrt_inverse'` sets `α_t ∝ 1/√freq(class_t)` — a softer downweighting than inverse frequency

### WeightedRandomSampler

Ensures each mini-batch sees a roughly uniform class distribution by oversampling rare classes. Enable by passing the sampler from `make_weighted_sampler` to `DataLoader`.

### SkeletonFeeder Augmentation Tiers

Rare classes automatically receive stronger augmentation to increase the effective number of training samples:

| Class frequency percentile | Tier | Augmentations applied |
|---------------------------|------|----------------------|
| > 75th | 0 | Gaussian noise |
| 25th–75th | 1 | Temporal shift + noise |
| 10th–25th | 2 | Spatial move + flip + noise |
| < 10th | 3 | Move + flip + scale + noise + time-reverse |

---

## Learning Rate Schedule

Default: **linear warmup** for `warmup_epochs`, then **cosine annealing** to `lr_min=1e-6`.

```
epoch 0 → warmup_epochs:  lr increases linearly from 0 to lr
epoch warmup_epochs → N:  lr decreases as cosine curve
```

For fine-tuning from a checkpoint, consider starting with a lower `lr` and fewer `warmup_epochs`.

---

## Recommended Experiment Configurations

### Baseline (fast, good for debugging)
```json
{
  "joint_spec_name": "penn14",
  "use_two_stream": false,
  "epochs": 20,
  "batch_size": 128,
  "lr": 0.001,
  "loss_name": "focal"
}
```

### Main submission configuration
```json
{
  "joint_spec_name": "coco18",
  "use_two_stream": true,
  "epochs": 30,
  "batch_size": 256,
  "lr": 0.001,
  "weight_decay": 0.0001,
  "dropout": 0.3,
  "loss_name": "focal",
  "focal_gamma": 2.0,
  "focal_alpha_mode": "sqrt_inverse",
  "warmup_epochs": 5
}
```

---

## Monitoring Training

`train_model` prints per-epoch loss and accuracy to stdout. The returned `history` dict can be plotted:

```python
from src.visualize import plot_training_history
history = train_model(...)
plot_training_history(history)
```

---

## Evaluating a Checkpoint

```bash
python scripts/evaluate.py \
  --dataset_path /path/to/gym99_skeleton.pkl \
  --checkpoint outputs/gym99_coco18_2s/best_model.pth \
  --split test
```

Prints per-class accuracy, macro F1, and confusion matrix.
