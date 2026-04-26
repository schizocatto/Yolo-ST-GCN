# Project Overview

## Goal

This project improves the classic ST-GCN (Spatial-Temporal Graph Convolutional Network) for skeleton-based action recognition on the **FineGym** dataset, specifically the **Gym99 FX (Floor Exercise) subset**. The work is submitted as a course final project (Y3-K2, TGMT).

The original ST-GCN was designed and evaluated on coarse-grained datasets (NTU RGB+D, Kinetics). Here we adapt and improve it for **fine-grained gymnastic action recognition**, where classes are highly similar movements (e.g., different types of leaps, turns, saltos) and class imbalance is severe.

---

## Pipeline at a Glance

```
Video
  ↓
YOLOv8-pose  ──→  2D keypoints per frame (COCO17)
  ↓
IoU-based person tracking  ──→  single-person skeleton sequence
  ↓
Preprocessing (normalize, align to 48 frames)
  ↓
ST-GCN (single- or two-stream)  ──→  class logits
  ↓
Predicted action class
```

For offline training, the skeleton sequences are pre-extracted and stored in `.pkl` files (Gym288/Gym99 format). The YOLO stage is only needed for inference on raw video.

---

## Key Improvements Over Classic ST-GCN

| Feature | Classic ST-GCN | This Project |
|---------|---------------|--------------|
| Dataset | NTU RGB+D (60 coarse classes) | FineGym99 (99 fine-grained gymnastics classes) |
| Joint spec | OpenPose 18 | Penn14 or COCO18 (registry-based, switchable) |
| Bone stream | Not used | Optional second stream on bone vectors |
| Fusion | — | Learnable sigmoid gate blending joint + bone logits |
| Class imbalance | Not addressed | FocalLoss + tier-based augmentation + WeightedRandomSampler |
| Augmentation | None | Temporal (shift, reverse, subsample) + spatial (flip, scale, noise) |
| Config system | Hardcoded | JSON experiment configs + CLI override |

---

## Directory Structure

```
Yolo-ST-GCN/
├── src/                    # Core library (models, datasets, training utilities)
│   ├── config.py           # Global constants (joint layouts, class names)
│   ├── joint_specs.py      # Joint specification registry (Penn14, COCO18)
│   ├── graph.py            # Spatial graph construction (3-partition adjacency)
│   ├── model.py            # ST-GCN block and full model
│   ├── two_stream_stgcn.py # Two-stream late-fusion model
│   ├── train.py            # Training loop utilities
│   ├── dataset.py          # Unified dataset dispatcher
│   ├── penn_dataset.py     # Penn Action .mat loader
│   ├── gym288_dataset.py   # FineGym288 pickle loader
│   ├── gym99_dataset.py    # FineGym99 pickle loader
│   ├── gym99_builder.py    # Build Gym99 from Gym288
│   ├── feeder.py           # Class-imbalance-aware augmentation feeder
│   ├── augmentation.py     # Skeleton augmentation primitives
│   ├── skeleton_utils.py   # Keypoint preprocessing utilities
│   ├── losses.py           # FocalLoss, CrossEntropy, loss factory
│   ├── checkpointing.py    # Metadata-aware checkpoint save/load
│   ├── experiment_config.py# JSON config loading + CLI override
│   ├── inference.py        # YOLO + ST-GCN inference pipeline
│   └── visualize.py        # Plotting utilities
├── scripts/                # Runnable training and inference CLI scripts
│   ├── train_gym99.py      # Main training script for Gym99
│   ├── train_gym288.py     # Training script for full Gym288
│   ├── train.py            # Legacy Penn Action training
│   ├── inference_demo.py   # Single-video demo
│   ├── inference_gym99.py  # Gym99 batch inference
│   ├── evaluate.py         # Full pipeline evaluation
│   └── build_gym99_from_gym288.py  # Gym99 dataset builder
├── configs/experiments/    # JSON experiment configuration files
├── notebooks/              # Jupyter notebooks (EDA, training, inference demos)
├── docs/                   # This documentation
├── tests/                  # Pytest smoke tests
├── requirements.txt        # Python dependencies
└── README.md
```

---

## Datasets Used

| Dataset | Classes | Clips | Notes |
|---------|---------|-------|-------|
| FineGym288 | 288 | ~38,223 | Full gymnastic benchmark (all apparatus) |
| FineGym99 | 99 | subset | Derived from Gym288; the main training target |
| Penn Action | 8 | ~2,326 | Exercise classes; used for baseline experiments |

**Focus**: FineGym99 FX (Floor Exercise) subset — the submitted report evaluates models on this split.

---

## Branches

| Branch | Purpose |
|--------|---------|
| `main` | Stable version |
| `duy` | COCO18 support, Gym99 pipeline |
| `experiment-bonestream` | Current — bone-stream architecture experiments |
| `refactor-1` | Archived refactoring attempt |

---

## Submission Artifacts

The final submission consists of Jupyter notebooks in `notebooks/` with training results, evaluation metrics, and analysis. See `notebooks/stgcn-gym99.ipynb` for the main result notebook.
