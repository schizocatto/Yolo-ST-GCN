# Module Reference

Quick reference for every file in `src/`. Each entry covers what the file does, its public API, and key design notes.

---

## src/config.py

**Role**: Global constants — joint layouts, class names, dataset sizes.

**Key contents**:
- `PENN_CLASSES`: list of 8 Penn Action class names
- `PENN_BONE_PAIRS`: 14 edges (13 Penn bones + 4 virtual-center connections)
- `COCO17_JOINT_NAMES`: names for 17 COCO keypoints
- `TARGET_FRAMES = 48`: all sequences are resampled to this length
- `IN_CHANNELS = 2`: (x, y) only, no depth
- `NUM_CLASSES_GYM288 = 288`, `NUM_CLASSES_GYM99 = 99`

**When to edit**: Add a new dataset's class count or joint layout names here first.

---

## src/joint_specs.py

**Role**: Registry of joint layout specifications. The single source of truth for skeleton topology.

**Public API**:
```python
get_joint_spec(name: str) -> dict
# name: 'penn14' | 'coco18'
```

**Spec fields**:
| Field | Description |
|-------|-------------|
| `num_joints` | Total joints (including virtual center) |
| `center_joint` | Index of virtual center joint |
| `bone_pairs` | List of (parent, child) index tuples |
| `coco_to_layout_idx` | Mapping from COCO17 indices to this layout |
| `virtual_center_parents` | Indices averaged to compute the virtual center |
| `has_virtual_center` | Always True for current specs |

**Specs defined**:
- `penn14`: 13 Penn Action joints + 1 virtual center (mean of shoulders+hips)
- `coco18`: 17 COCO keypoints + 1 virtual center

**Design note**: Adding a new joint layout only requires editing this file. All downstream components (Graph, Model, Dataset) accept the spec by name.

---

## src/graph.py

**Role**: Build the spatial graph adjacency tensor used in ST-GCN.

**Public API**:
```python
GraphSkeleton(joint_spec: str)
# .A  → np.ndarray (3, V, V)  — normalized 3-partition adjacency
```

**Adjacency partitions**:
- `A[0]`: Self-loops (each joint connected to itself)
- `A[1]`: Centripetal edges (joints pointing toward virtual center)
- `A[2]`: Centrifugal edges (joints pointing away from virtual center)

All partitions are row-normalized with a small epsilon (`1e-4`) for numerical stability.

**Legacy alias**: `Graph_PennAction_14Nodes` maps to `GraphSkeleton('penn14')`.

---

## src/model.py

**Role**: Single-stream ST-GCN architecture.

**Public API**:
```python
Model_STGCN(
    num_classes: int,
    joint_spec_name: str = 'penn14',   # 'penn14' or 'coco18'
    edge_importance_weighting: bool = True,
    dropout: float = 0.0,
)
# Forward: (N, 2, T, V, 1) → (N, num_classes)
```

**Architecture**:
```
Input (N, 2, T, V, 1)
  ↓ Data BatchNorm
  ↓ 10 × STGCN_Block  [stride-2 at blocks 5 and 8]
  ↓ Global Average Pool → (N, 256)
  ↓ Classifier Conv2D(256 → num_classes, 1×1)
Output (N, num_classes)
```

**STGCN_Block internals**:
- Spatial: graph convolution with 3-partition adjacency
- Temporal: Conv2D with 9×1 kernel (9-frame receptive field)
- Residual: 1×1 conv projection when channels/stride change
- Optional dropout before residual add

---

## src/two_stream_stgcn.py

**Role**: Two-stream late-fusion model (joint stream + bone stream).

**Public API**:
```python
TwoStream_STGCN(
    num_classes: int,
    joint_spec_name: str = 'penn14',
    edge_importance_weighting: bool = True,
    dropout: float = 0.0,
)
# Forward: (joint_data, bone_data) → (N, num_classes)
# Both inputs: (N, 2, T, V, 1)
```

**Fusion mechanism**:
- Both streams use identical `Model_STGCN` weights (not shared)
- A learnable scalar `alpha` (sigmoid-gated) blends the two logit vectors:
  ```
  output = alpha * joint_logits + (1 - alpha) * bone_logits
  ```
- `alpha` starts at 0.5 and is optimized via backprop

**Bone stream input**: Each bone vector = `child_joint_coords − parent_joint_coords` for each bone pair. This is computed in the dataset loader, not the model.

**Variant**: `TwoStream_STGCN_COCO18` is a convenience subclass that defaults `joint_spec_name='coco18'`.

---

## src/train.py

**Role**: Training loop functions.

**Public API**:
```python
train_epoch(model, loader, optimizer, criterion, device, clip_grad=1.0) -> (loss, acc)
eval_epoch(model, loader, criterion, device) -> (loss, acc, f1)
train_model(model, train_loader, val_loader, optimizer, criterion, scheduler, device,
            epochs, out_dir, warmup_epochs, ...) -> history
```

**train_model features**:
- Linear warmup over `warmup_epochs` then cosine decay (or any passed scheduler)
- Saves best checkpoint by validation accuracy
- Periodic checkpoint saves (`save_every_n_epochs`)
- Returns loss/accuracy history dict for plotting

**Two-stream support**: `train_epoch` and `eval_epoch` detect `TwoStream_STGCN` by checking if the loader yields `(joint, bone, label)` tuples.

---

## src/dataset.py

**Role**: Unified dataset dispatcher — routes to the right loader based on `dataset_format`.

**Public API**:
```python
build_data_tensors(
    dataset_format: str,    # 'penn', 'gym288', 'gym99', 'coco'
    joint_spec_name: str,   # 'penn14' or 'coco18'
    dataset_path: str,
    split: str = 'train',   # 'train', 'test', 'all'
    return_bone: bool = False,
    ...
) -> (data, labels, flags, frame_counts, video_ids)
# data: (N, 2, T, V, 1)
# labels: (N,) int64
```

**PennActionDataset**: PyTorch `Dataset` wrapper over the pre-built tensors, used with `DataLoader`.

---

## src/penn_dataset.py

**Role**: Load Penn Action `.mat` files into tensors.

**Key behavior**:
- Reads MATLAB structs: `action`, `nframes`, `train`, `x`, `y` (13-joint coords)
- Uses official `train` flag for split — **never use `sklearn.train_test_split`** (causes data leakage because subjects appear in both splits)
- Adds virtual center joint (mean of shoulders+hips)
- Temporal alignment to 48 frames

---

## src/gym288_dataset.py

**Role**: Load FineGym288 `.pkl` into tensors.

**Key behavior**:
- Reads `data['annotations']` list; filters by `data['split']['train'/'test']`
- Extracts `keypoint` field: shape `(1, T, 17, 2)` → drop person dim → `(T, 17, 2)`
- Falls back to `kp_w_gt` field (has confidence as 3rd channel; only x,y used)
- Remaps COCO17 indices → target joint spec (Penn14 or COCO18)
- Appends virtual center joint
- Temporal alignment to 48 frames
- Returns `(N, 2, T, V, 1)` tensor

---

## src/gym99_dataset.py

**Role**: Load FineGym99 `.pkl` into tensors. Identical structure to `gym288_dataset.py` but for the 99-class subset.

---

## src/gym99_builder.py

**Role**: Build a Gym99 `.pkl` from a Gym288 `.pkl` by re-mapping class labels.

**How it works**:
1. Downloads FineGym category files from `sdolivia.github.io/FineGym/resources/dataset/`
2. Parses `Glabel` (global ID) and `Clabel` fields
3. Matches Gym288 labels to Gym99 global IDs
4. Filters to Gym99 classes, re-indexes labels 0..98
5. Has fallback for off-by-one label mismatches in the source dataset

**CLI**: `scripts/build_gym99_from_gym288.py`

---

## src/feeder.py

**Role**: Class-imbalance-aware augmentation feeder for PyTorch `DataLoader`.

**Public API**:
```python
SkeletonFeeder(data, labels, bone_data=None, augmentation_tier_thresholds=...)
make_weighted_sampler(labels) -> WeightedRandomSampler
build_feeder_pair(data, labels, bone_data, ...) -> (train_feeder, val_feeder)
```

**Augmentation tiers** (by class frequency):
| Tier | Frequency threshold | Augmentations |
|------|-------------------|--------------|
| 0 (majority) | > 75th pct | noise only |
| 1 (moderate) | 25–75th pct | shift + noise |
| 2 (minority) | 10–25th pct | move + flip + noise |
| 3 (rare) | < 10th pct | move + flip + scale + noise + reverse |

Optional intra-class MixUp for Tier 3 samples.

---

## src/augmentation.py

**Role**: Stateless skeleton augmentation primitives.

**Functions**:
| Function | Type | Description |
|----------|------|-------------|
| `random_choose` | temporal | Random contiguous crop then resize |
| `random_shift` | temporal | Circular shift along time axis |
| `temporal_reverse` | temporal | Flip time order |
| `temporal_subsample` | temporal | Keep every Nth frame then tile |
| `random_move` | spatial | Small random affine offset per frame |
| `random_rotate` | spatial | 2D rotation around virtual center |
| `random_scale` | spatial | Scale skeleton by random factor |
| `flip_skeleton` | spatial | Horizontal flip (swap left/right joints) |
| `random_noise` | joint-level | Add Gaussian noise to coordinates |
| `joint_dropout` | joint-level | Zero out random joints |
| `skeleton_mixup` | blending | Linear interpolation between two sequences |
| `apply_augmentation_policy` | dispatch | Calls augmentations based on tier |

All functions operate on `(2, T, V, 1)` numpy arrays and return the same shape.

---

## src/skeleton_utils.py

**Role**: Keypoint preprocessing — normalization, alignment, remapping.

**Key functions**:
```python
bbox_normalize(seq)           # Scale coords to [0,1] per sample
center_normalize(seq)         # Shift origin to virtual center each frame
add_virtual_center_joint(seq) # Append mean(shoulders+hips) as new joint
temporal_align(seq, T=48)     # Resample sequence to T frames
remap_coco17_to_penn13(seq)   # Map COCO17 → Penn13 joint indices
remap_coco17_to_layout(seq, spec) # Map COCO17 → any registered spec
to_stgcn_input_from_coco(seq, spec) # Full pipeline → (2, T, V, 1)
```

**Valid mask**: `center_normalize` skips frames where the virtual center is (0,0) (missing detection) rather than subtracting a zero origin, which would corrupt the data.

---

## src/losses.py

**Role**: Classification losses and class-weight computation.

**Public API**:
```python
FocalLoss(gamma=2.0, alpha=None, reduction='mean')
build_classification_criterion(loss_name, labels=None, focal_gamma=2.0,
                                focal_alpha_mode='sqrt_inverse') -> nn.Module
compute_smoothed_alpha(labels, mode='sqrt_inverse') -> Tensor
```

**Loss names** for `build_classification_criterion`:
- `'ce'`: standard CrossEntropyLoss
- `'focal'`: FocalLoss with optional per-class alpha weighting
- `'dice'`: Dice loss (experimental)

**Alpha modes** for class weighting:
- `'uniform'`: all classes equal weight
- `'inverse'`: weight ∝ 1/frequency
- `'sqrt_inverse'`: weight ∝ 1/√frequency (recommended — less aggressive)

---

## src/checkpointing.py

**Role**: Save and load model checkpoints with metadata.

**Public API**:
```python
save_checkpoint(model, path, metadata: dict = None)
load_checkpoint(model, path) -> metadata: dict
```

**Metadata fields** (recommended to always save):
- `joint_spec_name`: Prevents loading COCO18 weights into a Penn14 model
- `use_two_stream`: Documents whether checkpoint is single or two-stream
- `num_classes`: Number of output classes
- `epoch`, `val_acc`, `val_loss`: Training state

---

## src/experiment_config.py

**Role**: Load JSON experiment configs and merge with CLI arguments.

**Public API**:
```python
load_experiment_config(path: str) -> dict
apply_overrides(args, config: dict) -> args
```

**Priority** (highest to lowest):
1. CLI arguments
2. JSON config values
3. Script defaults

`apply_overrides` only sets a config value if the CLI arg is still at its default, so explicit CLI flags always win.

---

## src/inference.py

**Role**: YOLO + ST-GCN inference pipeline for raw video input.

**Public API**:
```python
extract_yolo_keypoints(video_path, yolo_model, joint_spec_name) -> (T, V, 2)
run_stgcn_inference(keypoints, model, device) -> (class_idx, confidence)
load_stgcn_weights(model, checkpoint_path) -> metadata
```

**Person tracking**:
- Frame 0: select person with largest bounding box
- Frames 1+: match to previous frame via IoU (threshold 0.3)
- If IoU < threshold (person lost), fall back to largest bbox

---

## src/visualize.py

**Role**: Plotting utilities for training curves and skeleton visualization.

**Key functions**:
- `plot_training_history(history)`: Loss/accuracy curves from `train_model` output
- `plot_confusion_matrix(y_true, y_pred, class_names)`: Seaborn heatmap
- `plot_skeleton_frame(joints, bone_pairs)`: Matplotlib joint overlay
- `animate_skeleton(seq, bone_pairs)`: Animated sequence (saves to GIF/MP4)
