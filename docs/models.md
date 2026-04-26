# Model Architecture

## Joint Specifications

Before describing the models, it helps to understand the two supported joint layouts.

### Penn14
13 Penn Action joints + 1 virtual center joint (index 13).

```
Virtual Center (13)
      |
  Neck (0)
  /       \
LShoulder(1) RShoulder(2)
  |               |
LElbow(3)     RElbow(4)
  |               |
LWrist(5)     RWrist(6)
  |               |
LHip(7)         RHip(8)
  |               |
LKnee(9)       RKnee(10)
  |               |
LAnkle(11)   RAnkle(12)
```

### COCO18
17 COCO keypoints + 1 virtual center joint (index 17).

```
Nose(0)
LEye(1) REye(2)
LEar(3) REar(4)
LShoulder(5) RShoulder(6)
LElbow(7)    RElbow(8)
LWrist(9)    RWrist(10)
LHip(11)     RHip(12)
LKnee(13)    RKnee(14)
LAnkle(15)   RAnkle(16)
Virtual Center(17) = mean(5,6,11,12)
```

---

## Spatial Graph Construction

File: `src/graph.py`

The spatial graph is a 3-partition adjacency tensor `A` of shape `(3, V, V)`.

**Partition 0 — Self-loop**: Each joint is connected to itself and to joints at the same hop-distance from the virtual center.

**Partition 1 — Centripetal**: Edges that point *toward* the virtual center (child → parent).

**Partition 2 — Centrifugal**: Edges that point *away* from the virtual center (parent → child).

The three partitions allow the graph convolution to learn separate weights for same-level interactions, inward information flow, and outward information flow — which is the original ST-GCN spatial reasoning design.

All partitions are row-normalized:
```
A[k, j, :] = A[k, j, :] / (sum(A[k, j, :]) + 1e-4)
```

---

## Single-Stream ST-GCN

File: `src/model.py`

### Input / Output

```
Input:  (N, C=2, T=48, V, M=1)
         N = batch size
         C = 2  (x, y coordinates)
         T = 48 frames
         V = 14 (Penn14) or 18 (COCO18)
         M = 1 person

Output: (N, num_classes)  logits
```

### Architecture

```
Input (N, 2, 48, V, 1)
  ↓
Data BatchNorm1D  [reshape to (N·M·V, C, T)]
  ↓
Block  1: channels   3→64,   stride=1
Block  2: channels  64→64,   stride=1
Block  3: channels  64→64,   stride=1
Block  4: channels  64→64,   stride=1
Block  5: channels  64→128,  stride=2   ← temporal downsampling
Block  6: channels 128→128,  stride=1
Block  7: channels 128→128,  stride=1
Block  8: channels 128→256,  stride=2   ← temporal downsampling
Block  9: channels 256→256,  stride=1
Block 10: channels 256→256,  stride=1
  ↓
Global Average Pool → (N, 256, 1, 1)
  ↓
Classifier: Conv2D(256 → num_classes, kernel 1×1)
  ↓
Output: (N, num_classes) logits
```

### STGCN_Block

Each block is:
```
x  (N, C_in, T, V)
  ↓ Graph Conv: x = Σ_k  A[k] @ x @ W_k   → (N, C_out, T, V)
  ↓ Batch Norm + ReLU
  ↓ Temporal Conv: Conv2D(C_out, C_out, kernel=(9,1), stride=(stride,1), pad=(4,0))
  ↓ Batch Norm
  ↓ + Residual (1×1 conv projection if C_in ≠ C_out or stride ≠ 1)
  ↓ Dropout (optional)
  ↓ ReLU
```

**Edge importance weighting**: When enabled, each block has a learnable parameter matrix `M` (shape `(V, V)`) that acts as a soft mask on the adjacency, allowing the model to up- or down-weight specific joint connections during training.

---

## Two-Stream ST-GCN (Late Fusion)

File: `src/two_stream_stgcn.py`

### Motivation

Bone vectors encode relative motion between adjacent joints (velocity-like features), while joint coordinates encode absolute positions. The two views are complementary: some actions are better distinguished by where joints are, others by how they move.

### Architecture

```
joint_data (N,2,T,V,1)       bone_data (N,2,T,V,1)
       ↓                              ↓
  Joint Stream                   Bone Stream
 [Model_STGCN]                 [Model_STGCN]
       ↓                              ↓
 joint_logits (N,K)          bone_logits (N,K)
       ↓                              ↓
       └──────── Late Fusion ─────────┘
                      ↓
          α × joint_logits + (1-α) × bone_logits
                      ↓
              Output (N, K)
```

- `α` is a **learnable scalar** initialized at 0.5, passed through sigmoid so it stays in (0, 1)
- Both streams have **independent** weight sets — they do not share parameters
- The fusion is a soft weighted sum, not concatenation

### Bone Vector Construction

For each bone pair `(parent_idx, child_idx)`:
```python
bone[:, :, child_idx, :] = joint[:, :, child_idx, :] - joint[:, :, parent_idx, :]
```
Joints with no parent (the virtual center) get a zero vector.

This is computed in `src/gym99_dataset.py` and `src/gym288_dataset.py` when `return_bone=True`, not inside the model.

### When to Use Two-Stream

Use the two-stream model when:
- The dataset has enough samples to learn two sets of stream weights (generally > 5k samples)
- Actions are distinguishable by relative joint motion, not just absolute pose
- You have time for ~2× training compute

For quick baselines or small datasets, the single-stream model is a better starting point.

---

## Model Comparison

| Model | Params (Penn14) | Params (COCO18) | Notes |
|-------|----------------|----------------|-------|
| `Model_STGCN` | ~3.1M | ~3.1M | baseline |
| `TwoStream_STGCN` | ~6.2M | ~6.2M | 2× stream params + 1 scalar α |

Parameter count scales linearly with the number of joints only through the first layer (graph conv weight `W_k` shape `(C_in, V, C_out/3)`) — subsequent layers are joint-count-independent.
