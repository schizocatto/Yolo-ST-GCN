# Datasets

## FineGym288 (Primary Source)

The Gym288-skeleton dataset contains 38,223 skeleton sequences extracted from FineGym videos, covering 288 fine-grained gymnastic elements across four apparatus: Floor Exercise (FX), Balance Beam (BB), Uneven Bars (UB), and Vault (VT).

Full dataset documentation: [`docs/Gym288-skeleton.md`](Gym288-skeleton.md)

**File format**: `gym288_skeleton.pkl` — a Python dict with:
```python
{
  'split': {'train': [id, ...], 'test': [id, ...]},
  'annotations': [
    {
      'frame_dir': str,       # unique clip ID
      'label':     int,       # 0–287
      'total_frames': int,
      'keypoint':  np.ndarray,  # (1, T, 17, 2)  COCO17 x,y
      'kp_w_gt':   np.ndarray,  # (T, 17, 3)     x, y, score
    },
    ...
  ]
}
```

**Splits**: 28,739 train / 9,484 test (from `split` key — use these, not a random split).

**Loader**: `src/gym288_dataset.py`

---

## FineGym99 (Main Training Target)

Gym99 is a 99-class subset of Gym288 derived from the official FineGym hierarchical category mapping. It corresponds to the `gym99` split in the original FineGym benchmark.

**Building Gym99 from Gym288**:
```bash
python scripts/build_gym99_from_gym288.py \
  --gym288_path /path/to/gym288_skeleton.pkl \
  --out_path /path/to/gym99_skeleton.pkl
```

The builder (`src/gym99_builder.py`) downloads FineGym category files, matches Gym288 labels to Gym99 global IDs, filters to 99 classes, and re-indexes labels 0–98. It includes a fallback for off-by-one mismatches in the source data.

**File format**: identical structure to Gym288 but with labels 0–98 and only clips belonging to Gym99 classes.

**Loader**: `src/gym99_dataset.py`

**Class list**: `docs/gym99_categories.txt`

---

## FX (Floor Exercise) Subset

The submitted course project focuses on the **Floor Exercise** subset of Gym99. FX is the apparatus most amenable to single-person skeleton-based recognition because:
- Athletes perform without equipment (no parallel bars, beam, etc.)
- The full-body is visible throughout
- Actions include complex acrobatics distinguishable by skeleton motion

To filter to FX only, pass a label filter when loading — FX class indices are defined in `configs/experiments/` or can be read from the FineGym category mapping.

---

## Penn Action (Baseline Dataset)

8-class exercise recognition dataset. Used for early development and baseline comparisons.

**File format**: one `.mat` file per video with MATLAB struct fields:
| Field | Type | Description |
|-------|------|-------------|
| `action` | str | Class name (e.g., `'pushup'`) |
| `nframes` | int | Number of frames |
| `train` | 0 or 1 | Official train/test split flag |
| `x` | (nframes, 13) | x-coordinates for 13 joints |
| `y` | (nframes, 13) | y-coordinates for 13 joints |

**Classes**: bench_press, clean_and_jerk, jump_rope, jumping_jacks, pullup, pushup, situp, squat

**Critical rule**: Always use the `train` flag from the `.mat` file for splits. Do **not** use `sklearn.train_test_split` — Penn Action is subject-isolated, meaning the same person appears in both train and test sets if you split randomly, causing data leakage.

**Loader**: `src/penn_dataset.py`

---

## Data Tensor Format

All dataset loaders return tensors in the same shape:

```
(N, 2, T, V, 1)
 N  = number of clips
 2  = x and y coordinates
 T  = 48 frames (after temporal alignment)
 V  = number of joints (14 for Penn14, 18 for COCO18)
 1  = number of persons (always 1 in these datasets)
```

Labels are `(N,)` int64 tensors with values in `[0, num_classes-1]`.

When `return_bone=True`, loaders also return bone data in the same shape, where each joint's value is `coord(child) - coord(parent)`.

---

## Preprocessing Steps

Applied identically across all dataset loaders (in order):

1. **Extract keypoints**: Get `(T, 17, 2)` COCO17 coordinates from annotation
2. **Remap joints**: COCO17 → Penn13 or COCO18 layout via index mapping
3. **Add virtual center**: Append `mean(shoulders + hips)` as the last joint
4. **Temporal alignment**: Resample to exactly 48 frames (linear interpolation if `T < 48`, subsampling if `T > 48`)
5. **Center normalize**: Subtract the virtual center from all joints per frame (frames where virtual center is (0,0) are skipped to avoid corrupting missing-detection frames)
6. **Reshape**: Transpose to `(2, T, V, 1)` and add batch dimension

---

## Class Imbalance

Both Gym288 and Gym99 have severe class imbalance — the most common class can have 10× more samples than the rarest. Two mechanisms address this:

1. **FocalLoss** with `sqrt_inverse` alpha weighting: Reduces effective loss for well-classified majority classes and increases focus on hard minority examples.

2. **SkeletonFeeder** with `WeightedRandomSampler`: Applies heavier augmentation to minority classes and oversamples them in each mini-batch.

See `docs/training.md` for how to enable these.
