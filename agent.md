# agent.md — Yolo-ST-GCN

This file is the authoritative context for AI agents (Claude, Copilot, etc.) working in this repo.
Read it before making any changes.

---

## Project summary
End-to-end exercise action recognition:
**YOLOv8-pose** extracts per-frame 2-D keypoints → **IoU tracking** locks onto the subject
across frames → **ST-GCN** classifies the skeleton sequence into one of 8 Penn Action exercise classes.

---

## Module map

| File | Responsibility |
|------|----------------|
| `src/config.py`     | All shared constants — never duplicate these elsewhere |
| `src/graph.py`      | `Graph_PennAction_14Nodes` — builds the 3-partition (3,14,14) adjacency tensor |
| `src/model.py`      | `STGCN_Block`, `Model_STGCN` — PyTorch architecture |
| `src/dataset.py`    | Preprocessing helpers + `PennActionDataset`; `build_data_tensors` returns `(data, labels, flags, raw_counts)` |
| `src/train.py`      | `train_epoch`, `eval_epoch`, `train_model` |
| `src/inference.py`  | `select_best_person`, `compute_iou`, `extract_yolo_keypoints` (with IoU tracking), `run_stgcn_inference`, `load_stgcn_weights` |
| `src/visualize.py`  | All Matplotlib/Seaborn plots — `out_dir=None` → show, else save |
| `scripts/train.py`  | CLI for full training run — uses official `train` flag split |
| `scripts/evaluate.py` | CLI for YOLO + ST-GCN pipeline evaluation — uses official `train` flag split |
| `scripts/inference_demo.py` | CLI for single-video demo |
| `scripts/count_multiperson.py` | Diagnostic: counts frames where YOLO detects ≥2 people |
| `tests/test_pipeline.py` | pytest smoke tests — synthetic .mat files, no real dataset needed |

---

## Data shape convention
Model input: `(N, C=2, T=64, V=14, M=1)` — batch × channels × frames × joints × persons

---

## Train/test split — IMPORTANT
Use the **official Penn Action `train` flag** from each `.mat` file, NOT `sklearn.train_test_split`.

```python
train_flag = int(mat_data['train'][0][0])  # 1 = official train, 0 = official test
```

The official split is subject-isolated: the same person does not appear in both partitions.
Random splits cause data leakage (model sees same person's body proportions in train and val).

**Never use `train_test_split` on Penn Action data.**

---

## Inference — single-person selection + IoU tracking
YOLO detects all people in a frame. Penn Action videos have one labelled subject.

Two-step approach in `extract_yolo_keypoints()`:
1. **Frame 0**: select the person with the **largest bounding box area** (`select_best_person`).
2. **Frames 1+**: select the detection with **highest IoU** to the previous frame's bbox (`compute_iou`).
   Falls back to largest-bbox if no detection exceeds IoU threshold (0.3).

This ensures skeleton continuity across frames and eliminates multi-person noise.

---

## Key data contracts

### Joint indexing (0-based)
```
0 head      1 l_sho     2 r_sho     3 l_elbow   4 r_elbow
5 l_wrist   6 r_wrist   7 l_hip     8 r_hip      9 l_knee
10 r_knee   11 l_ankle  12 r_ankle  13 virtual_center (mean of 1,2,7,8)
```

### COCO → Penn Action remapping
`COCO_TO_PENN_IDX = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]`
Drops COCO indices 1–4 (eyes/ears).

### Adjacency tensor partitions — `A` shape `(3, 14, 14)`
- `A[0]` — self-loop / same hop-distance from center
- `A[1]` — centripetal (toward virtual center joint 13)
- `A[2]` — centrifugal (away from virtual center joint 13)

---

## Coding rules

- All constants live in `src/config.py`. Never hardcode class lists, joint indices, or bone pairs elsewhere.
- `src/` modules must not import from `scripts/`. Dependency only goes the other way.
- CLI argument parsing belongs only in `scripts/`.
- All plotting functions follow the `out_dir: Optional[str]` pattern in `src/visualize.py`.
- Use `load_stgcn_weights()` (not raw `torch.load`) — handles legacy key remapping transparently.
- Tests use real synthetic `.mat` files with the `train` flag field — no mocking.
- Do not use `train_test_split` anywhere in this codebase for Penn Action data.

---

## Running things

```bash
# Smoke tests (no dataset needed)
pytest tests/test_pipeline.py -v

# Train (uses official train flag split)
python scripts/train.py \
    --labels_dir /path/to/Penn_Action/labels \
    --out_dir outputs/

# Single-video demo
python scripts/inference_demo.py \
    --labels_dir /path/to/Penn_Action/labels \
    --frames_dir /path/to/Penn_Action/frames \
    --weights outputs/stgcn_penn_action.pth

# Full pipeline evaluation (uses official test flag split)
python scripts/evaluate.py \
    --labels_dir /path/to/Penn_Action/labels \
    --frames_dir /path/to/Penn_Action/frames \
    --weights outputs/stgcn_penn_action.pth

# Multi-person diagnostic (for report)
python scripts/count_multiperson.py \
    --labels_dir /path/to/Penn_Action/labels \
    --frames_dir /path/to/Penn_Action/frames \
    --out_dir outputs/
```

---

## Roadmap (post-pipeline-stabilization)

**Branch 3A** — if accuracy holds after single-person fix:
- Fine-tune YOLO on cropped single-person Penn Action frames
- Convert Penn Action 13-joint GT to YOLO label format
- Focus on low-F1 classes (bench_press, etc.)

**Branch 3B** — if accuracy drops:
- Joint Dropout augmentation in `src/dataset.py` (randomly zero keypoints during training)
- Bone-stream: add edge vectors between adjacent joints as a second input channel to ST-GCN

---

## Original notebooks
Archived in `notebooks/`. The `yolo_stgcn_inference_progress_bar_run_completed.ipynb`
contains the completed Colab run output and the HuggingFace weights URL for the published checkpoint.

---

## Gym99 work (branch: duy) — session 2026-04-15/16

### What was done
- Added full COCO-17 joint support for Gym99 (all 17 joints kept, including face landmarks).
  Face joints (eyes, ears) contribute to the centripetal/centrifugal adjacency partitions.
- Virtual center (joint 17 = mean of joints 5,6,11,12) appended → **18 joints total**.
- New files/classes added:
  - `src/config.py`: `COCO17_JOINT_NAMES`, `COCO17_BONES`, `COCO17_BONES_18`, `COCO17_BONE_PAIRS_18`
  - `src/graph.py`: `Graph_COCO17_18Nodes` — 3-partition adjacency `(3, 18, 18)`
  - `src/skeleton_utils.py`: `add_virtual_center_joint_coco17`, `to_stgcn_input_from_coco17_full` → `(2, T, 18, 1)`
  - `src/gym99_dataset.py`: switched to 18-joint pipeline; tensors are `(N, 2, T, 18, 1)`
  - `src/model.py`: `Model_STGCN_COCO18`
  - `src/two_stream_stgcn.py`: `TwoStream_STGCN_COCO18`
  - `scripts/train_gym99.py` / `inference_gym99.py`: use COCO18 models; weights saved as `stgcn_gym99_coco18.pth`
  - `notebooks/stgcn-gym99.ipynb`: full training notebook for Gym99

### Gym99 dataset — no standalone HuggingFace repo exists
Derived from Gym288 in the notebook:
1. Download `Lozumi/Gym288-skeleton`
2. Fetch FineGym category files from `sdolivia.github.io/FineGym/resources/dataset/`
3. Parse `Clabel` and `Glabel` fields (format: `Clabel: N; set: S; Glabel: G; desc`)
4. Join via `Glabel` (shared global ID): `Gym288 Clabel → Glabel → Gym99 Clabel`
5. Filter + remap annotations → save `gym99_skeleton.pkl`

### Currently training
- Command: `scripts/train_gym99.py --batch_size 256 --num_workers 2`
- GPU utilisation was low with batch_size=64 → quadrupled to 256
- To keep display awake on macOS during training: `caffeinate -id`

### Pending
- Teammate is refactoring the codebase on branch `duy` — pull latest before continuing
