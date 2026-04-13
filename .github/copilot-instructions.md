# GitHub Copilot Instructions — Yolo-ST-GCN

This project implements an end-to-end **human action recognition** pipeline that combines:
- **YOLOv8-pose** for per-frame skeleton keypoint extraction
- **ST-GCN** (Spatial-Temporal Graph Convolutional Network) for action classification

The dataset is [Penn Action](https://www.kaggle.com/datasets/kaushalbora18/penn-action-dataset),
filtered to 8 exercise classes.

---

## Repository layout

```
src/
  config.py       — all shared constants (classes, joint names, bone pairs, COCO→Penn mapping)
  graph.py        — Graph_PennAction_14Nodes (builds the 3-partition adjacency tensor A)
  model.py        — STGCN_Block, Model_STGCN
  dataset.py      — preprocessing helpers + PennActionDataset (PyTorch Dataset)
  train.py        — train_epoch(), eval_epoch(), train_model()
  inference.py    — extract_yolo_keypoints(), run_stgcn_inference(), load_stgcn_weights()
  visualize.py    — all Matplotlib/Seaborn plotting functions

scripts/
  train.py            — CLI: train the model end-to-end
  evaluate.py         — CLI: run YOLO + ST-GCN on val set, report metrics
  inference_demo.py   — CLI: single-video demo with skeleton overlay

tests/
  test_pipeline.py    — pytest smoke tests (no real dataset needed)

notebooks/          — archived Colab notebooks (source of truth for original code)
outputs/            — saved weights (.pth), plots (.png)
```

---

## Key data contracts

### Skeleton tensor format
All model inputs follow the shape convention: `(N, C, T, V, M)`
- `N` — batch size
- `C` — channels = 2 (x, y pixel coordinates)
- `T` — frames = 64 (after temporal alignment)
- `V` — joints = 14 (13 Penn Action GT + 1 virtual center joint)
- `M` — persons = 1

### Joint indexing (0-based)
```
0 head      1 l_sho     2 r_sho     3 l_elbow   4 r_elbow
5 l_wrist   6 r_wrist   7 l_hip     8 r_hip      9 l_knee
10 r_knee   11 l_ankle  12 r_ankle  13 virtual_center (mean of 1,2,7,8)
```

### COCO → Penn Action remapping
`COCO_TO_PENN_IDX = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]`
Drops COCO indices 1–4 (eyes/ears), maps the remaining 13 to Penn Action 0–12.

### Adjacency tensor partitions
`A` has shape `(3, 14, 14)`:
- `A[0]` — self-loop / same hop-distance from center
- `A[1]` — centripetal (toward virtual center joint 13)
- `A[2]` — centrifugal (away from virtual center joint 13)

---

## Coding conventions

- **All constants live in `src/config.py`**. Never hard-code class lists, joint counts,
  or index mappings inline in other files — import from config.
- **Type hints on all public functions**. Use `Optional`, `List`, `Tuple` from `typing`.
- **`out_dir` pattern in visualize.py**: every plotting function accepts
  `out_dir: Optional[str]`. When `None`, call `plt.show()`; when provided, save to disk.
- **No argparse in `src/`**. CLI argument parsing belongs only in `scripts/`.
- **Weight key remapping**: older checkpoints use `gcn_conv`/`tcn_conv`/`residual`;
  current model uses `gcn`/`tcn`/`res`. Always use `load_stgcn_weights()` from
  `src/inference.py` to handle both transparently.
- **No mock objects in tests**. Use real (synthetic) `.mat` files via the
  `synthetic_labels_dir` pytest fixture in `tests/test_pipeline.py`.
- Prefer `np.einsum` for the GCN aggregation step (`nkctv,kvw->nctw`).

---

## Common extension points

| Task | Where to work |
|------|---------------|
| Add a new action class | `src/config.py` → `EXERCISE_CLASSES` |
| Change the graph topology | `src/graph.py` → `Graph_PennAction_14Nodes.edges` |
| Add a new ST-GCN block | `src/model.py` → `Model_STGCN.st_gcn_networks` |
| Add a new plot | `src/visualize.py` — follow the `_save_or_show()` pattern |
| Add a new metric | `src/train.py` → `eval_epoch()` return values |
| Add a new CLI script | `scripts/` — import from `src/`, add `sys.path.insert` for repo root |

---

## Do not

- Do not import from `scripts/` inside `src/` — the dependency must go the other way.
- Do not hardcode file paths; always accept them as function parameters or CLI args.
- Do not skip the COCO→Penn remapping step when processing YOLO output.
- Do not use `plt.savefig` / `plt.show` directly in `src/` modules outside `visualize.py`.
