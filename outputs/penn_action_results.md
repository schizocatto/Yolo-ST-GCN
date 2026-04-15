# Penn Action — Experiment Results

## Dataset
- Total: 1163 videos (8 exercise classes)
- Split: subject-isolated, last 30% of video IDs per class = test
  (fallback applied — dataset has no official test flag)
- Train: 814 / Test: 349

## Split comparison: old vs new

| | Old notebook (leaky) | Exp 1 (subject-isolated) |
|---|---|---|
| Split method | `train_test_split` 80/20 stratified | Per-class last-30% by video ID |
| Train size | 930 | 814 |
| Val/Test size | 233 | 349 |
| Accuracy | 0.9828 | 0.9628 |
| Macro F1 | 0.9840 | 0.9629 |

The ~2% drop is expected — old numbers were inflated by subject leakage.

---

## Experiment 1 — ST-GCN on GT Skeleton (official split)

**Accuracy: 0.9628 / Macro F1: 0.9629** — 50 epochs, lr=1e-3

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| bench_press | 0.89 | 1.00 | 0.94 | 42 |
| clean_and_jerk | 1.00 | 1.00 | 1.00 | 26 |
| jump_rope | 0.96 | 0.96 | 0.96 | 25 |
| jumping_jacks | 1.00 | 0.97 | 0.99 | 34 |
| pullup | 0.95 | 0.95 | 0.95 | 60 |
| pushup | 0.98 | 0.97 | 0.98 | 63 |
| situp | 0.90 | 0.93 | 0.92 | 30 |
| squat | 1.00 | 0.94 | 0.97 | 69 |

---

## Experiment 2 — Multi-person Detection (YOLO)

| Class | Videos ≥2 people | Frames ≥2 people |
|---|---|---|
| bench_press | 122/140 (87.1%) | 6006/11576 (51.9%) |
| clean_and_jerk | 66/88 (75.0%) | 8893/23862 (37.3%) |
| squat | 155/231 (67.1%) | 7052/21351 (33.0%) |
| pushup | 141/211 (66.8%) | 3658/10513 (34.8%) |
| situp | 55/100 (55.0%) | 2391/8763 (27.3%) |
| jump_rope | 35/82 (42.7%) | 374/3642 (10.3%) |
| pullup | 85/199 (42.7%) | 2923/13865 (21.1%) |
| jumping_jacks | 47/112 (42.0%) | 501/3362 (14.9%) |
| **TOTAL** | **706/1163 (60.7%)** | **31798/96934 (32.8%)** |

---

## Experiment 3 — Full Pipeline (YOLO + IoU Tracking + ST-GCN)

**Accuracy: 0.7908 / Macro F1: 0.7626** — 349 evaluated, 0 skipped

| Class | GT F1 (Exp 1) | Pipeline F1 (Exp 3) | Drop |
|---|---|---|---|
| bench_press | 0.94 | 0.47 | −0.47 |
| situp | 0.92 | 0.69 | −0.23 |
| clean_and_jerk | 1.00 | 0.77 | −0.23 |
| pushup | 0.98 | 0.77 | −0.21 |
| jump_rope | 0.96 | 0.78 | −0.18 |
| jumping_jacks | 0.99 | 0.82 | −0.17 |
| pullup | 0.95 | 0.89 | −0.06 |
| squat | 0.97 | 0.91 | −0.06 |

---

## Decision — Branch 3B

F1 drop = **−0.2003** (threshold: −0.05) → **Branch 3B**

bench_press collapse (0.94 → 0.47) directly correlates with highest multi-person noise (87.1% of videos).
The model needs to be more robust to noisy/missing keypoints.

**Next steps:**
1. Joint Dropout — randomly zero keypoints during ST-GCN training
2. Bone-stream — add edge vectors between adjacent joints as a second input channel
