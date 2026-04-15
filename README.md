# Yolo-ST-GCN: Config-Driven Skeleton Pipeline

Tai lieu nay mo ta cach van hanh code theo huong chuyen nghiep de:
- cap nhat model thuong xuyen,
- cap nhat input/keypoint format thuong xuyen,
- giam toi da so file phai sua khi doi tu 14 khop sang 18 khop.

## 1) Nguyen tac kien truc da ap dung

### 1.1 Joint Spec Registry
Toan bo layout khop duoc quan ly tai `src/joint_specs.py`:
- `penn14`: 13 Penn + 1 khop ao.
- `coco18`: 17 COCO + 1 khop ao.

Moi spec khai bao day du:
- `num_joints`
- `center_joint`
- `bone_pairs`
- `coco_to_layout_idx`
- `virtual_center_parents`

=> Khi them layout moi, chi can them 1 spec moi (khong sua model core).

### 1.2 Graph / Model factory theo spec
- `src/graph.py`: `GraphSkeleton(joint_spec=...)`
- `src/model.py`: `Model_STGCN(..., joint_spec=...)`
- `src/two_stream_stgcn.py`: `TwoStream_STGCN(..., joint_spec=...)`

=> Model va graph tu dong lay so khop V theo spec.

### 1.3 Dataset adapter theo spec
Cac loader da support `joint_spec_name`:
- `src/coco_dataset.py`
- `src/gym288_dataset.py`
- `src/gym99_dataset.py`
- dispatcher `src/dataset.py`

=> Doi format input bang tham so, khong can doi logic train loop.

### 1.4 Checkpoint co metadata
- Save/load checkpoint thong qua `src/checkpointing.py`.
- Luu metadata: `joint_spec_name`, `use_two_stream`, `dataset_format`, `num_classes`.
- Infer script canh bao neu load weight khac spec.

=> Tranh loi load nham weight 14-joint cho model 18-joint.

### 1.5 Experiment config file (JSON)
- `src/experiment_config.py`
- Script train/infer da ho tro `--experiment_config`.
- Co file mau:
  - `configs/experiments/gym99_penn14_2s.json`
  - `configs/experiments/gym99_coco18_2s.json`

=> Batch update tham so ma khong can sua script.

---

## 2) Cach chay nhanh theo huong khong sua code

## 2.1 Train Gym99 theo Penn14 (2-stream)
```bash
python scripts/train_gym99.py \
  --dataset_path /path/to/gym99_skeleton.pkl \
  --out_dir outputs/gym99_penn14_2s \
  --joint_spec_name penn14 \
  --use_two_stream \
  --num_workers 2
```

## 2.2 Train Gym99 theo COCO18 (2-stream)
```bash
python scripts/train_gym99.py \
  --dataset_path /path/to/gym99_skeleton.pkl \
  --out_dir outputs/gym99_coco18_2s \
  --joint_spec_name coco18 \
  --use_two_stream \
  --num_workers 2
```

## 2.3 Chay bang config JSON
```bash
python scripts/train_gym99.py \
  --dataset_path /path/to/gym99_skeleton.pkl \
  --out_dir outputs/gym99_cfg \
  --experiment_config configs/experiments/gym99_coco18_2s.json
```

Luu y: tham so CLI van uu tien cao hon config (co the override de test nhanh).

---

## 3) Lo trinh migration 14 -> 18 khop (de nghi)

### Buoc 1: Baseline
- Train/infer `joint_spec_name=penn14`.
- Chot metric baseline (top1, macro_f1, speed).

### Buoc 2: Song song A/B
- Train/infer them `joint_spec_name=coco18`.
- So sanh cong bang cung split va hyperparam.

### Buoc 3: Chuyen doi
- Neu coco18 on dinh hon, doi config production sang `coco18`.
- Giu model penn14 lam fallback trong 1-2 chu ky phat hanh.

### Buoc 4: Chuan hoa monitoring
- Luu metadata checkpoint.
- Ghi log ro `joint_spec_name`, `use_two_stream`, `num_classes` cho moi run.

---

## 4) Them layout khop moi ma khong sua nhieu file

Chi can:
1. Them 1 spec moi trong `src/joint_specs.py`.
2. Chay script voi `--joint_spec_name <spec_moi>`.

Thong thuong KHONG can sua:
- `src/model.py`
- `src/graph.py`
- train/eval loop

---

## 5) Ghi chu ve du lieu Gym99/Gym288 hien tai

- Du lieu goc co 17 khop COCO.
- Khop ao thu 18 duoc tinh trong pipeline khi chon spec co virtual center (`coco18` hoac `penn14`).
- Neu chon `penn14`: se remap ve layout Penn + virtual center.
- Neu chon `coco18`: giu 17 COCO + them virtual center.

---

## 6) Cac script da ho tro joint spec + config

- `scripts/train.py`
- `scripts/train_gym288.py`
- `scripts/inference_gym288.py`
- `scripts/train_gym99.py`
- `scripts/inference_gym99.py`

Cac script tren deu ho tro:
- `--joint_spec_name {penn14,coco18}`
- `--experiment_config <json>`
- `--num_workers` (hoac alias `--num_wokers`)
