# Inference Guide

## Overview

The inference pipeline takes a raw video and outputs a predicted action class using:
1. **YOLOv8-pose** for per-frame 2D keypoint detection
2. **IoU-based person tracking** to follow a single person across frames
3. **ST-GCN** (single or two-stream) for sequence classification

This pipeline is only needed for real video input. For offline evaluation on the Gym99/Gym288 test set, use `scripts/evaluate.py` with pre-extracted skeletons.

---

## Quick Start

**Single video demo (command line):**
```bash
python scripts/inference_demo.py \
  --video /path/to/video.mp4 \
  --checkpoint outputs/gym99_coco18_2s/best_model.pth \
  --yolo_model yolov8n-pose.pt
```

**Batch inference on Gym99 test set:**
```bash
python scripts/inference_gym99.py \
  --dataset_path /path/to/gym99_skeleton.pkl \
  --checkpoint outputs/gym99_coco18_2s/best_model.pth
```

**Notebook demo:** `notebooks/yolo_stgcn_inference.ipynb`

---

## Pipeline Steps

### 1. YOLO Keypoint Extraction

```python
from src.inference import extract_yolo_keypoints

keypoints = extract_yolo_keypoints(
    video_path,
    yolo_model,          # loaded ultralytics YOLO model
    joint_spec_name,     # 'penn14' or 'coco18'
)
# Returns: (T, V, 2)  — T frames, V joints, (x,y) coords
```

**Frame 0**: Select the person with the largest bounding box (most prominent subject).

**Frames 1+**: Track the selected person via IoU matching:
- Compute IoU between current-frame detections and the previous frame's bbox
- Keep the detection with IoU > 0.3 (threshold)
- If no detection exceeds the threshold (person lost), fall back to largest bbox

This simple heuristic works well for single-performer gymnastics videos where the athlete is consistently the largest person in frame.

### 2. Preprocessing

Keypoints from YOLO are COCO17 format `(T, 17, 2)`. The preprocessing pipeline applies the same steps as offline training:
- Remap joints to Penn14 or COCO18 layout
- Add virtual center joint
- Temporal alignment to 48 frames
- Center normalize

This is handled inside `extract_yolo_keypoints` — the returned tensor is ready for the model.

### 3. ST-GCN Inference

```python
from src.inference import run_stgcn_inference, load_stgcn_weights

model = Model_STGCN(num_classes=99, joint_spec_name='coco18')
metadata = load_stgcn_weights(model, checkpoint_path)

class_idx, confidence = run_stgcn_inference(keypoints, model, device)
```

`load_stgcn_weights` reads the metadata saved with the checkpoint to verify the joint spec matches. It will warn if there is a mismatch.

---

## End-to-End Code Example

```python
import torch
from ultralytics import YOLO
from src.inference import extract_yolo_keypoints, run_stgcn_inference, load_stgcn_weights
from src.model import Model_STGCN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load models
yolo = YOLO('yolov8n-pose.pt')
stgcn = Model_STGCN(num_classes=99, joint_spec_name='coco18').to(device)
metadata = load_stgcn_weights(stgcn, 'outputs/best_model.pth')
stgcn.eval()

# Extract keypoints from video
keypoints = extract_yolo_keypoints('video.mp4', yolo, joint_spec_name='coco18')

# Run classification
class_idx, confidence = run_stgcn_inference(keypoints, stgcn, device)
print(f"Predicted class: {class_idx}  (confidence: {confidence:.2f})")
```

---

## Two-Stream Inference

For two-stream checkpoints, the bone stream input must be computed from the joint keypoints before calling the model:

```python
from src.two_stream_stgcn import TwoStream_STGCN
from src.gym99_dataset import compute_bone_data  # if this helper exists

stgcn = TwoStream_STGCN(num_classes=99, joint_spec_name='coco18').to(device)
load_stgcn_weights(stgcn, 'outputs/best_model_2s.pth')

joint_tensor = keypoints.unsqueeze(0)  # (1, 2, T, V, 1)
bone_tensor = compute_bone_data(joint_tensor, joint_spec_name='coco18')

with torch.no_grad():
    logits = stgcn(joint_tensor.to(device), bone_tensor.to(device))
    class_idx = logits.argmax(dim=1).item()
```

---

## YOLO Model Selection

| Model | Speed | Accuracy | Notes |
|-------|-------|----------|-------|
| `yolov8n-pose.pt` | Fastest | Lowest | Good for real-time, acceptable for gymnastics |
| `yolov8s-pose.pt` | Fast | Medium | Recommended for offline inference |
| `yolov8m-pose.pt` | Moderate | High | Better for occluded/distant athletes |
| `yolov8l-pose.pt` | Slow | Best | For maximum accuracy experiments |

Download with: `yolo download model=yolov8s-pose.pt`

---

## Known Limitations

**Multi-person videos**: The current IoU tracker selects a single person. If multiple gymnasts appear simultaneously, only the largest-bbox person is tracked. The `scripts/count_multiperson.py` script can diagnose how often this is an issue in a dataset.

**Camera cuts**: An abrupt camera angle change will cause the IoU tracker to lose the subject and fall back to the largest bbox in the new view. For competition footage, this is usually acceptable.

**Short clips**: Clips shorter than 16 frames produce poor results after temporal alignment because too many frames are interpolated. Pre-filter the dataset to remove very short clips.

**Missing keypoints**: YOLOv8 may produce (0, 0) for occluded joints. The `center_normalize` step skips frames where the virtual center is at the origin to avoid corrupting the entire sequence from one bad frame.
