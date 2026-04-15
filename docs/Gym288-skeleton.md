---
license: cc-by-4.0
task_categories:
- video-classification
language:
- en
size_categories:
- 10K<n<100K
---

# Gym288-skeleton Dataset

**License:** [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/)

## Overview

The **Gym288-skeleton** dataset is a human skeleton-based action recognition benchmark derived from the **Gym288** subset of the [FineGym](https://sdolivia.github.io/FineGym/) dataset. It provides temporally precise, fine-grained annotations of gymnastic actions along with 2D human pose sequences extracted from original video frames.

This dataset is designed to support research in:
- Fine-grained action recognition
- Temporally corrupted or incomplete action modeling
- Skeleton-based representation learning
- Physics-aware motion understanding

The dataset was introduced and used in the paper [**"FineTec: Fine-grained Action Recognition under Temporal Corruption"**](https://smartdianlab.github.io/projects-FineTec/), which has been accepted to AAAI 2026. In this work, the dataset serves as the primary evaluation benchmark for recognizing fine-grained actions from temporally corrupted skeleton sequences.


## Dataset Structure

The dataset is distributed as a single Python dictionary with two top-level keys: `split` and `annotations`.

### Top-Level Keys

- **`split`**: Dictionary containing train/test splits.
  - `train`: List of 28,739 sample IDs (strings)
  - `test`: List of 9,484 sample IDs (strings)

- **`annotations`**: List of 38,223 dictionaries, each representing one action instance with the following fields:

| Key | Type | Shape / Example | Description |
|-----|------|------------------|-------------|
| `frame_dir` | `str` | `"A0xAXXysHUo_002184_002237_0035_0036"` | Unique identifier for the action clip |
| `label` | `int` | `268` | Class label (0–287, corresponding to 288 fine-grained gymnastic elements) |
| `img_shape` | `tuple` | `(720, 1280)` | Height and width of original video frames |
| `original_shape` | `tuple` | `(720, 1280)` | Same as `img_shape` (for compatibility) |
| `total_frames` | `int` | `48` | Number of frames in the action sequence |
| `keypoint` | `np.ndarray` (float16) | `(1, T, 17, 2)` | 2D joint coordinates (x, y) for 17 COCO-style keypoints over T frames |
| `keypoint_score` | `np.ndarray` (float16) | `(1, T, 17)` | Confidence scores for each keypoint |
| `kp_wo_gt` | `np.ndarray` (float32) | `(T, 17, 3)` | Placeholder array (all zeros); originally intended for corrupted/noisy poses without ground truth |
| `kp_w_gt` | `np.ndarray` (float32) | `(T, 17, 3)` | Ground-truth 2D poses with confidence as third channel (x, y, score) |

> **Note**: The first dimension (`1`) in `keypoint` and `keypoint_score` corresponds to the number of persons (always 1 in this dataset).

## Action Classes

The dataset contains **288 distinct gymnastic elements** across four apparatuses:
- Floor Exercise (FX)
- Balance Beam (BB)
- Uneven Bars (UB)
- Vault – Women (VT)

Each class represents a highly specific movement (e.g., *"Switch leap with 0.5 turn"*, *"Clear hip circle backward with 1 turn to handstand"*), reflecting the fine-grained nature of competitive gymnastics scoring.

For the full list of class names and mappings, please refer to the [website](https://sdolivia.github.io/FineGym/) and [paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Shao_FineGym_A_Hierarchical_Video_Dataset_for_Fine-Grained_Action_Understanding_CVPR_2020_paper.html) of FineGym.

## Usage Example

```python
import numpy as np

# Load the dataset (e.g., using pickle or torch.load)
with open("gym288_skeleton.pkl", "rb") as f:
    data = pickle.load(f)

# Access training samples
train_ids = data["split"]["train"]  # list of strings

# Access annotations
sample = data["annotations"][0]
print("Label:", sample["label"])
print("Frames:", sample["total_frames"])
print("Keypoints shape:", sample["keypoint"].shape)  # (1, T, 17, 2)

# Extract skeleton sequence for model input
skeleton_seq = sample["keypoint"][0]  # (T, 17, 2)
```

## Citation

If you use this dataset in your research, please cite both the **FineTec** and **FineGym** papers. FineTec's citation information will be updated upon publication.

```bibtex
@inproceedings{shao2020finegym,
  title={FineGym: A Hierarchical Video Dataset for Fine-grained Action Understanding},
  author={Shao, Dian and Zhao, Yue and Dai, Bo and Lin, Dahua},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={2616--2625},
  year={2020}
}

@article{shao2026finetec,
  title={FineTec: Fine-Grained Action Recognition Under Temporal Corruption via Skeleton Decomposition and Sequence Completion},
  volume={40},
  url={https://ojs.aaai.org/index.php/AAAI/article/view/37838},
  DOI={10.1609/aaai.v40i11.37838},
  number={11},
  journal={Proceedings of the AAAI Conference on Artificial Intelligence},
  author={Shao, Dian and Shi, Mingfei and Liu, Like},
  year={2026},
  month={Mar.},
  pages={8842-8850}
}
```


## License

This dataset is licensed under [Creative Commons Attribution 4.0 International (CC-BY-4.0)](https://creativecommons.org/licenses/by/4.0/).  
You are free to share and adapt the material, even commercially, as long as appropriate credit is given.

> **Note**: The underlying video data remains the property of its original sources (e.g., YouTube). This dataset only distributes extracted pose annotations, not raw videos.


## Acknowledgements

- Skeletons were extracted using pose estimators [HRNet](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch) on the FineGym video corpus.
- We thank the authors of FineGym for their foundational work in fine-grained action recognition.