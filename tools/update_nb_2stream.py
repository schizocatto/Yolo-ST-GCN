import json

nb_path = 'notebooks/ensemble-learning-class-split.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
    
    src = "".join(cell['source'])
    
    # Update Training cells to include --use_two_stream
    if "'--use_augment_feeder'," in src and "Train Expert" in src:
        new_source = []
        for line in cell['source']:
            new_source.append(line)
            if "'--use_augment_feeder'," in line:
                new_source.append("    '--use_two_stream',\n")
        cell['source'] = new_source
    
    # Update Feature Extraction cell
    if "Stage 2 — Feature Extraction" in src:
        cell['source'] = [
            "# ── Cell 8: Stage 2 — Feature Extraction ─────────────────────────────────────\n",
            "# Each frozen two-stream expert backbone → 256-dim fused joint/bone vector.\n",
            "# Vectors are L2-normalised before concatenation to prevent any expert dominating.\n",
            "# Output: super_vectors (N, 1024), labels (N,), flags (N,)\n",
            "\n",
            "import numpy as np\n",
            "import torch\n",
            "import torch.nn.functional as F\n",
            "from torch.utils.data import DataLoader, TensorDataset\n",
            "\n",
            "from src.gym99_dataset import build_gym99_data_tensors\n",
            "from src.skeleton_utils import bbox_normalize\n",
            "from src.two_stream_stgcn import TwoStream_STGCN\n",
            "from src.checkpointing import load_checkpoint\n",
            "\n",
            "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
            "print(f'Device: {device}')\n",
            "\n",
            "print('\\nLoading full Gym99 tensors (train + val)...')\n",
            "joint_data, bone_data, labels, flags, _, _ = build_gym99_data_tensors(\n",
            "    dataset_path=GYM99_PKL,\n",
            "    joint_spec_name='coco18',\n",
            "    split='all',\n",
            "    keep_unknown_split=False,\n",
            "    return_bone_data=True,\n",
            ")\n",
            "joint_data = bbox_normalize(joint_data)\n",
            "bone_data  = bbox_normalize(bone_data)\n",
            "labels_np  = labels.numpy().astype('int64')\n",
            "flags_np   = flags.numpy().astype('int64')\n",
            "print(f'  Total={len(joint_data)}  train={int((flags_np==1).sum())}  val={int((flags_np==0).sum())}')\n",
            "\n",
            "def extract_features(b_joint: torch.Tensor, b_bone: torch.Tensor, apparatus: str, batch_size: int = 64) -> np.ndarray:\n",
            "    lo, hi = APPARATUS_RANGES[apparatus]\n",
            "    # Note the _2s in the weights filename\n",
            "    weights_path = Path(EXPERT_DIRS[apparatus]) / f'stgcn_gym99_coco18_2s_expert_{apparatus}.pth'\n",
            "    model = TwoStream_STGCN(num_classes=hi - lo + 1, joint_spec='coco18').to(device)\n",
            "    load_checkpoint(str(weights_path), model)\n",
            "    model.eval()\n",
            "    \n",
            "    def get_stream_feat(batch, stream_model):\n",
            "        n, c, t, v, m = batch.size()\n",
            "        x = batch.permute(0, 4, 3, 1, 2).contiguous().view(n, m * v * c, t)\n",
            "        x = stream_model.data_bn(x)\n",
            "        x = x.view(n, m, v, c, t).permute(0, 1, 3, 4, 2).contiguous().view(n * m, c, t, v)\n",
            "        for gcn in stream_model.st_gcn_networks:\n",
            "            x = gcn(x)\n",
            "        x = F.avg_pool2d(x, x.size()[2:])           # (N*M, 256, 1, 1)\n",
            "        x = x.view(n, m, -1).mean(dim=1)            # (N, 256)\n",
            "        return x\n",
            "\n",
            "    parts = []\n",
            "    loader = DataLoader(TensorDataset(b_joint, b_bone), batch_size=batch_size, shuffle=False)\n",
            "    with torch.no_grad():\n",
            "        for (bj, bb) in loader:\n",
            "            fj = get_stream_feat(bj.to(device), model.joint_stream)\n",
            "            fb = get_stream_feat(bb.to(device), model.bone_stream)\n",
            "            alpha = torch.sigmoid(model.alpha_logit)\n",
            "            fused = alpha * fj + (1.0 - alpha) * fb\n",
            "            parts.append(fused.cpu().numpy())\n",
            "    return np.concatenate(parts, axis=0)\n",
            "\n",
            "Path(FEATURES_DIR).mkdir(parents=True, exist_ok=True)\n",
            "all_parts = []\n",
            "\n",
            "for ap in APPARATUS_LIST:\n",
            "    print(f'\\nExtracting — Expert {ap}...')\n",
            "    feats = extract_features(joint_data, bone_data, ap)      # (N, 256)\n",
            "    norms = np.linalg.norm(feats, axis=1, keepdims=True).clip(min=1e-8)\n",
            "    feats = feats / norms                                # L2-normalize\n",
            "    all_parts.append(feats)\n",
            "    print(f'  shape={feats.shape}  |  avg L2-norm={np.linalg.norm(feats, axis=1).mean():.4f}')\n",
            "\n",
            "super_vectors = np.concatenate(all_parts, axis=1)        # (N, 1024)\n",
            "np.save(f'{FEATURES_DIR}/super_vectors.npy', super_vectors)\n",
            "np.save(f'{FEATURES_DIR}/labels.npy',        labels_np)\n",
            "np.save(f'{FEATURES_DIR}/flags.npy',         flags_np)\n",
            "print(f'\\n✅ Saved super_vectors {super_vectors.shape} → {FEATURES_DIR}')\n"
        ]

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print('Updated notebook to use Two-Stream components')
