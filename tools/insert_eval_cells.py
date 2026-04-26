import json
import ast

nb_path = 'notebooks/ensemble-learning-class-split.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

new_cells = []
for cell in nb['cells']:
    new_cells.append(cell)
    
    # Check if this is a training cell
    src = "".join(cell.get('source', []))
    if 'Train Expert' in src and 'train_gym99.py' in src:
        # Extract the Apparatus
        ap = 'VT' if 'VT' in src else 'FX' if 'FX' in src else 'BB' if 'BB' in src else 'UB' if 'UB' in src else None
        
        if ap:
            eval_src = [
                f"# ── Evaluate Expert {ap} ───────────────────────────────────────\n",
                "import numpy as np\n",
                "import torch\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "from sklearn.metrics import confusion_matrix, f1_score, accuracy_score\n",
                "from torch.utils.data import DataLoader, TensorDataset\n",
                "\n",
                "from src.two_stream_stgcn import TwoStream_STGCN\n",
                "from src.checkpointing import load_checkpoint\n",
                "from src.gym99_dataset import build_gym99_data_tensors\n",
                "from src.skeleton_utils import bbox_normalize, center_normalize\n",
                "\n",
                "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
                f"ap = '{ap}'\n",
                "lo, hi = APPARATUS_RANGES[ap]\n",
                "num_cls = hi - lo + 1\n",
                "\n",
                "if 'GYM99_J_DATA' not in globals():\n",
                "    print(f'Loading data into RAM for sequential evaluation...')\n",
                "    j_data, b_data, labels, flags, _, _ = build_gym99_data_tensors(\n",
                "        dataset_path=GYM99_PKL, joint_spec_name='coco18',\n",
                "        split='all', keep_unknown_split=False, return_bone_data=True\n",
                "    )\n",
                "    j_data = bbox_normalize(j_data)\n",
                "    b_data = bbox_normalize(b_data)\n",
                "    j_data = center_normalize(j_data, 17) # COCO18 center is 17\n",
                "    globals()['GYM99_J_DATA'] = j_data\n",
                "    globals()['GYM99_B_DATA'] = b_data\n",
                "    globals()['GYM99_LABELS'] = labels.numpy().astype(int)\n",
                "    globals()['GYM99_FLAGS']  = flags.numpy().astype(int)\n",
                "\n",
                "j_data = GYM99_J_DATA\n",
                "b_data = GYM99_B_DATA\n",
                "labels_np = GYM99_LABELS\n",
                "flags_np = GYM99_FLAGS\n",
                "\n",
                "mask = (labels_np >= lo) & (labels_np <= hi)\n",
                "val_mask = mask & (flags_np == 0)\n",
                "val_j = torch.tensor(j_data[val_mask], dtype=torch.float32)\n",
                "val_b = torch.tensor(b_data[val_mask], dtype=torch.float32)\n",
                "val_y = torch.tensor(labels_np[val_mask] - lo, dtype=torch.long)\n",
                "\n",
                "loader = DataLoader(TensorDataset(val_j, val_b, val_y), batch_size=64, shuffle=False)\n",
                "\n",
                "weights_path = f\"{EXPERT_DIRS[ap]}/stgcn_gym99_coco18_2s_expert_{ap}.pth\"\n",
                "model = TwoStream_STGCN(num_classes=num_cls, joint_spec='coco18').to(device)\n",
                "load_checkpoint(weights_path, model)\n",
                "model.eval()\n",
                "\n",
                "preds = []\n",
                "with torch.no_grad():\n",
                "    for bj, bb, _ in loader:\n",
                "        out = model(bj.to(device), bb.to(device))\n",
                "        preds.extend(out.argmax(1).cpu().tolist())\n",
                "\n",
                "gts = val_y.tolist()\n",
                "acc = accuracy_score(gts, preds)\n",
                "mf1 = f1_score(gts, preds, average='macro', zero_division=0)\n",
                "print(f\"\\n{'='*40}\\n[Val] {ap} - Acc: {acc:.4f} | Macro F1: {mf1:.4f}\\n{'='*40}\")\n",
                "\n",
                "cm = confusion_matrix(gts, preds, labels=list(range(num_cls)))\n",
                "plt.figure(figsize=(max(5, num_cls*0.35), max(4, num_cls*0.35)))\n",
                "sns.heatmap(cm, annot=num_cls <= 20, fmt='d', cmap='Blues',\n",
                "            xticklabels=list(range(num_cls)), yticklabels=list(range(num_cls)),\n",
                "            linewidths=0.3, linecolor='#e0e0e0', cbar=False)\n",
                "plt.title(f\"Confusion Matrix: Expert {ap}\", fontsize=13, fontweight='bold', pad=10)\n",
                "plt.ylabel('True Class', fontsize=11)\n",
                "plt.xlabel('Predicted Class', fontsize=11)\n",
                "plt.tight_layout()\n",
                "plt.show()\n"
            ]
            new_cells.append({
                'cell_type': 'code',
                'execution_count': None,
                'metadata': {},
                'outputs': [],
                'source': eval_src,
            })

nb['cells'] = new_cells

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print('Inserted per-expert Evaluation cells.')
