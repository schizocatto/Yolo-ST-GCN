import json

notebook_path = 'notebooks/flag-smoke-test.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

inference_cell = {
 "cell_type": "code",
 "execution_count": None,
 "metadata": {},
 "outputs": [],
 "source": [
  "# \u2500\u2500 Cell 6: Confusion Matrix Inference \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n",
  "# Evaluate the trained weights on both train and val subsets (unaugmented)\n",
  "# to generate the Confusion Matrices.\n",
  "\n",
  "import os\n",
  "import torch\n",
  "from torch.utils.data import DataLoader\n",
  "from IPython.display import Image, display\n",
  "\n",
  "from src.gym99_dataset import build_gym99_data_tensors, infer_num_gym99_classes\n",
  "from src.dataset import PennActionDataset\n",
  "from src.two_stream_stgcn import TwoStream_STGCN\n",
  "from src.model import Model_STGCN\n",
  "from src.losses import build_classification_criterion\n",
  "from src.train import eval_epoch\n",
  "from src.visualize import plot_confusion_matrix\n",
  "from src.skeleton_utils import bbox_normalize\n",
  "\n",
  "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
  "use_two_stream = '--use_two_stream' in cmd\n",
  "joint_spec = 'coco18'\n",
  "\n",
  "print('1/4. Loading dataset tensors...')\n",
  "data, bone_data, labels, flags, _, _ = build_gym99_data_tensors(\n",
  "    dataset_path=GYM99_PKL,\n",
  "    joint_spec_name=joint_spec,\n",
  "    split='all',\n",
  "    keep_unknown_split=False,\n",
  "    return_bone_data=use_two_stream\n",
  ")\n",
  "\n",
  "train_mask = flags == 1\n",
  "test_mask = flags == 0\n",
  "X_train, y_train = bbox_normalize(data[train_mask]), labels[train_mask]\n",
  "X_val, y_val = bbox_normalize(data[test_mask]), labels[test_mask]\n",
  "B_train = B_val = None\n",
  "if use_two_stream:\n",
  "    B_train, B_val = bbox_normalize(bone_data[train_mask]), bbox_normalize(bone_data[test_mask])\n",
  "\n",
  "# Limit evaluation to the exact same subset used during training so inference doesn't take forever\n",
  "N_TRAIN, N_VAL = 512, 128\n",
  "X_train, y_train = X_train[:N_TRAIN], y_train[:N_TRAIN]\n",
  "X_val,   y_val   = X_val[:N_VAL],   y_val[:N_VAL]\n",
  "if use_two_stream:\n",
  "    B_train, B_val = B_train[:N_TRAIN], B_val[:N_VAL]\n",
  "\n",
  "train_ds = PennActionDataset(X_train, y_train, bone_data=B_train, include_bone=use_two_stream, joint_spec_name=joint_spec)\n",
  "val_ds   = PennActionDataset(X_val, y_val, bone_data=B_val, include_bone=use_two_stream, joint_spec_name=joint_spec)\n",
  "train_loader = DataLoader(train_ds, batch_size=64, shuffle=False)\n",
  "val_loader   = DataLoader(val_ds, batch_size=64, shuffle=False)\n",
  "\n",
  "print('2/4. Loading model & weights...')\n",
  "num_classes = infer_num_gym99_classes(GYM99_PKL)\n",
  "model = (\n",
  "    TwoStream_STGCN(num_classes=num_classes, joint_spec=joint_spec)\n",
  "    if use_two_stream\n",
  "    else Model_STGCN(num_classes=num_classes, joint_spec=joint_spec)\n",
  ").to(device)\n",
  "\n",
  "weights_name = 'stgcn_gym99_coco18_2s.pth' if use_two_stream else 'stgcn_gym99_coco18.pth'\n",
  "weights_path = os.path.join(OUT_DIR, weights_name)\n",
  "model.load_state_dict(torch.load(weights_path, map_location=device)['model_state_dict'])\n",
  "\n",
  "eval_criterion = build_classification_criterion('ce', device)\n",
  "classes_list = [str(i) for i in range(num_classes)]\n",
  "\n",
  "print('3/4. Evaluating Train Set...')\n",
  "_, _, _, train_preds, train_gt = eval_epoch(model, train_loader, eval_criterion, device)\n",
  "plot_confusion_matrix(train_gt, train_preds, classes=classes_list, title='Confusion Matrix \u2014 Train Set', out_dir=OUT_DIR, filename='train_cm.png')\n",
  "\n",
  "print('4/4. Evaluating Val Set...')\n",
  "_, _, _, val_preds, val_gt = eval_epoch(model, val_loader, eval_criterion, device)\n",
  "plot_confusion_matrix(val_gt, val_preds, classes=classes_list, title='Confusion Matrix \u2014 Val Set', out_dir=OUT_DIR, filename='val_cm.png')\n",
  "\n",
  "print('\\n\\u2500\\u2500\\u2500\\u2500 TRAIN CONFUSION MATRIX \\u2500\\u2500\\u2500\\u2500')\n",
  "display(Image(filename=os.path.join(OUT_DIR, 'train_cm.png')))\n",
  "print('\\n\\u2500\\u2500\\u2500\\u2500 VAL CONFUSION MATRIX \\u2500\\u2500\\u2500\\u2500')\n",
  "display(Image(filename=os.path.join(OUT_DIR, 'val_cm.png')))\n"
 ]
}

# Ensure we don't accidentally insert it multiple times if run multiple times
has_cell_6 = any("Cell 6: Confusion Matrix Inference" in "".join(c.get("source", [])) for c in nb['cells'])
if not has_cell_6:
    nb['cells'].append(inference_cell)

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print(f'Added inference cell to {notebook_path}')
