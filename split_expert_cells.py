import json

nb_path = 'notebooks/ensemble-learning-class-split.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find and remove Cell 4 (the combined expert training cell)
new_cells = []
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and any('Stage 1' in l for l in cell.get('source', [])):
        # Replace with 4 separate cells
        EXPERT_CONFIGS = {
            'VT': {'epochs': '30',  'batch_size': '64'},
            'FX': {'epochs': '50',  'batch_size': '128'},
            'BB': {'epochs': '50',  'batch_size': '128'},
            'UB': {'epochs': '40',  'batch_size': '128'},
        }
        APPARATUS_RANGES = {'VT': (0, 5), 'FX': (6, 40), 'BB': (41, 73), 'UB': (74, 98)}

        for idx, (ap, cfg) in enumerate(EXPERT_CONFIGS.items(), start=4):
            lo, hi = APPARATUS_RANGES[ap]
            n_cls = hi - lo + 1
            cell_src = [
                f"# ── Cell {idx}: Train Expert {ap} — {n_cls} classes (Clabel {lo}-{hi}) ──\n",
                 "import sys, importlib\n",
                 "\n",
                f"sys.argv = [\n",
                f"    'train_gym99.py',\n",
                 "    '--auto_build_from_gym288',\n",
                 "    '--gym288_dataset_path', GYM288_PKL,\n",
                 "    '--dataset_path',        GYM99_PKL,\n",
                f"    '--out_dir',             EXPERT_DIRS['{ap}'],\n",
                f"    '--apparatus',           '{ap}',\n",
                f"    '--epochs',              '{cfg['epochs']}',\n",
                f"    '--batch_size',          '{cfg['batch_size']}',\n",
                 "    '--lr',                  '0.001',\n",
                 "    '--num_workers',         '0',\n",
                 "    '--joint_spec_name',     'coco18',\n",
                 "    '--loss_name',           'focal',\n",
                 "    '--focal_alpha_mode',    'sqrt_inverse',\n",
                 "    '--bbox_norm',\n",
                 "    '--warmup_epochs',       '5',\n",
                 "    '--use_augment_feeder',\n",
                 "    '--use_weighted_sampler',\n",
                 "    '--grad_clip_norm',      '1.0',\n",
                 "]\n",
                 "\n",
                 "import scripts.train_gym99 as _train_script\n",
                 "importlib.reload(_train_script)\n",
                f"print(f'\\n>>> Training Expert {ap} ({n_cls} classes) ...')\n",
                 "_train_script.main()\n",
                f"print(f'\\n✅ Expert {ap} done.')\n",
            ]
            new_cells.append({
                'cell_type': 'code',
                'execution_count': None,
                'metadata': {},
                'outputs': [],
                'source': cell_src,
            })
    else:
        new_cells.append(cell)

nb['cells'] = new_cells

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print('Split Cell 4 into 4 separate expert cells')
