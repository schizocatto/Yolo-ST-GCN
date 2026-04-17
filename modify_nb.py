import json

notebook_path = 'notebooks/flag-smoke-test.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        is_cmd_cell = False
        for line in source:
            if "scripts/train_gym99.py" in line:
                is_cmd_cell = True
                break
                
        if is_cmd_cell:
            new_source = []
            for line in source:
                new_source.append(line)
                if "preload_vram" in line:
                    new_source.append("    '--use_augment_feeder',\n")
                    new_source.append("    '--use_weighted_sampler',\n")
            cell['source'] = new_source

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print(f'Modified {notebook_path}')
