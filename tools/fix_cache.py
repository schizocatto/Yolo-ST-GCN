import json

nb_path = 'notebooks/ensemble-learning-class-split.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'GYM99_J_DATA' in ''.join(cell.get('source', [])):
        new_source = []
        for line in cell['source']:
            if line.startswith("if 'GYM99_J_DATA' not in globals():"):
                line = "if not all(k in globals() for k in ['GYM99_J_DATA', 'GYM99_B_DATA', 'GYM99_LABELS', 'GYM99_FLAGS']):\\n"
            new_source.append(line)
        cell['source'] = new_source

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print('Fixed robustness of globals cache.')
