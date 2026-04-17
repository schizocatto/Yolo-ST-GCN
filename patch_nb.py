import json

nb_path = 'notebooks/ensemble-learning-class-split.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
    new_src = []
    for line in cell['source']:
        # Fix the broken print line in Cell 6
        if 'META_EPOCHS}  ' in line and line.strip().startswith('print('):
            # Reconstruct properly: merge the split print into one clean line
            line = (
                "        print(f'Epoch {epoch:3d}/{META_EPOCHS}  '\n"
                "              f'train_loss={history[\"train_loss\"][-1]:.4f}  train_acc={t_acc:.4f}  '\n"
                "              f'val_loss={history[\"val_loss\"][-1]:.4f}  val_acc={v_acc:.4f}')\n"
            )
            # Skip next 2 lines that are continuation of the broken print
            new_src.append(line)
            continue
        # Skip broken continuation lines
        if "f'train_loss=" in line or "f'val_loss=" in line:
            continue
        new_src.append(line)
    cell['source'] = new_src

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print('Done patching Cell 6')
