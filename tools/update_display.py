import json

notebook_path = 'notebooks/flag-smoke-test.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        if any("Cell 6: Confusion Matrix Inference" in line for line in source):
            # Target the bottom display block
            new_source = []
            for line in source:
                if "from IPython.display import Image, display" in line or "TRAIN CONFUSION MATRIX" in line or "VAL CONFUSION MATRIX" in line or "display(" in line:
                    continue
                new_source.append(line)
            
            # Remove trailing newline so we can append neatly
            if new_source[-1].endswith('\n') == False:
                new_source[-1] += '\n'
                
            # Append new matplotlib block
            new_source.extend([
                "print('\\nRendering plots with matplotlib...')\n",
                "import matplotlib.pyplot as plt\n",
                "import matplotlib.image as mpimg\n",
                "\n",
                "fig, axes = plt.subplots(1, 2, figsize=(20, 9))\n",
                "axes[0].imshow(mpimg.imread(os.path.join(OUT_DIR, 'train_cm.png')))\n",
                "axes[0].axis('off')\n",
                "axes[0].set_title('Train Confusion Matrix', fontsize=16, fontweight='bold')\n",
                "\n",
                "axes[1].imshow(mpimg.imread(os.path.join(OUT_DIR, 'val_cm.png')))\n",
                "axes[1].axis('off')\n",
                "axes[1].set_title('Val Confusion Matrix', fontsize=16, fontweight='bold')\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()\n"
            ])
            cell['source'] = new_source

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print(f'Modified display block in {notebook_path}')
