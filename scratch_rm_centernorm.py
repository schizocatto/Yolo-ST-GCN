import json, sys
sys.stdout.reconfigure(encoding='utf-8')

NOTEBOOKS = [
    'd:/Coding/School/Y3-K2/TGMT/Final/ST-GCN/Yolo-ST-GCN/notebooks/experiments/experiment_ablation_study.ipynb',
    'd:/Coding/School/Y3-K2/TGMT/Final/ST-GCN/Yolo-ST-GCN/notebooks/experiments/ablation_singlestream.ipynb',
    'd:/Coding/School/Y3-K2/TGMT/Final/ST-GCN/Yolo-ST-GCN/notebooks/experiments/ablation_twostream.ipynb',
    'd:/Coding/School/Y3-K2/TGMT/Final/ST-GCN/Yolo-ST-GCN/notebooks/experiments/experiment_2stream_depth_search.ipynb',
]

# Strings to remove — every variant that can appear in a sys.argv block
REMOVE_PATTERNS = [
    "        '--center_norm',\\n",      # inside a multi-line string literal (source is str)
    "        '--center_norm',\n",       # same but already decoded
    "\"        '--center_norm',\\n\"",  # inside a list element
    "\"        '--center_norm',\n\"",
]

def strip_center_norm(src):
    """Remove '--center_norm' argv entry from a cell source (str or list)."""
    if isinstance(src, str):
        # Remove whichever variant is present
        for pat in [
            "        '--center_norm',\n",
            "        '--center_norm',\\n",
        ]:
            src = src.replace(pat, '')
        return src
    elif isinstance(src, list):
        # List of strings — filter out lines that are exactly the center_norm entry
        return [
            line for line in src
            if line.strip() not in ("'--center_norm',", "'--center_norm',\\n",)
            and "'--center_norm'" not in line
        ]
    return src

total_changed = 0

for nb_path in NOTEBOOKS:
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    changed = 0
    for cell in nb['cells']:
        if cell.get('cell_type') != 'code':
            continue
        src = cell.get('source', '')
        new_src = strip_center_norm(src)
        if new_src != src:
            cell['source'] = new_src
            changed += 1

    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"[{changed} cells patched] {nb_path.split('/')[-1]}")
    total_changed += changed

print(f"\nDone. Total cells patched: {total_changed}")

# Verify: grep for remaining center_norm
print("\n--- Verification ---")
for nb_path in NOTEBOOKS:
    with open(nb_path, 'r', encoding='utf-8') as f:
        content = f.read()
    hits = content.count('center_norm')
    status = "OK (0 hits)" if hits == 0 else f"WARNING: {hits} hits remaining!"
    print(f"  {nb_path.split('/')[-1]}: {status}")
