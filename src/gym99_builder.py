"""
gym99_builder.py
Build Gym99-style pickle from a Gym288-skeleton pickle using FineGym category files.
"""

import os
import pickle
import urllib.request
from typing import Dict, List, Tuple

from src.config import GYM288_CATEGORIES_URL, GYM99_CATEGORIES_URL


def parse_finegym_categories(text: str) -> List[Tuple[int, int]]:
    """Parse lines containing Clabel/Glabel pairs from FineGym category text."""
    pairs: List[Tuple[int, int]] = []
    for line in text.splitlines():
        if 'Clabel:' not in line or 'Glabel:' not in line:
            continue
        try:
            clabel = int(line.split('Clabel:')[1].split(';')[0].strip())
            glabel = int(line.split('Glabel:')[1].split(';')[0].strip())
            pairs.append((clabel, glabel))
        except Exception:
            continue
    return pairs


def build_gym99_from_gym288_pickle(
    gym288_dataset_path: str,
    gym99_dataset_path: str,
    gym288_categories_url: str = GYM288_CATEGORIES_URL,
    gym99_categories_url: str = GYM99_CATEGORIES_URL,
    allow_neighbor_fallback: bool = True,
) -> Dict[str, int]:
    """
    Convert a Gym288 pickle payload into a Gym99-compatible pickle.

    Mapping is done by matching FineGym glabel between gym288 and gym99.
    When ``allow_neighbor_fallback`` is True, label-1 and label+1 are
    also checked to tolerate off-by-one label indexing in some payloads.

    Returns
    -------
    stats: dict with mapping/match/split counts.
    """
    text_288 = urllib.request.urlopen(gym288_categories_url).read().decode('utf-8')
    text_99 = urllib.request.urlopen(gym99_categories_url).read().decode('utf-8')

    pairs_288 = parse_finegym_categories(text_288)
    pairs_99 = parse_finegym_categories(text_99)
    if not pairs_288 or not pairs_99:
        raise RuntimeError('Cannot parse FineGym category files for gym288/gym99.')

    clabel288_to_glabel = dict(pairs_288)
    glabel_to_clabel99 = {g: c for c, g in pairs_99}

    map_288_to_99: Dict[int, int] = {}
    for clabel_288, glabel in clabel288_to_glabel.items():
        if glabel in glabel_to_clabel99:
            map_288_to_99[clabel_288] = glabel_to_clabel99[glabel]

    with open(gym288_dataset_path, 'rb') as f:
        gym288_payload = pickle.load(f)

    annotations = gym288_payload.get('annotations', [])
    split_info = gym288_payload.get('split', {})

    gym99_annotations = []
    valid_ids = set()
    matched_direct = 0
    matched_minus1 = 0
    matched_plus1 = 0

    for ann in annotations:
        label = int(ann['label'])

        mapped = None
        if label in map_288_to_99:
            mapped = map_288_to_99[label]
            matched_direct += 1
        elif allow_neighbor_fallback and (label - 1) in map_288_to_99:
            mapped = map_288_to_99[label - 1]
            matched_minus1 += 1
        elif allow_neighbor_fallback and (label + 1) in map_288_to_99:
            mapped = map_288_to_99[label + 1]
            matched_plus1 += 1

        if mapped is None:
            continue

        ann_new = dict(ann)
        ann_new['label'] = int(mapped)
        gym99_annotations.append(ann_new)
        valid_ids.add(str(ann_new['frame_dir']))

    gym99_split = {
        'train': [vid for vid in split_info.get('train', []) if vid in valid_ids],
        'test': [vid for vid in split_info.get('test', []) if vid in valid_ids],
    }

    gym99_payload = {
        'split': gym99_split,
        'annotations': gym99_annotations,
    }

    out_dir = os.path.dirname(gym99_dataset_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(gym99_dataset_path, 'wb') as f:
        pickle.dump(gym99_payload, f)

    return {
        'map_size': len(map_288_to_99),
        'annotations_in': len(annotations),
        'annotations_out': len(gym99_annotations),
        'matched_direct': matched_direct,
        'matched_minus1': matched_minus1,
        'matched_plus1': matched_plus1,
        'train_count': len(gym99_split['train']),
        'test_count': len(gym99_split['test']),
    }
