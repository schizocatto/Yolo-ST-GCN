"""
scripts/build_gym99_from_gym288.py
Build Gym99-compatible pickle from Gym288-skeleton pickle.

Usage
-----
python scripts/build_gym99_from_gym288.py \
    --gym288_dataset_path /path/to/gym288_skeleton.pkl \
    --gym99_dataset_path /path/to/gym99_from_gym288.pkl
"""

import argparse
import os
import sys

# Allow running from repo root without installing the package
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.config import GYM288_CATEGORIES_URL, GYM99_CATEGORIES_URL
from src.experiment_config import apply_overrides, load_experiment_config
from src.gym99_builder import build_gym99_from_gym288_pickle


def parse_args():
    p = argparse.ArgumentParser(description='Build Gym99-from-Gym288 pickle')
    p.add_argument('--experiment_config', default='',
                   help='Optional JSON config for frequent experiment updates.')
    p.add_argument('--gym288_dataset_path', required=True, help='Path to gym288_skeleton.pkl')
    p.add_argument('--gym99_dataset_path', required=True, help='Output path for gym99_from_gym288.pkl')
    p.add_argument('--gym288_categories_url', default=GYM288_CATEGORIES_URL,
                   help='FineGym gym288 categories URL')
    p.add_argument('--gym99_categories_url', default=GYM99_CATEGORIES_URL,
                   help='FineGym gym99 categories URL')
    p.add_argument('--disable_neighbor_fallback', action='store_true',
                   help='Disable label-1/label+1 fallback during 288->99 mapping.')
    return p.parse_args()


def main():
    args = parse_args()
    if args.experiment_config:
        cfg = load_experiment_config(args.experiment_config)
        args = apply_overrides(args, cfg, sys.argv[1:])

    stats = build_gym99_from_gym288_pickle(
        gym288_dataset_path=args.gym288_dataset_path,
        gym99_dataset_path=args.gym99_dataset_path,
        gym288_categories_url=args.gym288_categories_url,
        gym99_categories_url=args.gym99_categories_url,
        allow_neighbor_fallback=not args.disable_neighbor_fallback,
    )

    print('Built Gym99 pickle:', args.gym99_dataset_path)
    print('Map size         :', stats['map_size'])
    print('Annotations in   :', stats['annotations_in'])
    print('Annotations out  :', stats['annotations_out'])
    print('Matched direct   :', stats['matched_direct'])
    print('Matched -1       :', stats['matched_minus1'])
    print('Matched +1       :', stats['matched_plus1'])
    print('Train/Test count :', stats['train_count'], stats['test_count'])

    if stats['train_count'] == 0 or stats['test_count'] == 0:
        raise RuntimeError('Gym99 split is empty after mapping.')


if __name__ == '__main__':
    main()
