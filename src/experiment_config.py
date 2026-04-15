"""
experiment_config.py
Lightweight config loading for training/inference scripts.
"""

import json
from typing import Any, Dict, Iterable, Set


def load_experiment_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError('Experiment config must be a JSON object at top level.')
    return data


def _extract_cli_keys(cli_tokens: Iterable[str]) -> Set[str]:
    keys: Set[str] = set()
    for tok in cli_tokens:
        if not tok.startswith('--'):
            continue
        key = tok[2:].strip()
        if not key:
            continue
        key = key.split('=')[0]
        if key == 'num_wokers':
            key = 'num_workers'
        keys.add(key.replace('-', '_'))
    return keys


def apply_overrides(args, config: Dict[str, Any], cli_tokens: Iterable[str] = ()): 
    """
    Apply config keys only when the CLI argument is still default/empty.
    This keeps CLI as highest-priority override.
    """
    cli_keys = _extract_cli_keys(cli_tokens)
    for key, value in config.items():
        if not hasattr(args, key):
            continue
        if key in cli_keys:
            continue
        setattr(args, key, value)
    return args
