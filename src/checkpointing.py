"""
checkpointing.py
Metadata-aware checkpoint save/load helpers.
"""

from typing import Any, Dict, Tuple

import torch


def save_checkpoint(path: str, model: torch.nn.Module, metadata: Dict[str, Any]) -> None:
    payload = {
        'model_state_dict': model.state_dict(),
        'metadata': metadata,
    }
    torch.save(payload, path)


def load_checkpoint(path: str, map_location=None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    payload = torch.load(path, map_location=map_location)
    # Backward-compatible: old files may be raw state_dict
    if isinstance(payload, dict) and 'model_state_dict' in payload:
        return payload['model_state_dict'], payload.get('metadata', {})
    return payload, {}
