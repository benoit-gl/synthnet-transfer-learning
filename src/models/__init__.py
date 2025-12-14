"""Model modules for image classification with transfer learning."""

import os
from collections import OrderedDict
from typing import Optional

import torch
from torch import nn


def load_checkpoint_weights(
    model: nn.Module,
    checkpoint_path: str,
    prefix: str = "net.",
    strict: bool = False,
) -> None:
    """Load pretrained weights from a Lightning checkpoint into a model.

    Lightning checkpoints store state_dict with module names prefixed (e.g., "net.layer1...").
    HuggingFace models loaded via from_pretrained() don't have this prefix.
    This function handles the prefix stripping and validates the checkpoint exists.

    Args:
        model: The model to load weights into (typically a HuggingFace model).
        checkpoint_path: Path to a PyTorch Lightning .ckpt file.
        prefix: The prefix to strip from state_dict keys. Default "net.".
        strict: Whether to strictly enforce that the keys match. Default False
                to allow classifier head size differences in fine-tuning.

    Raises:
        FileNotFoundError: If checkpoint_path does not exist.
        KeyError: If the checkpoint doesn't contain 'state_dict'.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"CH-FT experiments require a pre-trained checkpoint from a CH (classifier head) run."
        )

    checkpoint = torch.load(checkpoint_path, weights_only=False)

    if "state_dict" not in checkpoint:
        raise KeyError(
            f"Checkpoint at {checkpoint_path} does not contain 'state_dict'. "
            f"Expected a PyTorch Lightning checkpoint."
        )

    weights = checkpoint["state_dict"]
    weights_cleaned = OrderedDict()

    prefix_len = len(prefix)
    for name, param in weights.items():
        if name.startswith(prefix):
            weights_cleaned[name[prefix_len:]] = param

    if not weights_cleaned:
        raise ValueError(
            f"No weights found with prefix '{prefix}' in checkpoint. "
            f"Available keys start with: {list(weights.keys())[:5]}"
        )

    model.load_state_dict(weights_cleaned, strict=strict)

