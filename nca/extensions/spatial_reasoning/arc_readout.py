"""Categorical output-size readout and color decoding for the ARC aNCA state.

State channels 0:10 are color logits, 10 contains height scores, and 11
contains width scores. Each size candidate averages a complete canvas row
or column, including positions outside the original input rectangle.
"""

import torch


def size_logits(state):
    """Return height/width logits of shape ``(B, H)`` and ``(B, W)``."""
    height_logits = state[:, 10].float().mean(dim=2)
    width_logits = state[:, 11].float().mean(dim=1)
    return height_logits, width_logits


@torch.no_grad()
def decode_grids(state):
    """Return one top-left cropped integer grid per sample as nested lists."""
    height_logits, width_logits = size_logits(state)
    heights = (height_logits.argmax(dim=1) + 1).tolist()
    widths = (width_logits.argmax(dim=1) + 1).tolist()
    colors = state[:, :10].argmax(dim=1).cpu()
    return [
        colors[index, :height, :width].tolist()
        for index, (height, width) in enumerate(zip(heights, widths))
    ]
