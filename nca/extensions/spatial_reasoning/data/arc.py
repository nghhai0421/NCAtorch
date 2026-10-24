"""ARC input/output pairs from preprocessed NPZ archives on a 30 by 30 canvas.

Stored grids are padded at the bottom and right. Their original (height, width)
values determine the valid input and output rectangles.

Per-sample encoding:
    seed:      (channel_n, 30, 30) - input colors, then zero size/hidden channels.
    condition: (13, 30, 30) - ten colors, input mask, fixed row/column coordinates.
    target:    (11, 30, 30) - ten output colors and the true output mask.
"""

import numpy as np
import torch
import torch.nn.functional as F

from nca.data.datasets.base_dataset import NCADataset

from ..arc_readout import size_logits


class ArcDataset(NCADataset):
    """Load pairs from the ``train_*`` or ``test_*`` arrays of an NPZ file.

    ``rows`` maps dataset indices to archive rows and can be narrowed to select
    tasks. Each sample starts from a fresh input-derived seed on the CPU.
    """

    def __init__(self, path, split, channel_n):
        self.channel_n = channel_n
        self.grid_size = 30
        self.n_colors = 10

        # Fixed Coordinates
        axis = torch.linspace(0, 1, self.grid_size)
        rows, columns = torch.meshgrid(axis, axis, indexing="ij")
        self.coordinates = torch.stack([rows, columns])
        self.palette = torch.tensor([
            [0, 0, 0], [0, 116, 217], [255, 65, 54], [46, 204, 64],
            [255, 220, 0], [170, 170, 170], [240, 18, 190],
            [255, 133, 27], [127, 219, 255], [135, 12, 37],
        ], dtype=torch.float32) / 255

        # Load Pairs
        with np.load(path, allow_pickle=False) as archive:
            self.inputs = archive[f"{split}_input"]
            self.outputs = archive[f"{split}_output"]
            self.input_sizes = archive[f"{split}_input_size"]
            self.output_sizes = archive[f"{split}_output_size"]
            self.task_ids = archive[f"{split}_task_id"]

        self.rows = np.arange(len(self.inputs))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        """Encode one selected pair as ``(seed, condition, target)`` tensors."""
        row = int(self.rows[index])

        seed, condition = self._encode_input(
            self.inputs[row], self.input_sizes[row]
        )
        output_colors, output_mask = self._padded_colors(
            self.outputs[row], self.output_sizes[row]
        )
        target = torch.cat([output_colors, output_mask], dim=0)
        return seed, condition, target

    def _padded_colors(self, grid, size):
        """Return one-hot colors and a valid-cell mask using the original size."""
        grid = torch.as_tensor(grid)
        height, width = map(int, size)

        mask = torch.zeros(1, self.grid_size, self.grid_size)
        mask[:, :height, :width] = 1

        colors = F.one_hot(grid.long(), self.n_colors)
        # Keep real black cells in channel 0 while giving padding zero color features.
        colors = colors.permute(2, 0, 1).float() * mask
        return colors, mask

    def _encode_input(self, grid, size):
        """Build the mutable seed and fixed condition from an input grid."""
        colors, mask = self._padded_colors(grid, size)
        seed = torch.zeros(self.channel_n, self.grid_size, self.grid_size)
        seed[:self.n_colors] = colors
        condition = torch.cat([colors, mask, self.coordinates], dim=0)
        return seed, condition

    def _render_colors(self, scores, valid):
        """Map color classes to RGB, with a gray background outside the mask."""
        color_ids = scores[:, :self.n_colors].argmax(dim=1)
        rgb = self.palette.to(scores.device)[color_ids].permute(0, 3, 1, 2)
        return torch.where(valid[:, None].bool(), rgb, torch.full_like(rgb, 0.95))

    def _render_state(self, state):
        """Render the rectangle selected by the model's categorical size logits."""
        height_logits, width_logits = size_logits(state)
        heights = height_logits.argmax(dim=1) + 1
        widths = width_logits.argmax(dim=1) + 1
        rows = torch.arange(self.grid_size, device=state.device)[None, :, None]
        columns = torch.arange(self.grid_size, device=state.device)[None, None, :]
        valid = (rows < heights[:, None, None]) & (columns < widths[:, None, None])
        return self._render_colors(state, valid)

    def _colorize(self, x, x0=None, target=None, cond=None):
        return self._render_state(x)

    def batch_to_rgb(self, x0, x, target, cond=None):
        """Render input, predicted output, and target using their own extents."""
        input_rgb = self._render_colors(cond[:, :self.n_colors], cond[:, self.n_colors])
        output_rgb = self._render_state(x)
        target_rgb = self._render_colors(target[:, :self.n_colors], target[:, self.n_colors])
        return input_rgb, output_rgb, target_rgb
