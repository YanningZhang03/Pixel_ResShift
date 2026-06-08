"""Visualization utilities — pure PyTorch."""

import torch
import numpy as np


def make_grid_visualization(
    vis: torch.Tensor | np.ndarray, grid: int = 8, max_bz: int = 8
) -> np.ndarray:
    """Arrange images into a grid.

    Args:
        vis: (N, C, H, W) or (N, H, W, C) uint8 images.
        grid: number of columns.
        max_bz: max number of grids to return.

    Returns:
        (max_bz, grid_H, grid_W, C) uint8 numpy array.
    """
    if isinstance(vis, torch.Tensor):
        vis = vis.detach().cpu().numpy()

    # Ensure NHWC layout
    if vis.ndim == 4 and vis.shape[1] in (1, 3):
        vis = vis.transpose(0, 2, 3, 1)

    n, h, w, c = vis.shape
    col = grid
    row = min(grid, n // col)
    if n % (col * row) != 0:
        n = col * row * max_bz
        vis = vis[:n]
        n, h, w, c = vis.shape
    assert n % (col * row) == 0

    vis = vis.reshape(-1, col, row * h, w, c)
    vis = np.einsum("mlhwc->mhlwc", vis)
    vis = vis.reshape(-1, row * h, col * w, c)

    bz = min(vis.shape[0], max_bz)
    return vis[:bz]
