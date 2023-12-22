import torch
import numpy as np


def differentiation_matrix_1d(points: torch.Tensor) -> torch.Tensor:
    """Creates a 1-d finite differentiation matrix

    D[i,j] = w_j/ (w_i (x_i - x_j))

    Args:
        points (torch.Tensor): Has shape (p,)

    Returns:
        torch.Tensor: Has shape (p,p)
    """
    p = points.shape[0]

    weights = torch.ones_like(points)
    weights[0] = 1 / 2
    weights[-1] = 1 / 2

    for i in range(0, p, 2):
        weights[i] = weights[i] * -1

    points = points.unsqueeze(-1)
    points_t = points.permute(1, 0)
    diffs = points - points_t
    diffs = 1 / diffs

    weights = weights.unsqueeze(-1)
    weights_square = weights.permute(1, 0) / weights

    out = diffs * weights_square
    return out
