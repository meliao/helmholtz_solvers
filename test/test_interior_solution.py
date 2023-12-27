import pytest
import torch
import numpy as np


from src.interior_solution import LeafNode
from src.test_utils import (
    check_arrays_close,
    check_no_nan_in_array,
    check_scalars_close,
)


class Test_LeafNode:
    def test_0(self) -> None:
        """Make sure it initializes without error"""
        half_side_len = 0.5
        n = 12
        n_gauss = 13
        m = 15
        upper_left_x = -0.5
        upper_left_y = -0.5
        omega = 4.0

        x = torch.linspace(upper_left_y, 2 * half_side_len + upper_left_y, m)
        X, Y = torch.meshgrid(x, x, indexing="ij")

        pts = torch.concatenate((X.unsqueeze(-1), Y.unsqueeze(-1)), axis=-1).reshape(
            -1, 2
        )

        q = torch.randn(size=(m**2,))

        leaf_obj = LeafNode(
            half_side_len, n, n_gauss, upper_left_x, upper_left_y, omega, q, pts
        )
