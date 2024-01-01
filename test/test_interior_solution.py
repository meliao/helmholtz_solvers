import pytest
import torch
import numpy as np


from src.interior_solution import LeafNode, Merge
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
        leaf_obj.solve()

        assert leaf_obj.X.shape == (n**2, 4 * n - 4)

        R = leaf_obj.get_R()
        assert R.shape == (4 * n_gauss, 4 * n_gauss)

    def test_1(self) -> None:
        """Tests the differentiation matrices against polynomials in x and y"""

        def f(x: torch.Tensor) -> torch.Tensor:
            """x has shape (n_pts, 2).
            f(x) = 4 * x[:, 0]**4 + 3 * x[:, 1]**2 - 1
            """
            return 4 * torch.pow(x[:, 0], 4) + 3 * torch.square(x[:, 1]) - 1

        def f_x(x: torch.Tensor) -> torch.Tensor:
            """df/dx = 16 * x[:, 0]^3
            d2f/dx2 = 16 * 3 * x[:, 0]^2"""
            return 16 * 3 * torch.square(x[:, 0])

        def f_y(x: torch.Tensor) -> torch.Tensor:
            """df/dy = 6 * x[:, 1]
            d2f/dy2 = 6"""
            return 6 * torch.ones_like(x[:, 1])

        half_side_len = 1.0
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

        f_evals = f(leaf_obj.cheby_quad_obj.points_lst).numpy()
        f_x_evals = f_x(leaf_obj.cheby_quad_obj.points_lst).numpy()
        f_y_evals = f_y(leaf_obj.cheby_quad_obj.points_lst).numpy()

        f_x_pred = leaf_obj.D_x @ f_evals
        f_x_pred = f_x_pred.numpy()

        check_arrays_close(f_x_evals, f_x_pred)

        f_y_pred = leaf_obj.D_y @ f_evals
        f_y_pred = f_y_pred.numpy()

        print(f_y_evals)
        print(f_y_pred)
        check_arrays_close(f_y_evals, f_y_pred)

    @pytest.mark.skip()
    def test_2(self) -> None:
        """"""

        half_side_len = 1.0
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


class Test_Merge:
    def test_0(self) -> None:
        """Tests the a and b boundary indices are set correctly."""
        half_side_len = 1.0
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
        a_bdry_lst = [0, 1, 2, 3]
        b_bdry_lst = [2, 3, 0, 1]
        for a, b in zip(a_bdry_lst, b_bdry_lst):
            merge_obj = Merge(leaf_obj, leaf_obj, a)
            assert merge_obj.b_bdry_idx == b

    def test_1(self) -> None:
        """Tests the compute_merge() method returns without error"""

        """Tests the a and b boundary indices are set correctly."""
        half_side_len = 1.0
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
        q_a = torch.randn(size=(m**2,))
        q_b = torch.randn(size=(m**2,))
        leaf_obj_a = LeafNode(
            half_side_len, n, n_gauss, upper_left_x, upper_left_y, omega, q_a, pts
        )
        leaf_obj_b = LeafNode(
            half_side_len, n, n_gauss, upper_left_x, upper_left_y, omega, q_b, pts
        )

        for i in range(4):
            merge_obj = Merge(leaf_obj_a, leaf_obj_b, i)
            merge_obj.compute_merge()
