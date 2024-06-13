import pytest
import torch
import numpy as np


from src.hps.interior_solution import LeafNode, Merge
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

        sw_corner = torch.tensor([-0.5, -0.5])
        omega = 4.0

        x = torch.linspace(-0.5, 2 * half_side_len + -0.5, m)
        X, Y = torch.meshgrid(x, x, indexing="ij")

        pts = torch.concatenate((X.unsqueeze(-1), Y.unsqueeze(-1)), axis=-1).reshape(
            -1, 2
        )

        q = torch.randn(size=(m**2,))

        leaf_obj = LeafNode(half_side_len, n, n_gauss, sw_corner, omega, q, pts)
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
        sw_corner = torch.tensor([upper_left_x, upper_left_y])
        omega = 4.0

        x = torch.linspace(upper_left_y, 2 * half_side_len + upper_left_y, m)
        X, Y = torch.meshgrid(x, x, indexing="ij")

        pts = torch.concatenate((X.unsqueeze(-1), Y.unsqueeze(-1)), axis=-1).reshape(
            -1, 2
        )

        q = torch.randn(size=(m**2,))

        leaf_obj = LeafNode(half_side_len, n, n_gauss, sw_corner, omega, q, pts)

        f_evals = f(leaf_obj.cheby_quad_interior.points_lst).numpy()
        f_x_evals = f_x(leaf_obj.cheby_quad_interior.points_lst).numpy()
        f_y_evals = f_y(leaf_obj.cheby_quad_interior.points_lst).numpy()

        f_x_pred = leaf_obj.D_x @ f_evals
        f_x_pred = f_x_pred.numpy()

        check_arrays_close(f_x_evals, f_x_pred)

        f_y_pred = leaf_obj.D_y @ f_evals
        f_y_pred = f_y_pred.numpy()

        print(f_y_evals)
        print(f_y_pred)
        check_arrays_close(f_y_evals, f_y_pred)

    @pytest.mark.skip(
        reason="LeafNode.R_indices is now handled by BoundaryQuad object."
    )
    def test_2(self) -> None:
        """Checks the self.R_indices are set correctly"""

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

        expected_indices = torch.arange(4 * n_gauss)
        all_indices = torch.cat(
            (
                leaf_obj.quad_bdry.points_dd["S"],
                leaf_obj.quad_bdry.points_dd["E"],
                leaf_obj.quad_bdry.points_dd["N"],
                leaf_obj.quad_bdry.points_dd["W"],
            )
        )
        print("all_indices shape: ", all_indices.shape)
        print("expected_indices shape: ", expected_indices.shape)

        print("all_indices: ", all_indices)
        print("expected_indices: ", expected_indices)

        assert torch.all(all_indices == expected_indices)

        dirs = ["N", "E", "S", "W"]

        for i in dirs:
            for j in dirs:
                if i == j:
                    continue
                i_idxes = leaf_obj.R_indices[i]
                j_idxes = leaf_obj.R_indices[j]

                # Check that i_indices and j_indices are disjoint
                combined = torch.cat((i_idxes, j_idxes))
                uniques, counts = combined.unique(return_counts=True)
                difference = uniques[counts == 1]
                intersection = uniques[counts > 1]
                assert difference.shape[0] == combined.shape[0]

    def test_3(self) -> None:
        """Checks that the get_R_submatrices() method returns the correct shapes."""

        half_side_len = 1.0
        n = 12
        n_gauss = 13
        m = 15
        upper_left_x = -0.5
        upper_left_y = -0.5
        sw_corner = torch.tensor([upper_left_x, upper_left_y])
        omega = 4.0

        x = torch.linspace(upper_left_y, 2 * half_side_len + upper_left_y, m)
        X, Y = torch.meshgrid(x, x, indexing="ij")

        pts = torch.concatenate((X.unsqueeze(-1), Y.unsqueeze(-1)), axis=-1).reshape(
            -1, 2
        )

        q = torch.randn(size=(m**2,))

        leaf_obj = LeafNode(half_side_len, n, n_gauss, sw_corner, omega, q, pts)

        for i in ["S", "E", "N", "W"]:
            R_intint, R_intext, R_extint, R_extext = leaf_obj.get_R_submatrices(i)

            assert R_intint.shape == (n_gauss, n_gauss)
            assert R_intext.shape == (n_gauss, 3 * n_gauss)
            assert R_extint.shape == (3 * n_gauss, n_gauss)
            assert R_extext.shape == (3 * n_gauss, 3 * n_gauss)


class Test_Merge:
    def test_0(self) -> None:
        """Tests the a and b boundary indices are set correctly."""
        half_side_len = 1.0
        n = 12
        n_gauss = 13
        m = 15
        upper_left_x = -0.5
        upper_left_y = -0.5
        sw_corner = torch.tensor([upper_left_x, upper_left_y])
        omega = 4.0

        x = torch.linspace(upper_left_y, 2 * half_side_len + upper_left_y, m)
        X, Y = torch.meshgrid(x, x, indexing="ij")

        pts = torch.concatenate((X.unsqueeze(-1), Y.unsqueeze(-1)), axis=-1).reshape(
            -1, 2
        )
        q = torch.randn(size=(m**2,))
        leaf_obj = LeafNode(half_side_len, n, n_gauss, sw_corner, omega, q, pts)
        a_bdry_lst = ["S", "E", "N", "W"]
        b_bdry_lst = ["N", "W", "S", "E"]
        for a, b in zip(a_bdry_lst, b_bdry_lst):
            merge_obj = Merge(leaf_obj, leaf_obj, a)
            assert merge_obj.b_bdry_str == b

    def test_1(self) -> None:
        """Tests the compute_merge() method returns without error"""

        """Tests the a and b boundary indices are set correctly."""
        half_side_len = 1.0
        n = 12
        n_gauss = 13
        m = 15
        upper_left_x = -0.5
        upper_left_y = -0.5
        sw_corner = torch.tensor([upper_left_x, upper_left_y])
        omega = 4.0

        x = torch.linspace(upper_left_y, 2 * half_side_len + upper_left_y, m)
        X, Y = torch.meshgrid(x, x, indexing="ij")

        pts = torch.concatenate((X.unsqueeze(-1), Y.unsqueeze(-1)), axis=-1).reshape(
            -1, 2
        )
        q_a = torch.randn(size=(m**2,))
        q_b = torch.randn(size=(m**2,))
        leaf_obj_a = LeafNode(half_side_len, n, n_gauss, sw_corner, omega, q_a, pts)
        leaf_obj_b = LeafNode(half_side_len, n, n_gauss, sw_corner, omega, q_b, pts)

        dirs = ["N", "E", "S", "W"]

        for i in dirs:
            merge_obj = Merge(leaf_obj_a, leaf_obj_b, i)
            merge_obj.compute_merge()
