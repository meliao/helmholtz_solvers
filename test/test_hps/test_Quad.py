import pytest
import torch
import numpy as np


from src.hps.Quad import Quad1D, GaussLegendre1D, Cheby2D, BoundaryQuad
from src.test_utils import check_scalars_close, check_arrays_close


class Test_GaussLegendre1D:
    def test_0(self) -> None:
        """Make sure it initializes without error"""
        d = 0.5
        n = 10
        x = GaussLegendre1D(d, n)

    def test_1(self) -> None:
        """Checks accuracy on f(x) = x^2"""
        d = 0.5
        n = 10
        x = GaussLegendre1D(d, n)

        def f(x):
            return torch.square(x)

        exact = ((0.5) ** 3) / 3 - ((-0.5) ** 3) / 3
        quad = x.eval_func(f).item()
        check_scalars_close(exact, quad)

        evals = f(x.points)
        quad_2 = x.eval_tensor(evals).item()
        check_scalars_close(exact, quad_2)

    def test_2(self) -> None:
        """Checks accuracy on f(x) = e^x"""
        d = 0.25
        n = 10
        x = GaussLegendre1D(d, n)

        def f(x):
            return torch.exp(x)

        exact = np.exp(0.25) - np.exp(-0.25)
        quad = x.eval_func(f).item()
        check_scalars_close(exact, quad)

        evals = f(x.points)
        quad_2 = x.eval_tensor(evals).item()
        check_scalars_close(exact, quad_2)


class Test_BoundaryQuad:
    def test_0(self) -> None:
        """Make sure the class initializes without error."""
        corners = torch.randn(size=(4, 2))
        n_N = 10
        n_S = 11
        n_E = 12
        n_W = 13
        points_S = torch.randn(size=(n_S, 2))
        points_E = torch.randn(size=(n_E, 2))
        points_N = torch.randn(size=(n_N, 2))
        points_W = torch.randn(size=(n_W, 2))
        weights_S = torch.randn(size=(n_S,))
        weights_E = torch.randn(size=(n_E,))
        weights_N = torch.randn(size=(n_N,))
        weights_W = torch.randn(size=(n_W,))

        x = BoundaryQuad(
            corners,
            points_S,
            points_E,
            points_N,
            points_W,
            weights_S,
            weights_E,
            weights_N,
            weights_W,
        )

        assert np.sum(x.n_points) == n_N + n_S + n_E + n_W

    def test_1(self) -> None:
        """Check the initialization from corner and sidelen"""
        sw_corner = torch.Tensor([-0.5, -0.5])
        half_sidelen = 0.5
        n = 10
        x = BoundaryQuad.from_corner_and_half_side_len(sw_corner, half_sidelen, n)

        for side_key in ["S", "E", "N", "W"]:
            assert x.points_dd[side_key].shape == (n, 2)

            # These next two asserts are inequalitites because we don't expect the GL points to lie exactly on the corners.
            assert (
                x.points_dd[side_key].min() >= -0.5
            ), f"side_key: {side_key} and points: {x.points_dd[side_key]}"
            assert (
                x.points_dd[side_key].max() <= 0.5
            ), f"side_key: {side_key} and points: {x.points_dd[side_key]}"

        assert torch.all(x.points_dd["S"][:, 1] == -0.5)
        assert torch.all(x.points_dd["E"][:, 0] == 0.5)
        assert torch.all(x.points_dd["N"][:, 1] == 0.5)
        assert torch.all(x.points_dd["W"][:, 0] == -0.5)

    def test_2(self) -> None:
        """
        Check that get_submatrix_indices returns without error,
        returns the correct shapes, and
        returns disjoint indices
        """
        corners = torch.randn(size=(4, 2))
        n_N = 10
        n_S = 11
        n_E = 12
        n_W = 13
        points_S = torch.randn(size=(n_S, 2))
        points_E = torch.randn(size=(n_E, 2))
        points_N = torch.randn(size=(n_N, 2))
        points_W = torch.randn(size=(n_W, 2))
        weights_S = torch.randn(size=(n_S,))
        weights_E = torch.randn(size=(n_E,))
        weights_N = torch.randn(size=(n_N,))
        weights_W = torch.randn(size=(n_W,))

        x = BoundaryQuad(
            corners,
            points_S,
            points_E,
            points_N,
            points_W,
            weights_S,
            weights_E,
            weights_N,
            weights_W,
        )

        for i, side_key in enumerate(["S", "E", "N", "W"]):
            shared, not_shared = x.get_submatrix_indices(side_key)
            assert shared.shape == (x.n_points[i],)

            assert shared.shape[0] + not_shared.shape[0] == np.sum(x.n_points)

            # Assert the indices are disjoint
            combined = torch.cat((shared, not_shared))
            uniques, counts = combined.unique(return_counts=True)
            difference = uniques[counts == 1]
            intersection = uniques[counts > 1]
            assert difference.shape[0] == combined.shape[0]

    def test_3(self) -> None:
        """Tests the merge function."""
        sw_corner = torch.Tensor([-0.5, -0.5])
        half_sidelen = 0.5
        n = 10
        sw_corner_2 = torch.Tensor([0.1, -0.5])

        x = BoundaryQuad.from_corner_and_half_side_len(sw_corner, half_sidelen, n)
        x_2 = BoundaryQuad.from_corner_and_half_side_len(sw_corner_2, half_sidelen, n)
        z = x.merge(x_2, shared_side_key="E")


class Test_Cheby2D:
    def test_0(self) -> None:
        """Make sure it initializes without error"""
        d = 0.5
        n = 10
        x = Cheby2D(d, n)

    def test_1(self) -> None:
        """Make sure the boundary nodes come first and the interior nodes follow"""

        d = 1.0
        n = 10
        x = Cheby2D(d, n)

        pts = x.points_lst
        pts = torch.abs(pts)

        bdry_pts = pts[: 4 * n - 4]
        int_pts = pts[4 * n - 4 :]

        assert torch.all(torch.logical_or(bdry_pts[:, 0] == d, bdry_pts[:, 1] == d))

        assert torch.logical_not(
            torch.any(torch.logical_or(int_pts[:, 0] == d, int_pts[:, 1] == d))
        )

    def test_2(self) -> None:
        """Make sure self.idxes is filled correctly"""
        d = 1.0
        n = 10
        x = Cheby2D(d, n)

        n_sq = n**2

        idxes = x.idxes

        assert idxes.sum() == 0.5 * (n_sq - 1) * n_sq

    def test_3(self) -> None:
        """Make sure the interpolation method runs without error and returns correct size"""
        d = 1.0
        n = 10
        n_2 = 12

        x = Cheby2D(d, n)

        p = torch.linspace(-1, 1, n_2)

        xx, yy = torch.meshgrid(p, torch.flipud(p), indexing="ij")
        pts = torch.concatenate((xx.unsqueeze(-1), yy.unsqueeze(-1)), axis=-1)
        pts = pts.reshape(-1, 2)

        vals = 2 * pts[:, 0] - 3 * pts[:, 1] + 0.4

        o = x.interp_to_2d_points(pts, vals)

        assert o.shape == (n**2,)

    def test_4(self) -> None:
        """Tests the interpolation method on the affine function f(x,y) = 2 x - 3y + 0.4"""
        d = 1.0
        n = 10
        n_2 = 12

        x = Cheby2D(d, n)

        p = torch.linspace(-1, 1, n_2)

        xx, yy = torch.meshgrid(p, torch.flipud(p), indexing="ij")
        pts = torch.concatenate((xx.unsqueeze(-1), yy.unsqueeze(-1)), axis=-1)
        pts = pts.reshape(-1, 2)

        vals = 2 * pts[:, 0] - 3 * pts[:, 1] + 0.4
        expected_o = (
            2 * x.rasterized_pts_lst[:, 0] - 3 * x.rasterized_pts_lst[:, 1] + 0.4
        )

        o = x.interp_to_2d_points(pts, vals)

        check_arrays_close(expected_o.numpy(), o.numpy())


if __name__ == "__main__":
    pytest.main()
