import pytest
import torch
import numpy as np


from src.hps.Quad import Quad1D, GaussLegendre1D, Cheby2D
from src.test_utils import check_scalars_close, check_arrays_close


class Test_Quad1D:
    def test_0(self) -> None:
        pass


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
