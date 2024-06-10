import numpy as np
import torch
import pytest

from src.utils import (
    differentiation_matrix_1d,
    interp_matrix_to_Cheby,
    chebyshev_points,
    lagrange_interpolation_matrix,
    get_incident_plane_waves,
    lst_of_points_to_meshgrid,
    points_to_2d_lst_of_points,
)
from src.test_utils import check_arrays_close, check_scalars_close


class Test_differentiation_matrix_1d:
    def test_0(self) -> None:
        p = torch.arange(10)
        o = differentiation_matrix_1d(p)
        assert o.shape == (10, 10), o.shape

    def test_1(self) -> None:
        """Check n=2 against
        example from MATLAB textbook (T00)
        """

        p, _ = chebyshev_points(2)

        d = differentiation_matrix_1d(p).numpy()

        expected_d = np.array([[1 / 2, -1 / 2], [1 / 2, -1 / 2]])

        print(d)
        print(expected_d)

        check_arrays_close(d, expected_d)

    def test_2(self) -> None:
        """Check n=3 against
        example from MATLAB textbook (T00)
        """

        p, _ = chebyshev_points(3)

        d = differentiation_matrix_1d(p).numpy()

        expected_d = np.array(
            [[3 / 2, -2, 1 / 2], [1 / 2, 0, -1 / 2], [-1 / 2, 2, -3 / 2]]
        )

        print(d)
        print(expected_d)
        print(d - expected_d)

        check_arrays_close(d, expected_d)

    def test_3(self) -> None:
        """Check n=4 against example from MATLAB textbook (T00)"""

        p, _ = chebyshev_points(4)

        d = differentiation_matrix_1d(p).numpy()

        expected_d = np.array(
            [
                [19 / 6, -4, 4 / 3, -1 / 2],
                [1, -1 / 3, -1, 1 / 3],
                [-1 / 3, 1, 1 / 3, -1],
                [1 / 2, -4 / 3, 4, -19 / 6],
            ]
        )

        print(d)
        print(expected_d)
        print(d - expected_d)

        check_arrays_close(d, expected_d)


class Test_interp_matrix_to_Cheby:
    @pytest.mark.skip()
    def test_0(self) -> None:
        """Just checking to make sure code runs without error"""
        p = 13
        points = torch.linspace(-1, 1, p)
        n = 5
        z = interp_matrix_to_Cheby(n, points)
        assert z.shape == (n, p)

    @pytest.mark.skip()
    def test_1(self) -> None:
        """The matrix for n=3 should be a 3x3 identity when the sample points match the Cheby points"""
        n = 3
        cheb_pts, _ = chebyshev_points(n)
        other_pts = torch.linspace(-1, 1, n)
        # out = interp_matrix_to_Cheby(n, other_pts).numpy()
        # print(out)
        # check_arrays_close(out, np.eye(n))

        out_2 = interp_matrix_to_Cheby(n, cheb_pts).numpy()
        print(out_2)
        check_arrays_close(out_2, np.eye(n))

    @pytest.mark.skip()
    def test_2(self) -> None:
        """Interpolating from Cheby points to Cheby points should work perfectly on low-degree polynomials."""

        n = 4
        p = 10

        def l(x: torch.Tensor) -> torch.Tensor:
            # 4 x^3 - 2x^2 + .5
            o = 4 * torch.pow(x, 3) - 2 * torch.square(x) + 1 / 2
            return o

        equispaced_points = torch.linspace(-1, 1, p)
        cheb_points, _ = chebyshev_points(n)

        l_samples_equi = l(equispaced_points)
        l_samples_cheb = l(cheb_points)

        interp_mat = interp_matrix_to_Cheby(n, equispaced_points)

        cheb_interp = interp_mat @ l_samples_equi

        print("Cheb_interp", cheb_interp)
        print("l_samples_cheb", l_samples_cheb)
        print(cheb_interp - l_samples_cheb)

        check_arrays_close(cheb_interp.numpy(), l_samples_cheb.numpy())


class Test_chebyshev_points:
    def test_0(self) -> None:
        n = 2
        p, a = chebyshev_points(n)
        expected_p = np.array([-1, 1.0])
        check_arrays_close(p.numpy(), expected_p)

        expected_a = np.array([np.pi, 0.0])
        print(a)
        print(expected_a)
        check_arrays_close(a.numpy(), expected_a)

    def test_1(self) -> None:
        n = 3
        p, a = chebyshev_points(n)
        expected_p = np.array([-1, 0.0, 1.0])
        expected_a = np.array([np.pi, np.pi / 2, 0])
        print(p)
        print(expected_p)
        check_arrays_close(p.numpy(), expected_p)

        print(a)
        print(expected_a)
        check_arrays_close(a.numpy(), expected_a)


class Test_lagrange_intepolation_matrix:
    def test_0(self) -> None:
        """Checks the function returns without error"""
        n = 10
        p = 7

        x = torch.linspace(-1, 1, p)

        y, _ = chebyshev_points(n)

        out = lagrange_interpolation_matrix(x, y)
        assert out.shape == (n, p)

    def test_1(self) -> None:
        """Interpolating from equispaced to Cheby points should work perfectly on low-degree polynomials."""
        n = 5
        p = 10

        def l(x: torch.Tensor) -> torch.Tensor:
            # 4 x^3 - 2x^2 + .5
            o = 4 * torch.pow(x, 3) - 2 * torch.square(x) + 1 / 2
            return o

        equispaced_points = torch.linspace(-1, 1, p)
        cheb_points, _ = chebyshev_points(n)

        l_samples_equi = l(equispaced_points)
        l_samples_cheb = l(cheb_points)

        interp_mat = lagrange_interpolation_matrix(equispaced_points, cheb_points)

        cheb_interp = interp_mat @ l_samples_equi

        print("Cheb_interp", cheb_interp)
        print("l_samples_cheb", l_samples_cheb)
        print(cheb_interp - l_samples_cheb)

        check_arrays_close(cheb_interp.numpy(), l_samples_cheb.numpy())


class Test_get_incident_plane_waves:
    def test_0(self) -> None:
        """Checks the function get_incident_plane_waves returns without error. Chooses equispaced source directions,
        and then chooses random eval points."""

        n_sources = 17
        n_eval_pts = 100
        frequency = 1.0

        source_dirs = torch.linspace(0, 2 * np.pi, n_sources)
        eval_pts = torch.rand(n_eval_pts, 2)

        out = get_incident_plane_waves(source_dirs, frequency, eval_pts)
        assert out.shape == (n_eval_pts, n_sources)


class Test_lst_of_points_to_meshgrid:
    def test_0(self) -> None:
        """Checks the function returns without error"""
        n = 5
        x = torch.rand(n**2, 2)
        xx, yy = lst_of_points_to_meshgrid(x)
        assert xx.shape == (n, n)
        assert yy.shape == (n, n)

    def test_1(self) -> None:

        x = np.linspace(0, 1, 4)
        X, Y = np.meshgrid(x, np.flipud(x), indexing="ij")
        pts_lst = points_to_2d_lst_of_points(torch.from_numpy(x))
        xx, yy = lst_of_points_to_meshgrid(pts_lst)
        check_arrays_close(xx.numpy(), X)
        check_arrays_close(yy.numpy(), Y)


if __name__ == "__main__":
    pytest.main()
