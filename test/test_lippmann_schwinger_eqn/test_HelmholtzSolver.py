import pytest

cola = pytest.importorskip("cola")

import numpy as np
import torch
from src.lippmann_schwinger_eqn.solver_utils import (
    greensfunction2,
    greensfunction3,
    getGscat2circ,
    find_diag_correction,
)
from src.lippmann_schwinger_eqn.HelmholtzSolver import (
    HelmholtzSolverAccelerated,
    setup_accelerated_solver,
    setup_dense_solver,
    HelmholtzSolverDense,
)
from src.test_utils import check_arrays_close, check_scalars_close


N_PIXELS = 50
SPATIAL_DOMAIN_MAX = 0.5
WAVENUMBER = 16
RECEIVER_RADIUS = 100
SOLVER_OBJ = setup_accelerated_solver(
    N_PIXELS, SPATIAL_DOMAIN_MAX, WAVENUMBER, RECEIVER_RADIUS
)
SOLVER_OBJ_BICGSTAB = setup_accelerated_solver(
    N_PIXELS,
    SPATIAL_DOMAIN_MAX,
    WAVENUMBER,
    RECEIVER_RADIUS,
    use_bicgstab=True,
    max_iter_bicgstab=10,
)


class TestSetupMethods:
    def test_0(self) -> None:
        o = setup_accelerated_solver(
            N_PIXELS, SPATIAL_DOMAIN_MAX, WAVENUMBER, RECEIVER_RADIUS
        )
        assert type(o) == HelmholtzSolverAccelerated

        assert SOLVER_OBJ_BICGSTAB.use_bicgstab

    def test_1(self) -> None:
        o = setup_dense_solver(
            N_PIXELS, SPATIAL_DOMAIN_MAX, WAVENUMBER, RECEIVER_RADIUS
        )
        assert type(o) == HelmholtzSolverDense


class TestHelmholtzSolverAccelerated:
    def test_0(self) -> None:
        """Tests _get_uin"""

        source_directions = torch.Tensor([0, np.pi, 3 * np.pi / 2]).to(
            SOLVER_OBJ.device
        )
        n_dirs = source_directions.shape[0]
        o = SOLVER_OBJ._get_uin(source_directions)
        assert o.shape == (n_dirs, SOLVER_OBJ.N**2)

    def test_1(self) -> None:
        # solver_obj = setup_accelerated_solver(
        #     N_PIXELS, SPATIAL_DOMAIN_MAX, WAVENUMBER, RECEIVER_RADIUS
        # )

        n_dirs = 3

        x = torch.randn(
            size=(
                N_PIXELS * N_PIXELS,
                n_dirs,
            )
        ).to(SOLVER_OBJ.device)

        y = SOLVER_OBJ._G_apply(x)

        assert y.shape == x.shape

    def test_2(self) -> None:
        """Tests the accelerated G apply against dense G apply for one-sparse
        input.
        """
        # solver_obj = setup_accelerated_solver(
        #     N_PIXELS, SPATIAL_DOMAIN_MAX, WAVENUMBER, RECEIVER_RADIUS
        # )

        print(SOLVER_OBJ.h)

        diag_correction = find_diag_correction(SOLVER_OBJ.h, SOLVER_OBJ.frequency)

        int_greens_matrix = greensfunction2(
            SOLVER_OBJ.domain_points.numpy(),
            SOLVER_OBJ.frequency,
            diag_correction=diag_correction,
            dx=SOLVER_OBJ.h,
        )

        q_0 = torch.zeros((N_PIXELS**2, 1))
        q_0[0, 0] = 1.0

        out = SOLVER_OBJ._G_apply(q_0).cpu().numpy()

        check_arrays_close(out.flatten(), int_greens_matrix[:, 0].flatten())

    def test_3(self) -> None:
        """Tests the accelerated G apply against dense G apply for random input."""
        # solver_obj = setup_accelerated_solver(
        #     N_PIXELS, SPATIAL_DOMAIN_MAX, WAVENUMBER, RECEIVER_RADIUS
        # )

        print(SOLVER_OBJ.h)

        diag_correction = find_diag_correction(SOLVER_OBJ.h, SOLVER_OBJ.frequency)

        int_greens_matrix = greensfunction2(
            SOLVER_OBJ.domain_points.numpy(),
            SOLVER_OBJ.frequency,
            diag_correction=diag_correction,
            dx=SOLVER_OBJ.h,
        )
        n_dirs = 3
        sigma = torch.randn(size=(N_PIXELS**2, n_dirs))

        out_a = SOLVER_OBJ._G_apply(sigma).cpu().numpy()

        out_b = int_greens_matrix @ sigma.numpy()

        check_arrays_close(out_a, out_b)

    def test_4(self) -> None:
        """Tests that the Helmholtz_solve_exterior routine returns without error
        on a single direction"""

        scattering_obj = np.zeros((N_PIXELS, N_PIXELS))
        z = int(N_PIXELS / 2)
        scattering_obj[z : z + 2, z : z + 2] = np.ones_like(
            scattering_obj[z : z + 2, z : z + 2]
        )

        dirs = np.array(
            [
                np.pi / 2,
            ]
        )
        out = SOLVER_OBJ.Helmholtz_solve_exterior(dirs, scattering_obj)
        assert out.shape == (
            1,
            N_PIXELS,
        )

    def test_5(self) -> None:
        """Tests Helmholtz_solve_interior on a zero scattering potential."""
        # SOLVER_OBJ = setup_accelerated_solver(
        #     N_PIXELS, SPATIAL_DOMAIN_MAX, WAVENUMBER, RECEIVER_RADIUS
        # )

        scattering_obj = np.zeros((N_PIXELS, N_PIXELS))
        dirs = np.array([np.pi / 2, np.pi / 4, 3, 0])
        u_tot, u_in, u_scat = SOLVER_OBJ.Helmholtz_solve_interior(dirs, scattering_obj)

        check_arrays_close(u_tot, u_in)
        check_arrays_close(u_scat, np.zeros_like(u_scat))

    def test_6(self) -> None:
        """Tests that Helmholtz_solve_interior routine returns without error"""

        # SOLVER_OBJ = setup_accelerated_solver(
        #     N_PIXELS, SPATIAL_DOMAIN_MAX, WAVENUMBER, RECEIVER_RADIUS
        # )

        scattering_obj = np.zeros((N_PIXELS, N_PIXELS))
        z = int(N_PIXELS / 2)
        scattering_obj[z : z + 2, z : z + 2] = np.ones_like(
            scattering_obj[z : z + 2, z : z + 2]
        )
        dirs = np.array([np.pi / 4, -np.pi / 4])
        n_dirs = dirs.shape[0]

        out = SOLVER_OBJ.Helmholtz_solve_interior(dirs, scattering_obj)

        for x in out:
            assert x.shape == (n_dirs, N_PIXELS, N_PIXELS)

    def test_7(self) -> None:
        """Tests that the Helmholtz_solve_exterior routine returns without error
        on multiple directions"""

        # SOLVER_OBJ = setup_accelerated_solver(
        #     N_PIXELS, SPATIAL_DOMAIN_MAX, WAVENUMBER, RECEIVER_RADIUS
        # )

        scattering_obj = np.zeros((N_PIXELS, N_PIXELS))
        z = int(N_PIXELS / 2)
        scattering_obj[z : z + 2, z : z + 2] = np.ones_like(
            scattering_obj[z : z + 2, z : z + 2]
        )

        dirs = np.array([np.pi / 2, np.pi / 3, 5 * np.pi / 2])
        n_dirs = dirs.shape[0]
        out = SOLVER_OBJ.Helmholtz_solve_exterior(dirs, scattering_obj)
        assert out.shape == (
            n_dirs,
            N_PIXELS,
        )

    def test_8(self) -> None:
        """Tests the Helmholtz_solve_exterior routine on an all-zero scattering obj"""

        scattering_obj = np.zeros((N_PIXELS, N_PIXELS))
        dirs = np.array([np.pi / 2, np.pi / 3, 5 * np.pi / 2])
        out = SOLVER_OBJ.Helmholtz_solve_exterior(dirs, scattering_obj)
        check_arrays_close(out, np.zeros_like(out))

    def test_9(self) -> None:
        """Tests that the Helmholtz_solve_full method returns without error."""

        scattering_obj = np.random.normal(size=(N_PIXELS, N_PIXELS))
        dirs = np.array([np.pi / 2, np.pi / 3, 5 * np.pi / 2])
        N_dirs = dirs.shape[0]

        out_ext, out_int = SOLVER_OBJ.Helmholtz_solve_full(dirs, scattering_obj)

        assert out_ext.shape == (N_dirs, N_PIXELS)
        assert out_int.shape == (N_dirs, N_PIXELS, N_PIXELS)

    def test_10(self) -> None:
        """Tests that the Helmholtz_solve_full method returns the same thing as
        Helmholtz_solve_interior and Helmholtz_solve_exterior
        """
        scattering_obj = np.random.normal(size=(N_PIXELS, N_PIXELS))
        dirs = np.array([np.pi / 2, np.pi / 3, 5 * np.pi / 2])
        N_dirs = dirs.shape[0]

        out_ext_a, out_int_a = SOLVER_OBJ.Helmholtz_solve_full(dirs, scattering_obj)

        out_ext_b = SOLVER_OBJ.Helmholtz_solve_exterior(dirs, scattering_obj)

        check_arrays_close(out_ext_a, out_ext_b)

        out_int_b, _, _ = SOLVER_OBJ.Helmholtz_solve_interior(dirs, scattering_obj)

        check_arrays_close(out_int_a, out_int_b)

    def test_11(self) -> None:
        """Tests the Helmholtz_solve_exterior routine when using BICGSTAB returns without error
        on multiple directions"""

        scattering_obj = np.zeros((N_PIXELS, N_PIXELS))
        z = int(N_PIXELS / 2)
        scattering_obj[z : z + 2, z : z + 2] = np.ones_like(
            scattering_obj[z : z + 2, z : z + 2]
        )

        dirs = np.array([np.pi / 2, np.pi / 3, 5 * np.pi / 2])
        n_dirs = dirs.shape[0]
        out = SOLVER_OBJ_BICGSTAB.Helmholtz_solve_exterior(dirs, scattering_obj)
        assert out.shape == (
            n_dirs,
            N_PIXELS,
        )

    def test_12(self) -> None:
        """Tests the Helmholtz_solve_exterior routine when using BICGSTAB returns without error
        on multiple directions and multiple scattering objects
        """
        K = 2
        scattering_obj = np.zeros((K, N_PIXELS, N_PIXELS))
        z = int(N_PIXELS / 2)
        scattering_obj[:, z : z + 2, z : z + 2] = np.ones_like(
            scattering_obj[:, z : z + 2, z : z + 2]
        )

        dirs = np.array([np.pi / 2, np.pi / 3, 5 * np.pi / 2])
        n_dirs = dirs.shape[0]
        out = SOLVER_OBJ_BICGSTAB.Helmholtz_solve_exterior(dirs, scattering_obj)
        assert out.shape == (
            K,
            n_dirs,
            N_PIXELS,
        )

    def test_13(self) -> None:
        """Tests the Helmholtz_solve_interior routine when using BICGSTAB returns without error
        on multiple directions and multiple scattering objects
        """
        K = 3
        scattering_obj = np.zeros((K, N_PIXELS, N_PIXELS))
        z = int(N_PIXELS / 2)
        scattering_obj[:, z : z + 2, z : z + 2] = np.ones_like(
            scattering_obj[:, z : z + 2, z : z + 2]
        )

        dirs = np.array([np.pi / 2, np.pi / 3, 5 * np.pi / 2])
        n_dirs = dirs.shape[0]
        out_tot, out_inc, out_scat = SOLVER_OBJ_BICGSTAB.Helmholtz_solve_interior(
            dirs, scattering_obj
        )
        assert out_tot.shape == (K, n_dirs, N_PIXELS, N_PIXELS)
        assert out_inc.shape == (n_dirs, N_PIXELS, N_PIXELS)
        assert out_scat.shape == (K, n_dirs, N_PIXELS, N_PIXELS)


if __name__ == "__main__":
    pytest.main()
