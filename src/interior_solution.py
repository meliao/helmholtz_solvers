import torch

from src.Quad import GaussLegendre1D, Cheby2D
from src.utils import differentiation_matrix_1d, lagrange_interpolation_matrix


class LeafNode:
    def __init__(
        self,
        half_side_len: float,
        n_cheb_pts: int,
        n_gauss_pts: int,
        upper_left_x: float,
        upper_left_y: float,
        omega: float,
        q: torch.Tensor,
        sample_points: torch.Tensor,
    ) -> None:
        self.half_side_len = half_side_len
        self.n_cheb_pts = n_cheb_pts
        self.n_gauss_pts = n_gauss_pts
        self.upper_left_pos = (upper_left_x, upper_left_y)
        self.omega = omega
        eta = omega
        self.q = q

        self.gauss_quad_obj = GaussLegendre1D(half_side_len, n_gauss_pts)
        self.cheby_quad_obj = Cheby2D(half_side_len, n)

        self.D = (
            1
            / self.half_side_len
            * differentiation_matrix_1d(self.cheby_quad_obj.points_1d)
        )

        # Differentiation wrt x
        self.D_x = torch.kron(self.D, torch.eye(n))
        self.D_x = self.D_x[self.cheby_quad_obj.idxes]
        self.D_x = self.D_x[:, self.cheby_quad_obj.idxes]

        # Differentiation wrt y
        self.D_y = torch.kron(torch.eye(n), self.D)
        self.D_y = self.D_y[self.cheby_quad_obj.idxes]
        self.D_y = self.D_y[:, self.cheby_quad_obj.idxes]

        # Really we want the second derivatives wrt x and y
        self.D_x = self.D_x @ self.D_x
        self.D_y = self.D_y @ self.D_y

        # This is the diagonal of the omega^2(1 - q(x)) operator
        self.diag_ordered = self.omega**2 * (
            1
            - self.cheby_quad_obj.interp_to_2d_points(sample_points, q)[
                self.cheby_quad_obj.idxes
            ]
        )

        # A is the discretization of the inhomogeneous Helmholtz PDE on the Chebyshev grid
        self.A = self.D_x + self.D_y + torch.diag(self.diag_ordered)

        # F is the 4n - 4 x n^2 matrix from eqn 2.9 of GBM15. I am constructing
        # it by constructing N, and then changing the diagonal to make F
        n_boundary_points = 4 * self.n_cheb_pts - 4
        self.F = torch.empty(
            (n_boundary_points, self.n_cheb_pts**2), dtype=torch.cfloat
        )
        self.F[: self.n_cheb_pts] = -1 * self.D_y[: self.n_cheb_pts]
        self.F[self.n_cheb_pts : 2 * self.n_cheb_pts - 1] = self.D_x[
            self.n_cheb_pts : 2 * self.n_cheb_pts - 1
        ]
        self.F[2 * self.n_cheb_pts - 1 : 3 * self.n_cheb_pts - 2] = self.D_y[
            2 * self.n_cheb_pts - 1 : 3 * self.n_cheb_pts - 2
        ]
        self.F[3 * self.n_cheb_pts - 2 : n_boundary_points] = (
            -1 * self.D_x[3 * self.n_cheb_pts - 2 : n_boundary_points]
        )
        N_diag = self.F.diagonal()
        F_diag = N_diag + 1j * eta
        self.F.diagonal().copy_(F_diag)

        # B maps a solution u to the vector [f; 0] where f is incoming impedance data
        self.B = torch.empty(
            (self.n_cheb_pts**2, self.n_cheb_pts**2), dtype=torch.cfloat
        )
        self.B[:n_boundary_points] = self.F
        self.B[n_boundary_points:] = self.A[n_boundary_points:]

        # Interpolate from Gauss to Cheby with last row removed
        self.P_0 = lagrange_interpolation_matrix(
            self.gauss_quad_obj.points, self.cheby_quad_obj.points_1d
        )[:-1]
        self.I_P_0 = torch.kron(torch.eye(4), self.P_0)

        # Interpolate from Cheby to Gauss
        self.Q = lagrange_interpolation_matrix(
            self.cheby_quad_obj.points_1d, self.gauss_quad_obj.points
        )
        self.I_Q = torch.kron(torch.eye(4), self.Q)

        # G maps the solution u to outgoing impedance data
        self.G = torch.empty(
            (4 * self.n_cheb_pts, self.n_cheb_pts**2), dtype=torch.cfloat
        )
        self.G[: self.n_cheb_pts] = -1 * self.D_y[: self.n_cheb_pts]
        self.G[self.n_cheb_pts : 2 * self.n_cheb_pts] = self.D_x[
            self.n_cheb_pts - 1 : 2 * self.n_cheb_pts - 1
        ]
        self.G[2 * self.n_cheb_pts : 3 * self.n_cheb_pts] = self.D_y[
            2 * self.n_cheb_pts - 2 : 3 * self.n_cheb_pts - 2
        ]
        self.G[3 * self.n_cheb_pts : 4 * self.n_cheb_pts] = (
            -1 * self.D_x[3 * self.n_cheb_pts - 3 : 4 * self.n_cheb_pts - 3]
        )

    def solve(self) -> None:
        """Solves the linear system BX = [I; 0] for X. The RHS has size (4 * self.n_cheb_bts - 4, self.n_cheb_pts ** 2) Stores the solution in self.X"""

        RHS = torch.zeros(
            (4 * self.n_cheb_pts - 4, self.n_cheb_pts**2), dtype=torch.cfloat
        )

        X = torch.linalg.solve(self.B, RHS)


class Merge:
    def __init__(self) -> None:
        pass
