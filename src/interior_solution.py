import torch

from src.Quad import GaussLegendre1D, Cheby2D
from src.utils import differentiation_matrix_1d, lagrange_interpolation_matrix


class Node:
    def __init__(
        self,
        boundary_pts: torch.Tensor,
        R: torch.Tensor = None,
    ) -> None:
        self.boundary_pts = boundary_pts
        self.R = R

    def get_R(self) -> torch.Tensor:
        pass


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
        self.cheby_quad_obj = Cheby2D(half_side_len, self.n_cheb_pts)

        self.D = (
            1
            / self.half_side_len
            * differentiation_matrix_1d(self.cheby_quad_obj.points_1d)
        )

        # Differentiation wrt x
        self.D_x = torch.kron(self.D, torch.eye(self.n_cheb_pts))
        self.D_x = self.D_x[self.cheby_quad_obj.idxes]
        self.D_x = self.D_x[:, self.cheby_quad_obj.idxes]

        # Differentiation wrt y
        self.D_y = torch.kron(torch.eye(self.n_cheb_pts), self.D)
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
        self.I_P_0 = torch.kron(torch.eye(4), self.P_0).to(torch.cfloat)

        # Interpolate from Cheby to Gauss
        self.Q = lagrange_interpolation_matrix(
            self.cheby_quad_obj.points_1d, self.gauss_quad_obj.points
        )
        self.I_Q = torch.kron(torch.eye(4), self.Q).to(torch.cfloat)

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
        G_diag_pre = self.G.diagonal()
        G_diag_post = G_diag_pre - 1j * eta
        self.G.diagonal().copy_(G_diag_post)

        self.X = None
        self.R = None

    def solve(self) -> None:
        """Solves the linear system BX = [I; 0] for X. The RHS has size (4 * self.n_cheb_bts - 4, self.n_cheb_pts ** 2) Stores the solution in self.X"""
        n_cheb_bdry = 4 * self.n_cheb_pts - 4
        RHS = torch.zeros((self.n_cheb_pts**2, n_cheb_bdry), dtype=torch.cfloat)

        RHS[:n_cheb_bdry, :n_cheb_bdry] = torch.eye(n_cheb_bdry)
        self.X = torch.linalg.solve(self.B, RHS)

    def get_R(self) -> torch.Tensor:
        """Computes the solution R = I_Q @ G @ Y. This is from the end of section 2.3 in GMB15

        Output has shape (4 * self.n_gauss_pts, 4 * n_cheb_pts - 4)"""

        if self.R is None:
            if self.X is None:
                self.solve()
            Y = self.X @ self.I_P_0
            # print(Y.shape)
            a = self.G @ Y
            # print(a.shape)
            self.R = self.I_Q @ a
        return self.R


class Merge:
    def __init__(self, node_a: LeafNode, node_b: LeafNode, a_bdry_idx: int) -> None:
        self.node_a = node_a
        self.node_b = node_b

        # These indices are 0 for South edge, 1 for East edge, and continue on counterclockwise around the rectangular domain.
        self.a_bdry_idx = a_bdry_idx
        self.b_bdry_idx = (a_bdry_idx + 2) % 4

        idxes = torch.arange(4 * self.node_a.n_gauss_pts)
        self.a_shared_bool = torch.logical_and(
            idxes >= self.a_bdry_idx * self.node_a.n_gauss_pts,
            idxes < (self.a_bdry_idx + 1) * self.node_a.n_gauss_pts,
        )
        self.b_shared_bool = torch.logical_and(
            idxes >= self.b_bdry_idx * self.node_b.n_gauss_pts,
            idxes < (self.b_bdry_idx + 1) * self.node_b.n_gauss_pts,
        )

        self.R_a = self.node_a.get_R()
        self.R_b = self.node_b.get_R()

        self.R_a_shared = self.R_a[self.a_shared_bool, self.a_shared_bool]

        self.R_b_shared = self.R_b[self.b_shared_bool, self.b_shared_bool]

        # This W will be filled with a matrix inverse in the self._get_W() routine.
        self.W = None

    def _get_W(self) -> None:
        """Fills self.W with the matrix described in section 2.4 of GMB15"""

        self.W = torch.linalg.inv(
            torch.eye(self.node_a.n_gauss_pts) - self.R_b_shared @ self.R_a_shared,
        )

    def compute_merge(self) -> None:
        if self.W is None:
            self._get_W()

        # Need to precompute W R^b_shared R^a_31
        R_a_31 = self.R_a[self.a_shared_bool]
        R_a_31 = R_a_31[:, ~self.a_shared_bool]
        print("Merge.compute_merge: R_a_31 shape:", R_a_31.shape)

        p_1 = self.W @ self.R_b_shared @ R_a_31

        # Need to precompute W R^b_32
        R_b_32 = self.R_b[self.b_shared_bool]
        R_b_32 = R_b_32[:, ~self.b_shared_bool]

        p_2 = self.W @ R_b_32

        # Fill in the operator on the LHS of eqn 2.16 in GBM15
        R = torch.empty((6 * self.node_a.n_gauss_pts, 6 * self.node_b.n_gauss_pts))
