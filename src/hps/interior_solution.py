from typing import Tuple
import torch
import math
from src.hps.Quad import GaussLegendre1D, Cheby2D
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
        center_x = upper_left_x + half_side_len
        center_y = upper_left_y - half_side_len
        self.omega = omega
        eta = omega
        self.q = q

        self.gauss_quad_obj = GaussLegendre1D(half_side_len, n_gauss_pts)
        self.cheby_quad_obj = Cheby2D(
            half_side_len, self.n_cheb_pts, center_x, center_y
        )

        # norm_factor = 1 / (math.sqrt(self.half_side_len))
        norm_factor = 1
        print("norm_factor: ", norm_factor)
        self.D = differentiation_matrix_1d(self.cheby_quad_obj.points_1d) * norm_factor

        # Differentiation wrt x
        self.D_x = torch.kron(self.D, torch.eye(self.n_cheb_pts))
        self.D_x = self.D_x[self.cheby_quad_obj.idxes]
        self.D_x = self.D_x[:, self.cheby_quad_obj.idxes]
        self.D_x_single = self.D_x

        # Differentiation wrt y
        self.D_y = torch.kron(torch.eye(self.n_cheb_pts), self.D)
        self.D_y = self.D_y[self.cheby_quad_obj.idxes]
        self.D_y = self.D_y[:, self.cheby_quad_obj.idxes]
        self.D_y_single = self.D_y

        # Really we want the second derivatives wrt x and y
        self.D_x = self.D_x @ self.D_x
        self.D_y = self.D_y @ self.D_y

        # self.D_x = norm_factor * self.D_x
        # self.D_y = norm_factor * self.D_y

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

        self._dirs = ["S", "E", "N", "W"]

        # This ordering comes from the first sentence of section 2.3 in GMB15
        self.R_indices = {
            "S": torch.arange(self.n_gauss_pts),
            "E": torch.arange(self.n_gauss_pts, 2 * self.n_gauss_pts),
            "N": torch.arange(2 * self.n_gauss_pts, 3 * self.n_gauss_pts),
            "W": torch.arange(3 * self.n_gauss_pts, 4 * self.n_gauss_pts),
        }

    def solve(self) -> None:
        """Solves the linear system BX = [I; 0] for X. The RHS has size
        (4 * self.n_cheb_bts - 4, self.n_cheb_pts ** 2)
        Stores the solution in self.X
        """
        n_cheb_bdry = 4 * self.n_cheb_pts - 4
        RHS = torch.zeros((self.n_cheb_pts**2, n_cheb_bdry), dtype=torch.cfloat)

        RHS[:n_cheb_bdry, :n_cheb_bdry] = torch.eye(n_cheb_bdry)
        self.X = torch.linalg.solve(self.B, RHS)

    def get_R(self) -> torch.Tensor:
        """Computes the solution R = I_Q @ G @ Y. Conceptually, R maps the incoming impedance data
        to the outgoing impedance data. In the paper, this is written as g = Rf.
        This is from the end of section 2.3 in GMB15.

        Output has shape (4 * self.n_gauss_pts, 4 * n_cheb_pts)"""

        if self.R is None:
            if self.X is None:
                self.solve()
            Y = self.X @ self.I_P_0
            # print(Y.shape)
            a = self.G @ Y
            # print(a.shape)
            self.R = self.I_Q @ a
        return self.R

    def get_R_submatrices(self, shared_side_key: str) -> torch.Tensor:
        """Returns four submatrices of the R matrix. The R matrix maps incoming impedance data to outgoing impedance data.
        To describe the submatrices, think about the indices tabulating the boundary points of two adjacent nodes, which we wish
        to join. The indices start on the South edge and go around counter-clockwise.
        The submatrices are:

        R = (R_intint, R_intext
             R_extint, R_extext)

        The shared_side_key indicates which side of the node is shared with the adjacent node, and therefore which rows and cols are
        included in the int/ext submatrices.
        """
        assert shared_side_key in self._dirs, f"idx_rows must be in {self._dirs}"

        if self.R is None:
            self.get_R()

        int_idxes = self.R_indices[shared_side_key]
        ext_indices = torch.cat(
            [self.R_indices[dir] for dir in self._dirs if dir != shared_side_key]
        )

        R_intint = self.R[int_idxes][:, int_idxes]
        R_intext = self.R[int_idxes][:, ext_indices]
        R_extint = self.R[ext_indices][:, int_idxes]
        R_extext = self.R[ext_indices][:, ext_indices]

        return R_intint, R_intext, R_extint, R_extext


def _get_opposite_boundary_str(bdry_str: str) -> str:
    """Returns the string for the boundary opposite to bdry_str"""
    bdry_dict = {"N": "S", "E": "W", "S": "N", "W": "E"}
    return bdry_dict[bdry_str]


class Merge:

    def __init__(self, node_a: LeafNode, node_b: LeafNode, a_bdry_str: str) -> None:
        self.node_a = node_a
        self.node_b = node_b

        # These indices are 0 for South edge, 1 for East edge, and continue on counterclockwise around the rectangular domain.
        self.a_bdry_str = a_bdry_str
        self.b_bdry_str = _get_opposite_boundary_str(a_bdry_str)

        idxes = torch.arange(4 * self.node_a.n_gauss_pts)

        self.R_a = self.node_a.get_R()
        self.R_b = self.node_b.get_R()
        print("Merge.__init__: R_a shape:", self.R_a.shape)
        print("Merge.__init__: R_b shape:", self.R_b.shape)

        (self.R_a_33, self.R_a_31, self.R_a_13, self.R_a_11) = (
            self.node_a.get_R_submatrices(self.a_bdry_str)
        )

        (self.R_b_33, self.R_b_32, self.R_b_23, self.R_b_22) = (
            self.node_b.get_R_submatrices(self.b_bdry_str)
        )

        # This W will be filled with a matrix inverse in the self._get_W() routine.
        self.W = None

    def _get_W(self) -> None:
        """Fills self.W with the matrix described in section 2.4 of GMB15"""

        print("Merge._get_W: self.R_b_33 shape:", self.R_b_33.shape)
        print("Merge._get_W: self.R_a_33 shape:", self.R_a_33.shape)
        print("Merge._get_W: self.node_a.n_gauss_pts:", self.node_a.n_gauss_pts)
        self.W = torch.linalg.inv(
            torch.eye(self.node_a.n_gauss_pts) - self.R_b_33 @ self.R_a_33,
        )

    def compute_merge(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes the merge of the two nodes.
        Returns R, S_a, and S_b.
        R is the operator that maps incoming impedance data to outgoing impedance data on the merged node.
        S_a maps incoming impedance data on the merged node to incoming impedance data on the shared edge in node a.
        S_b maps incoming impedance data on the merged node to incoming impedance data on the shared edge in node b.

        R has shape (6 * self.node_a.n_gauss_pts, 6 * self.node_b.n_gauss_pts)
        S_a and S_b have shape (self.node_a.n_gauss_pts, 6 * self.node_a.n_gauss_pts)
        """
        if self.W is None:
            self._get_W()

        # Need to precompute W R^b_shared R^a_31
        # print("Merge.compute_merge: self.R_a_31 shape:", self.R_a_31.shape)

        # print("Merge.compute_merge: self.W shape:", self.W.shape)
        # print("Merge.compute_merge: self.R_a_31 shape:", self.R_a_31.shape)

        p_1 = self.W @ self.R_b_33 @ self.R_a_31

        # Need to precompute W R^b_32
        # R_b_32 = self.R_b[self.b_shared_bool]
        # R_b_32 = R_b_32[:, ~self.b_shared_bool]

        # R_b_23 = self.R_b[~self.b_shared_bool]
        # R_b_23 = R_b_23[:, self.b_shared_bool]

        p_2 = self.W @ self.R_b_32

        # Fill in the operator on the LHS of eqn 2.16 in GBM15
        R = torch.empty(
            (6 * self.node_a.n_gauss_pts, 6 * self.node_b.n_gauss_pts),
            dtype=self.node_a.R.dtype,
        )

        print("Merge.compute_merge: R shape:", R.shape)
        print("Merge.compute_merge: p_1 shape:", p_1.shape)
        print("Merge.compute_merge: R_a_13 shape:", self.R_a_13.shape)

        pre_1 = self.R_a_11 + self.R_a_13 @ p_1
        R[: 3 * self.node_a.n_gauss_pts, : 3 * self.node_a.n_gauss_pts] = pre_1

        pre_2 = -1 * self.R_a_13 @ p_2
        R[: 3 * self.node_a.n_gauss_pts, 3 * self.node_a.n_gauss_pts :] = pre_2

        pre_3a = self.R_a_31 + self.R_a_33 @ p_1
        pre_3 = -1 * self.R_b_23 @ pre_3a
        R[3 * self.node_a.n_gauss_pts :, : 3 * self.node_a.n_gauss_pts] = pre_3

        pre_4 = self.R_b_22 + self.R_b_23 @ self.R_a_33 @ p_2
        R[3 * self.node_a.n_gauss_pts :, 3 * self.node_a.n_gauss_pts :] = pre_4

        # Fill in the S_a and S_b matrices
        S_a = torch.empty(
            (self.node_a.n_gauss_pts, 6 * self.node_a.n_gauss_pts),
            dtype=self.node_a.R.dtype,
        )
        S_a[:, : 3 * self.node_a.n_gauss_pts] = p_1
        S_a[:, 3 * self.node_a.n_gauss_pts :] = -1 * p_2

        S_b = torch.empty(
            (self.node_a.n_gauss_pts, 6 * self.node_a.n_gauss_pts),
            dtype=self.node_a.R.dtype,
        )
        S_b[:, : 3 * self.node_a.n_gauss_pts] = -1 * pre_3a
        S_b[:, 3 * self.node_a.n_gauss_pts :] = -1 * self.R_a_33 @ p_2

        return R, S_a, S_b


def build_interior_dtn_map(node: Node, eta: float) -> torch.Tensor:
    """Builds the Dirichlet-to-Neumann map for a node.
    It builds the DtN operator from the ItI operator,  which is called  R.
    This uses equation 2.17 from GBM15.
    """
    R = node.get_R()
    n_bdry_points = R.shape[0]
    prefactor = 1j * eta * -1
    I = torch.eye(n_bdry_points)
    X = torch.linalg.solve(R - I)
    DtN = prefactor * X @ (R + I)

    return DtN
