from typing import Callable, Tuple
import torch
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from src.utils import chebyshev_points, lst_of_points_to_meshgrid


class Quad1D:
    def __init__(
        self,
        spatial_domain_max: float,
        points: torch.Tensor,
        weights: torch.Tensor,
        n: int,
    ) -> None:
        """Integrates on the 1-D interval [-spatial_domain_max, spatial_domain_max]"""
        self.spatial_domain_max = spatial_domain_max
        self.points = points
        self.weights = weights
        self.n = n

    def eval_func(self, f: Callable[[torch.Tensor], torch.Tensor]) -> float:
        evals = f(self.points)
        return self.eval_tensor(evals)

    def eval_tensor(self, evals: torch.Tensor) -> float:
        return torch.dot(evals, self.weights)


class GaussLegendre1D(Quad1D):
    def __init__(self, spatial_domain_max: float, n: int) -> None:
        # Returns the points, weights for integration over [-1, 1]
        points, weights = np.polynomial.legendre.leggauss(n)

        weights = spatial_domain_max * torch.from_numpy(weights)
        points = spatial_domain_max * torch.from_numpy(points)

        super().__init__(spatial_domain_max, points, weights, n)


class BoundaryQuad:
    def __init__(
        self,
        corners: torch.Tensor,
        points_S: torch.Tensor,
        points_E: torch.Tensor,
        points_N: torch.Tensor,
        points_W: torch.Tensor,
        weights_S: torch.Tensor,
        weights_E: torch.Tensor,
        weights_N: torch.Tensor,
        weights_W: torch.Tensor,
        quad_obj: GaussLegendre1D = None,
    ) -> None:
        self.corners = corners
        self.points_dd = {
            "S": points_S,
            "E": points_E,
            "N": points_N,
            "W": points_W,
        }
        self.weights_dd = {
            "S": weights_S,
            "E": weights_E,
            "N": weights_N,
            "W": weights_W,
        }
        self._dirs = ["S", "E", "N", "W"]

        self.n_points = [
            points_S.shape[0],
            points_E.shape[0],
            points_N.shape[0],
            points_W.shape[0],
        ]
        self.quad_obj = quad_obj

    def get_submatrix_indices(
        self, shared_side_key: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """The boundary points are ordered to start on the South edge, and then wrap around counter-clockwise. This function returns the indices of the boundary points that are shared
        along the specified edge, and the indices of the boundary points that are not shared.
        """
        idxes = torch.arange(
            self.n_points[0] + self.n_points[1] + self.n_points[2] + self.n_points[3]
        )

        if shared_side_key == "S":
            shared_side_idxes = idxes[: self.n_points[0]]
            non_shared_idxes = idxes[self.n_points[0] :]

        elif shared_side_key == "E":
            shared_side_idxes = idxes[
                self.n_points[0] : self.n_points[0] + self.n_points[1]
            ]
            non_shared_idxes = torch.cat(
                (
                    idxes[: self.n_points[0]],
                    idxes[self.n_points[0] + self.n_points[1] :],
                )
            )
        elif shared_side_key == "N":
            shared_side_idxes = idxes[
                self.n_points[0]
                + self.n_points[1] : self.n_points[0]
                + self.n_points[1]
                + self.n_points[2]
            ]
            non_shared_idxes = torch.cat(
                (
                    idxes[: self.n_points[0] + self.n_points[1]],
                    idxes[self.n_points[0] + self.n_points[1] + self.n_points[2] :],
                )
            )
        elif shared_side_key == "W":
            shared_side_idxes = idxes[-self.n_points[3] :]
            non_shared_idxes = idxes[: -self.n_points[3]]
        else:
            raise ValueError("shared_side_key must be one of ['S', 'E', 'N', 'W']")

        return shared_side_idxes, non_shared_idxes

    def merge(self, other: "BoundaryQuad", shared_side_key: str) -> "BoundaryQuad":
        if shared_side_key == "S":
            new_points_S = other.points_dd["S"]
            new_weights_S = other.weights_dd["S"]

            new_points_E = torch.cat((other.points_dd["E"], self.points_dd["E"]))
            new_weights_E = torch.cat((other.weights_dd["E"], self.weights_dd["E"]))

            new_points_N = self.points_dd["N"]
            new_weights_N = self.weights_dd["N"]

            new_points_W = torch.cat((self.points_dd["W"], other.points_dd["W"]))
            new_weights_W = torch.cat((self.weights_dd["W"], other.weights_dd["W"]))

        elif shared_side_key == "E":
            pass
        elif shared_side_key == "N":
            pass
        elif shared_side_key == "W":
            pass
        else:
            raise ValueError("shared_side_key must be one of ['S', 'E', 'N', 'W']")

    @classmethod
    def from_corner_and_half_side_len(
        cls, southwest_corner: torch.Tensor, half_side_len: float, n: int
    ):
        # First, define the four corners in an array of shape (4,2). The corners are ordered counter-clockwise starting from the SW corner.
        corners = torch.zeros(4, 2)
        corners[0] = southwest_corner
        corners[1] = southwest_corner + torch.tensor(
            [2 * half_side_len, 0.0]
        )  # SE corner
        corners[2] = southwest_corner + torch.tensor(
            [2 * half_side_len, 2 * half_side_len]
        )  # NE corner
        corners[3] = southwest_corner + torch.tensor(
            [0.0, 2 * half_side_len]
        )  # NW corner

        # Next, get a 1D Gauss-Legendre quadrature object
        quad_1d = GaussLegendre1D(half_side_len, n)
        quad_points = quad_1d.points + half_side_len  # points are from 0 to side_len
        quad_weights = quad_1d.weights

        # Now, create the points arrays for each of the four sides.

        S_points = torch.zeros(n, 2)
        S_points[:, 0] = corners[0, 0] + quad_points
        S_points[:, 1] = corners[0, 1]

        E_points = torch.zeros(n, 2)
        E_points[:, 0] = corners[1, 0]
        E_points[:, 1] = corners[1, 1] + quad_points

        N_points = torch.zeros(n, 2)
        N_points[:, 0] = corners[2, 0] - quad_points
        N_points[:, 1] = corners[2, 1]

        W_points = torch.zeros(n, 2)
        W_points[:, 0] = corners[3, 0]
        W_points[:, 1] = corners[3, 1] - quad_points

        # initialize the object
        return cls(
            corners,
            S_points,
            E_points,
            N_points,
            W_points,
            quad_weights,
            quad_weights,
            quad_weights,
            quad_weights,
            quad_obj=quad_1d,
        )


class Cheby2D:
    """This object represents the 2D quadrature of a leaf node specified in Section 2.3 of GBM15.
    It creates a 2-D Chebyshev quadrature of a square domain with half-side length <spatial_domain_max>
    using <n>^2 points.

    The special thing is the ordering of the points. Throughout this object's use, it is necessary the
    points are ordered in a certain way:
     * First the 4(n-1) boundary points, starting in the SW corner and going counter-clockwise.
     * Next all of the n^2 - 4(n-1) interior points. The ordering of these points is not important, as
     long as it's held consistent.

    We are thinking about the square leaf box with upper-left corner at
    (center_x - spatial_domain_max, center_y + spatial_domain_max) and lower-right corner
    (center_x + spatial_domain_max, center_y - spatial_domain_max).

    The 1-D Chebyshev nodes are defined as x_i = spatial_domain_max * cos(pi * (i - 1) / (n - 1)). We store the list of
    2-D Chebyshev points (in the above order) in self.points_lst.
    """

    def __init__(
        self,
        spatial_domain_max: float,
        n: int,
        center_x: float = 0.0,
        center_y: float = 0.0,
    ) -> None:
        # Returns the points, weights for integration over [-1, 1]
        weights = np.ones(n) / n
        # points, weights = np.polynomial.chebyshev.chebgauss(n)

        weights = torch.flipud(spatial_domain_max * torch.from_numpy(weights))
        points, _ = chebyshev_points(n)
        points = spatial_domain_max * points

        # super().__init__(spatial_domain_max, points, weights, n)
        self.center_x = center_x
        self.center_y = center_y
        self.spatial_domain_max = spatial_domain_max
        self.points_1d = points
        # print(self.points_1d)
        self.weights_1d = weights
        self.n = n

        self._make_2d_points_and_weights(self.points_1d, weights)

    def _make_2d_points_and_weights(
        self, points: torch.Tensor, weights: torch.Tensor
    ) -> None:
        n = points.shape[0]
        # Make a list of the interior points
        xx, yy = torch.meshgrid(points, torch.flipud(points), indexing="ij")
        xx = self.center_x + xx
        yy = self.center_y + yy
        pts = torch.concatenate((xx.unsqueeze(-1), yy.unsqueeze(-1)), axis=-1)
        # print("pts shape: ", pts.shape)
        pts = pts.reshape(-1, 2)
        # print("pts shape: ", pts.shape)

        # pts is like the 2D grid in column-rasterized format.

        # The idxes array is the array of indices that
        # re-orders the 2D grid col-rasterized <pts> array into the ordering specified by GBM15
        idxes = torch.zeros(n**2, dtype=int)
        # S border
        for i, j in enumerate(range(n - 1, n**2, n)):
            idxes[i] = j
        # W border
        for i, j in enumerate(range(n**2 - 2, n**2 - n - 1, -1)):
            idxes[n + i] = j
        # N border
        for i, j in enumerate(range(n**2 - 2 * n, 0, -n)):
            idxes[2 * n - 1 + i] = j
        # S border
        for i, j in enumerate(range(1, n - 1)):
            idxes[3 * n - 2 + i] = j
        # Loop through the indices in column-rasterized form and fill in the ones from the interior.
        current_idx = 4 * n - 4
        nums = torch.arange(n**2)
        for i in nums:
            if i not in idxes:
                idxes[current_idx] = i
                current_idx += 1
            else:
                continue
        self.idxes = idxes
        self.points_lst = pts[idxes]
        self.rasterized_pts_lst = pts

    def eval_func(self, f: Callable[[torch.Tensor], torch.Tensor]) -> float:
        evals = f(self.points_lst)
        return self.eval_tensor(evals)

    def eval_tensor(self, evals: torch.Tensor) -> float:
        return torch.dot(evals, self.w_lst)

    def interp_to_2d_points(
        self, ref_points: torch.Tensor, ref_vals: torch.Tensor
    ) -> torch.Tensor:
        """This function uses scipy's linear 2D interpolation to interpolate from <ref_points> to the 2d chebyshev points.

        TOOD: I think this operation can be written as a linear operator in pure pytorch.

        Args:
            ref_points (torch.Tensor): The points that the function is evaluated on. Has shape (N,2)
            ref_vals (torch.Tensor): Function evaluations. Has shape (N,)

        Returns:
            torch.Tensor: Has shape (self.n ** 2)
        """
        points_np = ref_points.numpy()
        vals_np = ref_vals.numpy()

        interp_obj = LinearNDInterpolator(points_np, vals_np)

        X, Y = torch.meshgrid(
            self.points_1d, torch.flipud(self.points_1d), indexing="ij"
        )

        out = interp_obj(X.numpy(), Y.numpy())
        return torch.from_numpy(out).reshape(-1)

    def interp_from_2d_points(
        self, ref_points: torch.Tensor, ref_vals: torch.Tensor
    ) -> torch.Tensor:
        """This function uses scipy's linear 2D interpolation to interpolate from the 2d chebyshev points to <ref_points>.

        TOOD: I think this operation can be written as a linear operator in pure pytorch.

        Args:
            ref_points (torch.Tensor): The points that we want the function evaluated on. Has shape (N_ref,2)
            ref_vals (torch.Tensor): Function evaluations on the 2D chebyshev grid stored at
                                        self.points_lst. Has shape (N,)

        Returns:
            torch.Tensor: Has shape (N_ref,)
        """

        points_np = self.points_lst.numpy()
        vals_np = ref_vals.numpy()

        interp_obj = LinearNDInterpolator(points_np, vals_np)

        xx, yy = lst_of_points_to_meshgrid(ref_points)
        out = interp_obj(xx.numpy(), yy.numpy())
        return torch.from_numpy(out)
