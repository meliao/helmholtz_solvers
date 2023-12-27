from typing import Callable
import torch
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from src.utils import chebyshev_points


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


class Cheby2D:
    """This object represents the 2D quadrature of a leaf node specified in Section 2.3 of GBM15. It creates a 2-D Chebyshev quadrature of a square domain with half-side length <spatial_domain_max> using <n>^2 points.

    The special thing is the ordering of the points. Throughout this object's use, it is necessary the points are ordered in a certain way:
     * First the 4(n-1) boundary points, starting in the SW corner and going counter-clockwise.
     * Next all of the n^2 - 4(n-1) interior points. The ordering of these points is not important, as long as it's held consistent.

    The 1-D Chebyshev nodes are defined as x_i = spatial_domain_max * cos(pi * (i - 1) / (n - 1)). We store the list of 2-D Chebyshev points (in the above order) in self.points_lst.



    """

    def __init__(self, spatial_domain_max: float, n: int) -> None:
        # Returns the points, weights for integration over [-1, 1]
        weights = np.ones(n) / n
        # points, weights = np.polynomial.chebyshev.chebgauss(n)

        weights = torch.flipud(spatial_domain_max * torch.from_numpy(weights))
        points, _ = chebyshev_points(n)
        points = spatial_domain_max * points

        # super().__init__(spatial_domain_max, points, weights, n)
        self.spatial_domain_max = spatial_domain_max
        self.points_1d = points
        # print(self.points_1d)
        self.weights_1d = weights
        self.n = n

        self._make_2d_points_and_weights(points, weights)

    def _make_2d_points_and_weights(
        self, points: torch.Tensor, weights: torch.Tensor
    ) -> None:
        n = points.shape[0]
        # Make a list of the interior points
        xx, yy = torch.meshgrid(points, torch.flipud(points), indexing="ij")
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
