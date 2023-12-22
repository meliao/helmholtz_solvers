from typing import Callable
import torch
import numpy as np


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
        points = np.cos(np.pi / (n - 1) * np.arange(n))
        weights = np.ones(n) / n
        # points, weights = np.polynomial.chebyshev.chebgauss(n)

        weights = torch.flipud(spatial_domain_max * torch.from_numpy(weights))
        points = torch.flipud(spatial_domain_max * torch.from_numpy(points))

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
        print("pts shape: ", pts.shape)
        pts = pts.reshape(-1, 2)
        print("pts shape: ", pts.shape)

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
        # print(nums)
        # idxes_lst = idxes.tolist()
        # print(idxes_lst)
        for i in nums:
            if i not in idxes:
                idxes[current_idx] = i
                current_idx += 1
            else:
                continue
        self.idxes = idxes
        self.points_lst = pts[idxes]
        self.rasterized_pts_lst = pts

    # def _make_2d_points_and_weights(
    #     self, points: torch.Tensor, weights: torch.Tensor
    # ) -> None:
    #     n = points.shape[0]
    #     # Make a list of the interior points
    #     interior_x, interior_y = torch.meshgrid(points[1:-1], points[1:-1])
    #     interior_pts = torch.concatenate(
    #         (interior_x.unsqueeze(-1), interior_y.unsqueeze(-1)), axis=-1
    #     ).reshape(-1, 2)
    #     # Boundary points start in the SW corner of the square
    #     # domain and go around counter clockwise
    #     boundary_pts = torch.empty((4 * (self.n - 1), 2), dtype=torch.float32)
    #     # S edge
    #     boundary_pts[:n, 0] = points
    #     boundary_pts[:n, 1] = points[0]
    #     # E edge
    #     boundary_pts[n : 2 * n, 0] = points[-1]
    #     boundary_pts[n : 2 * n - 1, 1] = points[1:]
    #     # N edge
    #     boundary_pts[2 * n - 1 : 3 * n - 2, 0] = torch.flipud(points[:-1])
    #     boundary_pts[2 * n - 1 : 3 * n - 2, 1] = points[-1]
    #     # W edge
    #     boundary_pts[3 * n - 2 :, 0] = points[0]
    #     boundary_pts[3 * n - 2 :, 1] = torch.flipud(points[1:-1])

    #     self.boundary_pts_lst = boundary_pts
    #     self.interior_pts_lst = interior_pts
    #     self.points_lst = torch.concatenate((boundary_pts, interior_pts), axis=0)

    #     # Do the same thing with the weights. First a list of the interior weights
    #     interior_w_x, interior_w_y = torch.meshgrid(weights[1:-1], weights[1:-1])
    #     interior_w = torch.concatenate((interior_w_x, interior_w_y), axis=-1).reshape(
    #         -1, 2
    #     )
    #     boundary_w = torch.empty((4 * (self.n - 1), 2), dtype=torch.float32)
    #     # S edge
    #     boundary_w[:n, 0] = weights
    #     boundary_w[:n, 1] = weights[0]
    #     # E edge
    #     boundary_w[n : 2 * n, 0] = weights[-1]
    #     boundary_w[n : 2 * n - 1, 1] = weights[1:]
    #     # N edge
    #     boundary_w[2 * n - 1 : 3 * n - 2, 0] = torch.flipud(weights[:-1])
    #     boundary_w[2 * n - 1 : 3 * n - 2, 1] = weights[-1]
    #     # W edge
    #     boundary_w[3 * n - 2 :, 0] = weights[0]
    #     boundary_w[3 * n - 2 :, 1] = torch.flipud(weights[1:-1])
    #     self.boundary_w_lst = boundary_w
    #     self.interior_w_lst = interior_w
    #     self.w_lst = torch.concatenate((boundary_w, interior_w), axis=0)

    def eval_func(self, f: Callable[[torch.Tensor], torch.Tensor]) -> float:
        evals = f(self.points_lst)
        return self.eval_tensor(evals)

    def eval_tensor(self, evals: torch.Tensor) -> float:
        return torch.dot(evals, self.w_lst)
