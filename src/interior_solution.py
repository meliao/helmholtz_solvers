import torch

from src.Quad import GaussLegendre1D, Cheby2D
from src.utils import differentiation_matrix_1d


class LeafNode:
    def __init__(
        self, half_side_len: float, n: int, upper_left_x: float, upper_left_y: float
    ) -> None:
        self.half_side_len = half_side_len
        self.n = n
        self.upper_left_pos = (upper_left_x, upper_left_y)

        self.gauss_quad_obj = GaussLegendre1D(half_side_len, n)
        self.cheby_quad_obj = Cheby2D(half_side_len, n)

        self.D = differentiation_matrix_1d(self.cheby_quad_obj.points_1d)
