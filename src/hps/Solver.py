import numpy as np
import torch
from Solution import Solution


class Solver:
    def __init__(
        self,
        spatial_domain_max: int,
        frequency: float,
        m: int,
    ) -> None:
        """Initializes the Solver object. This function
        calls methods that pre-compute solution maps, and
        as such it may take some time to complete.

        Args:
            spatial_domain_max (int): Half of the side length of the square domain.
            frequency (float): Frequency of the incoming wave.
            m (int): The number of levels in the heirarchical space decomposition.
        """
        self.spatial_domain_max = spatial_domain_max
        self.frequency = frequency
        self.m = m

    def _construct_T_int(self) -> None:
        # TODO: fill in this function.
        pass

    def _construct_T_ext(self) -> None:
        # TODO: fill in this function.
        pass

    def get_particular_solution(
        self, q: torch.Tensor, in_wave_directions: torch.Tensor
    ) -> Solution:
        # TODO: fill this in.
        pass

    def _partition_domain(self, m: int) -> None:
        pass
