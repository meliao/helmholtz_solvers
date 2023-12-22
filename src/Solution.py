import torch

from Solver import Solver


class Solution:
    def __init__(
        self, solver_obj: Solver, q: torch.Tensor, in_wave_directions: torch.Tensor
    ) -> None:
        """Initializes the Solution object. This is typically called by
        the Solver.get_particular_solution() object.

        This function does not actually do the computational work required to solve the PDE. That work is completed by calling
        self.solve()

        Args:
            solver_obj (Solver): Solver object with pre-computed T_int and T_ext solution operators.
            q (torch.Tensor): Scattering potential. Has shape (N, N)
            in_wave_directions (torch.Tensor): The incident wave directions. Has shape (N_dirs,)
        """
        self.solver_obj = solver_obj
        self.q = q
        self.in_wave_directions = in_wave_directions

    def solve(self) -> None:
        pass
