import torch
from scipy import special
from src.utils import differentiation_matrix_1d, lagrange_interpolation_matrix


def build_single_layer_potential(
    bdry_points: torch.Tensor, k: float, diag_correction: float = 0.0
) -> torch.Tensor:
    """This function constructs a single-layer Boundary integral operator for the frequency-k Helmholtz equation.
    The boundary is specified by the points in bdry_points.

    Inputs:
    bdry_points (torch.Tensor): The boundary points. Has shape (N_bdry, 2)
    k (float): The wavenumber.

    Returns:
    torch.Tensor: The single-layer potential operator. Has shape (N_bdry, N_bdry)
    """
    N_bdry = bdry_points.shape[0]

    # Construct the single-layer potential operator.
    SLP = torch.zeros(N_bdry, N_bdry, dtype=torch.complex64)
    for i in range(N_bdry):
        for j in range(N_bdry):
            if i == j:
                SLP[i, j] = diag_correction
            else:
                r = torch.norm(bdry_points[i] - bdry_points[j])
                SLP[i, j] = 1j / 4 * special.hankel1(0, k * r)

    return SLP
