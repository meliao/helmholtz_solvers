from typing import Tuple
import torch
import numpy as np


def differentiation_matrix_1d(
    points: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Creates a 1-D Chebyshev differentiation matrix as described in (T00) Ch 6. Expects points are Chebyshev points on [-1, 1].

    Args:
        points (torch.Tensor): Has shape (p,)

    Returns:
        torch.Tensor: Has shape (p,p)
    """
    p = points.shape[0]
    print(p)

    # Here's the code from the MATLAB recipe book
    # c = [2; ones(N-1,1); 2].*(-1).^(0:N)';
    # X = repmat(x,1,N+1);
    # dX = X-X';
    # D = (c*(1./c)')./(dX+(eye(N+1))); % off-diagonal entries
    # D = D - diag(sum(D'));

    # Here's the pytorch version Owen wrote
    c = torch.ones(p)
    c[0] = 2
    c[-1] = 2
    for i in range(1, p, 2):
        c[i] = -1 * c[i]
    x = points.unsqueeze(-1).repeat(1, p)
    dx = x - x.permute(1, 0)
    coeff = torch.outer(c, 1 / c)
    d = coeff / (dx + torch.eye(p))
    dd = torch.diag(torch.sum(d, dim=1))

    # IDK why but I had to add the sign error below to cancel out a sign error I made sonewhere above.
    d = dd - d

    return d


def interp_matrix_to_Cheby(n: int, other_points: torch.Tensor) -> torch.Tensor:
    """This function returns a matrix which interpolates a function evaluated at <other_points> to the function evaluated at the n Chebyshev points. It does this by expanding the function into a basis of Chebyshev polynomials and then evaluating the Chebyshev polynomials at the sample point.

    Args:
        n (int): The number of Chebyshev points on which to evaluate the solution
        other_points (torch.Tensor): Has shape (p,)

    Returns:
        torch.Tensor: Has shape (n, p)
    """
    p = other_points.shape[0]

    # First make a (n, p) matrix evaluating the first n Chebyshev polynomials in the rows at each of the points in <other_points>

    first_op = torch.empty((n, p), dtype=other_points.dtype)

    first_op[0] = torch.ones_like(first_op[0])
    first_op[1] = other_points.clone()

    # Recursion: T_n(x) = 2x T_{n-1}(x) - T_{n-2}(x)
    for i in range(2, n):
        first_op[i] = 2 * other_points * first_op[i - 1] - first_op[i - 2]

    # first_op = first_op / n
    print("first_op", first_op)

    # Second, make a (n, n) matrix evaluating the sum of the first n Chebyshev polynomials at the points self.points_1d.
    # Do this by exploiting the formula T_n(cos(theta)) = cos(n theta).
    # The jth Chebyshev point is cos( pi * j / (n-1)) where j ranges {0,...,n-1}
    # That means the jth angle is pi * j / (n-1)

    second_op = torch.empty((n, n), dtype=other_points.dtype)

    _, angles = chebyshev_points(n)
    print("Angles", angles)
    for i in range(n):
        cos_angles = torch.cos(i * angles)
        print("i: ", i)
        print("cos_angles: ", cos_angles)
        second_op[:, i] = cos_angles

    print("second_op", second_op)
    return second_op @ first_op


def chebyshev_points(n: int) -> torch.Tensor:
    """Returns n Chebyshev points over the interval [-1, 1]

    out[i] = cos(pi * i / (n-1)) for i={0,...,n-1}

    Actually I return the reversed array so the smallest points come first.

    Args:
        n (int): number of Chebyshev points to return

    Returns:
        torch.Tensor: The sampled points in [-1, 1]
    """
    cos_args = torch.arange(n).to(torch.float64) / (n - 1)
    angles = torch.flipud(torch.pi * cos_args)
    return torch.cos(angles), angles


a = """
    Generate a matrix for Lagrange polynomial interpolation from points x to points y.

    Parameters:
    - x: Tensor containing the x-coordinates of the interpolation points
    - y: Tensor containing the y-coordinates of the interpolation points

    Returns:
    - lagrange_matrix: Matrix for Lagrange polynomial interpolation
    """


def lagrange_interpolation_matrix(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Generates a Lagrange 1D polynomial interpolation matrix, which interpolates from the points in x to the points in y

    Args:
        x (torch.Tensor): Has shape (p,)
        y (torch.Tensor): Has shape (n,)

    Returns:
        torch.Tensor: Has shape (n,p)
    """
    p = len(x)
    n = len(y)

    lagrange_matrix = torch.zeros((n, p), dtype=torch.float32)

    for j in range(n):
        for i in range(p):
            num_1 = torch.prod(y[j] - x[:i])
            num_2 = torch.prod(y[j] - x[i + 1 :])
            numerator = num_1 * num_2
            denominator = torch.prod(x[i] - x[:i]) * torch.prod(x[i] - x[i + 1 :])
            lagrange_matrix[j, i] = numerator / denominator

    return lagrange_matrix
