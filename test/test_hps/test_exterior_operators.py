import pytest
import torch
import numpy as np


from src.hps.exterior_operators import build_single_layer_potential
from src.test_utils import (
    check_arrays_close,
    check_no_nan_in_array,
    check_scalars_close,
)


class Test_build_single_layer_potential:
    def test_0(self) -> None:
        """Makes sure the function returns without error and
        returns the correct shape."""
        N_bdry = 10
        bdry_points = torch.randn(size=(N_bdry, 2))
        k = 10.0
        diag_correction = 0.5

        SLP = build_single_layer_potential(bdry_points, k, diag_correction).numpy()

        print(SLP.dtype)

        # Check that SLP is the correct shape and does not
        # contain any NaNs.
        assert SLP.shape == (N_bdry, N_bdry)
        check_no_nan_in_array(SLP)


if __name__ == "__main__":
    pytest.main()
