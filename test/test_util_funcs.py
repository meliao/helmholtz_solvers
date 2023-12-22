import numpy as np
import torch
import pytest

from src.utils import differentiation_matrix_1d
from src.test_utils import check_arrays_close, check_scalars_close


class Test_differentiation_matrix_1d:
    def test_0(self) -> None:
        p = torch.arange(10)
        o = differentiation_matrix_1d(p)
        assert o.shape == (10, 10), o.shape


if __name__ == "__main__":
    pytest.main()
