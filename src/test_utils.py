"""
This module does not contain runnable tests, but it contains utility functions that other tests call.
It is named 'test_utils.py' so that pytest will import it as a top-level module when running the 
following command from the project root directory.
```
python -m pytest test/
```
"""
import numpy as np
import os
import sys
import pytest
import shutil


def check_arrays_close(
    a: np.ndarray,
    b: np.ndarray,
    a_name: str = "a",
    b_name: str = "b",
    msg: str = "",
    atol: float = 1e-8,
    rtol: float = 1e-05,
) -> None:
    s = _evaluate_arrays_close(a, b, msg, atol, rtol)
    allclose_bool = np.allclose(a, b, atol=atol, rtol=rtol)
    assert allclose_bool, s


def _evaluate_arrays_close(
    a: np.ndarray,
    b: np.ndarray,
    msg: str = "",
    atol: float = 1e-08,
    rtol: float = 1e-05,
) -> str:
    assert a.size == b.size, f"Sizes don't match: {a.size} vs {b.size}"

    max_diff = np.max(np.abs(a - b))
    samp_n = 5

    # Compute relative difference
    x = a.flatten()
    y = b.flatten()
    difference_count = np.logical_not(np.isclose(x, y, atol=atol, rtol=rtol)).sum()

    bool_arr = np.abs(x) >= 1e-15
    rel_diffs = np.abs((x[bool_arr] - y[bool_arr]) / x[bool_arr])
    if bool_arr.astype(int).sum() == 0:
        return msg + "No nonzero entries in A"
    max_rel_diff = np.max(rel_diffs)
    s = (
        msg
        + "Arrays differ in {} / {} entries. Max absolute diff: {}; max relative diff: {}".format(
            difference_count, a.size, max_diff, max_rel_diff
        )
    )

    return s


def check_scalars_close(
    a, b, a_name: str = "a", b_name: str = "b", msg: str = "", atol=1e-08, rtol=1e-05
):
    max_diff = np.max(np.abs(a - b))
    s = msg + "Max diff: {:.8f}, {}: {}, {}: {}".format(max_diff, a_name, a, b_name, b)
    allclose_bool = np.allclose(a, b, atol=atol, rtol=rtol)
    assert allclose_bool, s


def check_no_nan_in_array(arr: np.ndarray) -> None:
    nan_points = np.argwhere(np.isnan(arr))

    s = f"Found NaNs in arr of shape {arr.shape}. Some of the points are at indices {nan_points.flatten()[:5]}"

    assert not np.any(np.isnan(arr)), s


TEMP_SERIALIZE_DIR = "test/tmp/"


@pytest.fixture()
def add_and_remove_tmpdir(request):
    if not os.path.isdir(TEMP_SERIALIZE_DIR):
        os.mkdir(TEMP_SERIALIZE_DIR)

    def teardown():
        if os.path.isdir(TEMP_SERIALIZE_DIR):
            shutil.rmtree(TEMP_SERIALIZE_DIR)

    request.addfinalizer(teardown)
