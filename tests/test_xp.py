"""Tests for src/lightbeam/xp.py"""
import numpy as np
import pytest
import os


def test_default_backend_is_numpy():
    if os.environ.get("LIGHTBEAM_BACKEND", "").lower() == "cupy":
        pytest.skip("CuPy backend is active")
    from lightbeam import xp as xpmod
    assert xpmod.xp is np


def test_is_gpu_false_by_default():
    if os.environ.get("LIGHTBEAM_BACKEND", "").lower() == "cupy":
        pytest.skip("CuPy backend is active")
    from lightbeam.xp import is_gpu
    assert not is_gpu()


def test_to_cpu_passthrough():
    from lightbeam.xp import to_cpu
    arr = np.array([1, 2, 3])
    result = to_cpu(arr)
    assert isinstance(result, np.ndarray)
    assert np.array_equal(result, arr)


def test_to_device_passthrough():
    from lightbeam.xp import to_device
    arr = np.array([1, 2, 3])
    result = to_device(arr)
    assert np.array_equal(result, arr)


def test_is_gpu_and_to_cpu_consistent():
    from lightbeam.xp import is_gpu, to_cpu, to_device
    arr = np.array([1.0, 2.0, 3.0])
    device_arr = to_device(arr)
    cpu_arr = to_cpu(device_arr)
    assert np.allclose(cpu_arr, arr)
    if not is_gpu():
        assert np.array_equal(cpu_arr, device_arr)
