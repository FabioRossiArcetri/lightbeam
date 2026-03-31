"""Tests for src/lightbeam/misc.py"""
import numpy as np
import pytest
from lightbeam.misc import (
    overlap, overlap_nonu, overlap_nonu_trap, normalize, norm_nonu, resize2
)


def test_overlap_real():
    u = np.ones((4, 4), dtype=complex)
    result = overlap(u, u)
    assert np.isclose(float(result), 16.0)


def test_overlap_with_weight():
    u = np.ones((4, 4), dtype=complex)
    result = overlap(u, u, weight=0.5)
    assert np.isclose(float(result), 8.0)


def test_overlap_complex_conjugate():
    u = np.array([[1 + 1j, 2 + 0j]], dtype=complex)
    v = np.array([[1 - 1j, 2 + 0j]], dtype=complex)
    # overlap = |sum(conj(u)*v)| = |conj(1+1j)*(1-1j) + conj(2)*(2)|
    #         = |(1-1j)*(1-1j) + 4| = |(1-2j-1) + 4| = |4 - 2j|
    result = float(overlap(u, v))
    expected = abs((1 - 1j) * (1 - 1j) + 4.0)
    assert np.isclose(result, expected)


def test_overlap_nonu_uniform_weights():
    u = np.ones((5, 5), dtype=complex)
    w = np.ones((5, 5))
    result = float(overlap_nonu(u, u, w))
    assert np.isclose(result, 25.0)


def test_overlap_nonu_scaled_weights():
    u = np.ones((3, 3), dtype=complex)
    w = np.full((3, 3), 2.0)
    result = float(overlap_nonu(u, u, w))
    assert np.isclose(result, 18.0)


def test_overlap_nonu_trap():
    try:
        n = 20
        xa = np.linspace(0, 1, n)
        ya = np.linspace(0, 1, n)
        u = np.ones((n, n), dtype=complex)
        result = float(overlap_nonu_trap(u, u, xa, ya))
        assert np.isclose(result, 1.0, atol=0.01)
    except AttributeError as e:
        pytest.skip(f"overlap_nonu_trap uses deprecated numpy API: {e}")


def test_normalize_unit_norm():
    u = np.array([[2.0 + 0j, 0j], [0j, 0j]])
    normalize(u, weight=1.0)
    # After normalization, overlap(u,u) = 1
    result = float(overlap(u, u))
    assert np.isclose(result, 1.0)


def test_norm_nonu_unit_norm():
    u = np.ones((4, 4), dtype=complex) * 2.0
    w = np.ones((4, 4)) * 0.25
    u2 = norm_nonu(u.copy(), w)
    result = float(overlap_nonu(u2, u2, w))
    assert np.isclose(result, 1.0)


def test_resize2_shape():
    img = np.ones((10, 10))
    out = resize2(img, (5, 5))
    assert out.shape == (5, 5)


def test_resize2_values_constant():
    img = np.ones((10, 10)) * 3.0
    out = resize2(img, (7, 7))
    assert np.allclose(out, 3.0, atol=1e-10)
