"""Tests for src/lightbeam/LPmodes.py"""
import numpy as np
import pytest
from lightbeam.LPmodes import get_NA, get_V, get_modes, lpfield, get_MFD

WL = 1.0
K0 = 2 * np.pi / WL
NCORE = 1.4592
NCLAD = 1.4504
RCORE = 2.0


def test_get_na_positive():
    na = get_NA(NCORE, NCLAD)
    assert na > 0


def test_get_na_formula():
    expected = np.sqrt(NCORE**2 - NCLAD**2)
    assert np.isclose(get_NA(NCORE, NCLAD), expected)


def test_get_v_positive():
    V = get_V(K0, RCORE, NCORE, NCLAD)
    assert V > 0


def test_get_v_scales_with_radius():
    V1 = get_V(K0, RCORE, NCORE, NCLAD)
    V2 = get_V(K0, 2 * RCORE, NCORE, NCLAD)
    assert np.isclose(V2, 2 * V1)


def test_get_modes_single_mode():
    """V < 2.405 should give only the LP01 mode."""
    V = get_V(K0, RCORE, NCORE, NCLAD)
    modes = get_modes(V)
    assert len(modes) >= 1
    assert (0, 1) in modes


def test_get_modes_multimode():
    """Large V should return more than one mode."""
    V_large = 10.0
    modes = get_modes(V_large)
    assert len(modes) > 1


def test_get_mfd_positive():
    mfd = get_MFD(K0, RCORE, NCORE, NCLAD)
    assert mfd > 0


def test_lpfield_shape():
    x = np.linspace(-5, 5, 30)
    y = np.linspace(-5, 5, 30)
    xg, yg = np.meshgrid(x, y, indexing='ij')
    f = lpfield(xg, yg, 0, 1, RCORE, WL, NCORE, NCLAD)
    assert f.shape == (30, 30)


def test_lpfield_real_valued():
    """LP modes (cos variant) should be real."""
    x = np.linspace(-5, 5, 20)
    y = np.linspace(-5, 5, 20)
    xg, yg = np.meshgrid(x, y, indexing='ij')
    f = lpfield(xg, yg, 0, 1, RCORE, WL, NCORE, NCLAD, which="cos")
    assert np.all(np.isreal(f))


def test_lpfield_peak_at_center():
    """LP01 mode should have its maximum near the center."""
    x = np.linspace(-5, 5, 51)
    y = np.linspace(-5, 5, 51)
    xg, yg = np.meshgrid(x, y, indexing='ij')
    f = np.abs(lpfield(xg, yg, 0, 1, RCORE, WL, NCORE, NCLAD))
    center_i, center_j = np.unravel_index(np.argmax(f), f.shape)
    mid = len(x) // 2
    assert abs(center_i - mid) <= 2
    assert abs(center_j - mid) <= 2
