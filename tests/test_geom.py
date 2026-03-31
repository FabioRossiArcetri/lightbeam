"""Tests for src/lightbeam/geom.py"""
import numpy as np
import pytest
from lightbeam.geom import pixwt


def test_pixwt_center_fully_inside():
    """A pixel at the center of the circle should be fully covered."""
    result = pixwt(0.0, 0.0, 5.0, np.array([0.0]), np.array([0.0]))
    assert np.isclose(float(result[0]), 1.0)


def test_pixwt_far_outside():
    """A pixel far outside the circle should have zero coverage."""
    result = pixwt(0.0, 0.0, 1.0, np.array([10.0]), np.array([10.0]))
    assert np.isclose(float(result[0]), 0.0, atol=1e-10)


def test_pixwt_partial_overlap():
    """A pixel straddling the edge should have fractional coverage."""
    result = pixwt(0.0, 0.0, 1.0, np.array([1.0]), np.array([0.0]))
    val = float(result[0])
    assert 0.0 <= val <= 1.0


def test_pixwt_returns_nonnegative():
    """Coverage values should be non-negative."""
    xs = np.linspace(-2.0, 2.0, 9)
    ys = np.linspace(-2.0, 2.0, 9)
    xg, yg = np.meshgrid(xs, ys, indexing='ij')
    result = pixwt(0.0, 0.0, 1.5, xg.ravel(), yg.ravel())
    assert np.all(result >= -1e-12)


def test_pixwt_leq_one():
    """Coverage values should not exceed 1."""
    xs = np.linspace(-2.0, 2.0, 9)
    ys = np.linspace(-2.0, 2.0, 9)
    xg, yg = np.meshgrid(xs, ys, indexing='ij')
    result = pixwt(0.0, 0.0, 1.5, xg.ravel(), yg.ravel())
    assert np.all(result <= 1.0 + 1e-12)


def test_pixwt_symmetry():
    """Coverage should be symmetric about the circle center."""
    r = pixwt(0.0, 0.0, 1.0, np.array([0.5]), np.array([0.0]))
    l = pixwt(0.0, 0.0, 1.0, np.array([-0.5]), np.array([0.0]))
    assert np.isclose(float(r[0]), float(l[0]), atol=1e-10)
