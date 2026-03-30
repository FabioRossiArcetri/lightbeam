"""Shared fixtures for the lightbeam test suite."""
import numpy as np
import pytest
from lightbeam.mesh import RectMesh2D, RectMesh3D, UniformMesh2D, UniformMesh3D
import lightbeam.optics as optics

WL    = 1.0
DS    = 0.5
DZ    = 1.0
XW    = 20.0
YW    = 20.0
ZW    = 10.0
PML   = 4
NCLAD = 1.4504
NCORE = NCLAD + 0.0088
NJACK = NCLAD - 5.5e-3
RCORE = 2.0
RCLAD = 8.0


@pytest.fixture(scope="module")
def rect_mesh2d():
    return RectMesh2D(XW, YW, DS, DS, Nbc=PML)


@pytest.fixture(scope="module")
def uniform_mesh2d():
    return UniformMesh2D(XW, YW, DS, DS, Nbc=PML)


@pytest.fixture(scope="module")
def rect_mesh3d():
    return RectMesh3D(XW, YW, ZW, DS, DZ, PML)


@pytest.fixture(scope="module")
def uniform_mesh3d():
    return UniformMesh3D(XW, YW, ZW, DS, DZ, PML)


@pytest.fixture(scope="module")
def straight_fiber_optic():
    z_ex = ZW
    core = optics.scaled_cyl([0, 0], RCORE, z_ex, NCORE, NCLAD)
    clad = optics.scaled_cyl([0, 0], RCLAD, z_ex, NCLAD, NJACK)
    return optics.OpticSys([clad, core], NJACK)


@pytest.fixture(scope="module")
def homogeneous_optic():
    """Background-only optical system (infinite uniform cylinder)."""
    z_ex = ZW
    clad = optics.scaled_cyl([0, 0], 1e6, z_ex, NCLAD, NCLAD)
    return optics.OpticSys([clad], NCLAD)
