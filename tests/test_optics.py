"""Tests for src/lightbeam/optics.py"""
import numpy as np
import pytest
from lightbeam.mesh import RectMesh2D
import lightbeam.optics as optics

XW, YW, ZW = 20.0, 20.0, 10.0
DS = 0.5
PML = 4
NCORE = 1.4592
NCLAD = 1.4504
NJACK = 1.4449
RCORE = 2.0
RCLAD = 8.0


@pytest.fixture
def mesh2d():
    return RectMesh2D(XW, YW, DS, DS, Nbc=PML)


@pytest.fixture
def simple_optic(mesh2d):
    clad = optics.scaled_cyl([0, 0], RCLAD, ZW, NCLAD, NJACK)
    core = optics.scaled_cyl([0, 0], RCORE, ZW, NCORE, NCLAD)
    sys = optics.OpticSys([clad, core], NJACK)
    sys.set_sampling(mesh2d)
    return sys


class TestOpticPrim:
    def test_n_attribute(self):
        prim = optics.OpticPrim(NCORE)
        assert prim.n == NCORE

    def test_n2_attribute(self):
        prim = optics.OpticPrim(NCORE)
        assert np.isclose(prim.n2, NCORE**2)


class TestScaledCyl:
    def test_constructor(self):
        cyl = optics.scaled_cyl([0, 0], RCORE, ZW, NCORE, NCLAD)
        assert cyl.r == RCORE

    def test_nb2(self):
        cyl = optics.scaled_cyl([0, 0], RCORE, ZW, NCORE, NCLAD)
        assert np.isclose(cyl.nb2, NCLAD**2)

    def test_n2(self):
        cyl = optics.scaled_cyl([0, 0], RCORE, ZW, NCORE, NCLAD)
        assert np.isclose(cyl.n2, NCORE**2)

    def test_set_ior_sq(self, mesh2d):
        cyl = optics.scaled_cyl([0, 0], RCLAD, ZW, NCLAD, NJACK)
        cyl.set_sampling(mesh2d)
        out = np.zeros(mesh2d.shape, dtype=complex)
        out[:] = NJACK**2
        cyl.set_IORsq(out, ZW / 2)
        # Values inside cyl should be NCLAD^2
        center = mesh2d.shape[0] // 2
        assert float(np.real(out[center, center])) > NJACK**2


class TestOpticSys:
    def test_nb(self, simple_optic):
        assert np.isclose(simple_optic.nb, NJACK)

    def test_nb2(self, simple_optic):
        assert np.isclose(simple_optic.nb2, NJACK**2)

    def test_set_sampling_assigns_mesh(self, simple_optic, mesh2d):
        assert simple_optic.xymesh is mesh2d

    def test_set_ior_sq_core_region(self, simple_optic, mesh2d):
        out = np.zeros(mesh2d.shape, dtype=complex)
        simple_optic.set_IORsq(out, 0.0)
        center = mesh2d.shape[0] // 2
        val = float(np.real(out[center, center]))
        assert np.isclose(val, NCORE**2, rtol=0.01)

    def test_set_ior_sq_outside(self, simple_optic, mesh2d):
        out = np.zeros(mesh2d.shape, dtype=complex)
        out[:] = NJACK**2  # pre-fill with background
        simple_optic.set_IORsq(out, 0.0)
        # corner (PML region) should be background
        val = float(np.real(out[0, 0]))
        assert np.isclose(val, NJACK**2, rtol=0.05)


class TestLant3Big:
    def test_constructor(self):
        lant = optics.lant3big(RCORE, RCLAD, NCORE, NCLAD, NJACK, 5.0, ZW)
        assert isinstance(lant, optics.OpticSys)

    def test_nb(self):
        lant = optics.lant3big(RCORE, RCLAD, NCORE, NCLAD, NJACK, 5.0, ZW)
        assert np.isclose(lant.nb, NJACK)

    def test_has_correct_num_elements(self):
        lant = optics.lant3big(RCORE, RCLAD, NCORE, NCLAD, NJACK, 5.0, ZW)
        # 1 clad + 3 cores = 4 elements
        assert len(lant.elmnts) == 4


class TestLant19:
    def test_constructor(self):
        lant = optics.lant19(RCORE, RCLAD, NCORE, NCLAD, NJACK, 5.0, ZW)
        assert isinstance(lant, optics.OpticSys)

    def test_nb(self):
        lant = optics.lant19(RCORE, RCLAD, NCORE, NCLAD, NJACK, 5.0, ZW)
        assert np.isclose(lant.nb, NJACK)
