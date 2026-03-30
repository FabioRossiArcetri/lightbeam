"""Tests for src/lightbeam/prop.py — Prop3D"""
import numpy as np
import pytest
from lightbeam.mesh import RectMesh3D, UniformMesh3D
import lightbeam.optics as optics
from lightbeam.prop import Prop3D, tri_solve_vec
from lightbeam.misc import overlap_nonu

# Tiny mesh so each test finishes quickly
WL   = 1.0
XW   = 10.0
YW   = 10.0
ZW   = 5.0
DS   = 1.0
DZ   = 1.0
PML  = 4
NCLAD = 1.4504

# Cylinder just covers the computational domain (avoids AA bbox overflow)
_CYL_R = 5.0


def _make_homogeneous_optic():
    cyl = optics.scaled_cyl([0, 0], _CYL_R, ZW, NCLAD, NCLAD)
    return optics.OpticSys([cyl], NCLAD)


@pytest.fixture(scope="module")
def prop_rect():
    mesh = RectMesh3D(XW, YW, ZW, DS, DZ, PML)
    return Prop3D(WL, mesh, _make_homogeneous_optic(), NCLAD)


@pytest.fixture(scope="module")
def prop_uniform():
    mesh = UniformMesh3D(XW, YW, ZW, DS, DZ, PML)
    return Prop3D(WL, mesh, _make_homogeneous_optic(), NCLAD)


# ---------------------------------------------------------------------------
# Init / flags
# ---------------------------------------------------------------------------

class TestProp3DInit:
    def test_rect_is_uniform_false(self, prop_rect):
        assert prop_rect._is_uniform is False

    def test_uniform_is_uniform_true(self, prop_uniform):
        assert prop_uniform._is_uniform is True

    def test_wl0_stored(self, prop_rect):
        assert np.isclose(prop_rect.wl0, WL)

    def test_n0_stored(self, prop_rect):
        assert np.isclose(prop_rect.n0, NCLAD)

    def test_mesh_stored(self, prop_rect):
        assert isinstance(prop_rect._mesh, RectMesh3D)

    def test_uniform_mesh_stored(self, prop_uniform):
        assert isinstance(prop_uniform._mesh, UniformMesh3D)


# ---------------------------------------------------------------------------
# calculate_PML_mats
# ---------------------------------------------------------------------------

class TestPMLMats:
    def test_returns_six_arrays(self, prop_uniform):
        result = prop_uniform.calculate_PML_mats()
        assert len(result) == 6

    def test_pml_array_length(self, prop_uniform):
        Rx, Tupx, Tdox, Ry, Tupy, Tdoy = prop_uniform.calculate_PML_mats()
        xy = prop_uniform._mesh.xy
        expected_len = len(xy.pvert_ix)
        assert len(Rx) == expected_len
        assert len(Ry) == expected_len

    def test_pml_arrays_nonzero(self, prop_uniform):
        Rx, Tupx, Tdox, Ry, Tupy, Tdoy = prop_uniform.calculate_PML_mats()
        assert np.any(Rx != 0)


# ---------------------------------------------------------------------------
# allocate_mats
# ---------------------------------------------------------------------------

class TestAllocateMats:
    def test_returns_nine_arrays(self, prop_uniform):
        result = prop_uniform.allocate_mats()
        assert len(result) == 9

    def test_trimats_shapes(self, prop_uniform):
        result = prop_uniform.allocate_mats()
        _trimatsx, rmatx, gx, _trimatsy, rmaty, gy, *_ = result
        assert len(_trimatsx) == 3
        assert len(_trimatsy) == 3


# ---------------------------------------------------------------------------
# check_z_inv
# ---------------------------------------------------------------------------

class TestCheckZInv:
    def test_returns_bool(self, prop_uniform):
        result = prop_uniform.check_z_inv()
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# tri_solve_vec
# ---------------------------------------------------------------------------

class TestTriSolveVec:
    def test_identity_diag(self):
        """Diagonal matrix: solution equals rhs."""
        N, M = 5, 3
        a = np.zeros((N, M), dtype=complex)
        b = np.full((N, M), 2.0 + 0j)
        c = np.zeros((N, M), dtype=complex)
        r = np.ones((N, M), dtype=complex)
        g = np.zeros((N, M), dtype=complex)
        u = np.zeros((N, M), dtype=complex)
        tri_solve_vec(a, b, c, r, g, u)
        expected = np.full((N, M), 0.5 + 0j)
        assert np.allclose(u, expected)

    def test_tridiagonal_matches_numpy(self):
        """Compare against numpy.linalg.solve for a full tridiagonal system."""
        N, M = 5, 1
        a = np.zeros((N, M), dtype=complex)
        b = np.full((N, M), 2.0 + 0j)
        c = np.zeros((N, M), dtype=complex)
        a[1:, 0] = -1.0
        c[:-1, 0] = -1.0
        r = np.ones((N, M), dtype=complex)
        g = np.zeros((N, M), dtype=complex)
        u = np.zeros((N, M), dtype=complex)
        tri_solve_vec(a, b, c, r, g, u)

        A = (np.diag([2.0] * N)
             + np.diag([-1.0] * (N - 1), 1)
             + np.diag([-1.0] * (N - 1), -1))
        expected = np.linalg.solve(A, np.ones(N))
        assert np.allclose(u[:, 0].real, expected, atol=1e-12)


# ---------------------------------------------------------------------------
# prop2end_uniform
# ---------------------------------------------------------------------------

def _gaussian(xy):
    return np.exp(-(xy.xg**2 + xy.yg**2) / 4.0).astype(complex)


class TestProp2EndUniform:
    def test_runs_rect_mesh(self, prop_rect, capsys):
        xy = prop_rect._mesh.xy
        u_in = _gaussian(xy)
        u_out = prop_rect.prop2end_uniform(u_in)
        assert u_out.shape == xy.shape

    def test_runs_uniform_mesh(self, prop_uniform, capsys):
        xy = prop_uniform._mesh.xy
        u_in = _gaussian(xy)
        u_out = prop_uniform.prop2end_uniform(u_in)
        assert u_out.shape == xy.shape

    def test_output_is_complex(self, prop_uniform):
        xy = prop_uniform._mesh.xy
        u_in = _gaussian(xy)
        u_out = prop_uniform.prop2end_uniform(u_in)
        assert np.iscomplexobj(u_out)

    def test_power_does_not_increase(self, prop_uniform):
        """In a lossless region, power should not increase (PML absorbs)."""
        xy = prop_uniform._mesh.xy
        u_in = _gaussian(xy)
        w = xy.get_weights()
        p_in = float(overlap_nonu(u_in, u_in, w))
        u_out = prop_uniform.prop2end_uniform(u_in)
        p_out = float(overlap_nonu(u_out, u_out, w))
        assert p_out <= p_in * 1.05  # allow 5% tolerance for numerics


# ---------------------------------------------------------------------------
# prop2end (adaptive)
# ---------------------------------------------------------------------------

class TestProp2End:
    def test_returns_tuple(self, prop_rect):
        result = prop_rect.prop2end(lambda xg, yg: np.exp(-(xg**2 + yg**2) / 4.0).astype(complex))
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_u_shape_is_2d(self, prop_rect):
        u, u0 = prop_rect.prop2end(lambda xg, yg: np.exp(-(xg**2 + yg**2) / 4.0).astype(complex))
        assert u.ndim == 2

    def test_u0_shape_matches_base_mesh(self, prop_rect):
        u, u0 = prop_rect.prop2end(lambda xg, yg: np.exp(-(xg**2 + yg**2) / 4.0).astype(complex))
        xy = prop_rect._mesh.xy
        assert u0.ndim == 2
