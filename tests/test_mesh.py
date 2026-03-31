"""Tests for src/lightbeam/mesh.py — RectMesh2D, RectMesh3D, UniformMesh2D, UniformMesh3D."""
import numpy as np
import pytest
from lightbeam.mesh import RectMesh2D, RectMesh3D, UniformMesh2D, UniformMesh3D

XW, YW, ZW = 20.0, 20.0, 10.0
DS, DZ = 0.5, 1.0
PML = 4

# Expected base resolution: int(round(XW/DS)+1) + 2*PML = 41 + 8 = 49
XRES = int(round(XW / DS) + 1) + 2 * PML


# ---------------------------------------------------------------------------
# RectMesh2D
# ---------------------------------------------------------------------------

class TestRectMesh2D:
    def test_shape(self, rect_mesh2d):
        assert rect_mesh2d.shape == (XRES, XRES)

    def test_dx0(self, rect_mesh2d):
        assert rect_mesh2d.dx0 == DS
        assert rect_mesh2d.dy0 == DS

    def test_cvert_ix_length(self, rect_mesh2d):
        m = rect_mesh2d
        assert len(m.xa[m.cvert_ix]) == int(round(XW / DS) + 1)

    def test_pvert_ix_length(self, rect_mesh2d):
        m = rect_mesh2d
        assert len(m.pvert_ix) == 2 * (PML + 1)

    def test_get_weights_shape(self, rect_mesh2d):
        w = rect_mesh2d.get_weights()
        assert w.shape == (XRES, XRES)

    def test_get_weights_positive(self, rect_mesh2d):
        w = rect_mesh2d.get_weights()
        assert np.all(w > 0)

    def test_total_area(self, rect_mesh2d):
        w = rect_mesh2d.get_weights()
        # Total area = (number of cells) * dx * dy = 49*49*0.25
        expected = rect_mesh2d.shape[0] * rect_mesh2d.shape[1] * DS * DS
        assert np.isclose(float(np.sum(w)), expected, rtol=0.01)

    def test_rxa_rya_initially_ones(self, rect_mesh2d):
        m = rect_mesh2d
        assert np.allclose(m.rxa[1:-1], 1.0)
        assert np.allclose(m.rya[1:-1], 1.0)

    def test_refine_by_two_increases_shape(self):
        m = RectMesh2D(XW, YW, DS, DS, Nbc=PML)
        u = np.zeros(m.shape, dtype=complex)
        u[XRES // 2, XRES // 2] = 1.0
        u2 = m.refine_by_two(u, 1e-10)
        assert u2.shape[0] >= m.shape[0]

    def test_resample_complex_identity(self, rect_mesh2d):
        m = rect_mesh2d
        u = np.ones(m.shape, dtype=complex)
        xa = m.xa.copy()
        ya = m.ya.copy()
        out = m.resample_complex(u, xa, ya, xa, ya)
        assert out.shape == m.shape

    def test_get_base_field_shape(self, rect_mesh2d):
        m = rect_mesh2d
        u = np.ones(m.shape, dtype=complex)
        ub = m.get_base_field(u)
        assert ub.shape == m.shape0

    def test_snapto_multiples(self, rect_mesh2d):
        xw2, yw2 = rect_mesh2d.snapto(15.3, 15.3)
        assert abs(xw2 % DS) < 1e-10 or abs(xw2 % DS - DS) < 1e-10

    def test_dxa2xa_monotonic(self, rect_mesh2d):
        m = rect_mesh2d
        xa = m.dxa2xa(m.dxa[1:])
        assert np.all(np.diff(xa) > 0)

    def test_reinit_changes_xw(self):
        m = RectMesh2D(XW, YW, DS, DS, Nbc=PML)
        m.reinit(2 * XW, 2 * YW)
        assert m.xw == 2 * XW

    def test_xhg_shape(self, rect_mesh2d):
        m = rect_mesh2d
        assert m.xhg.shape[0] == m.xg.shape[0] + 1


# ---------------------------------------------------------------------------
# RectMesh3D
# ---------------------------------------------------------------------------

class TestRectMesh3D:
    def test_xy_is_rect_mesh2d(self, rect_mesh3d):
        assert isinstance(rect_mesh3d.xy, RectMesh2D)

    def test_za_length(self, rect_mesh3d):
        m = rect_mesh3d
        assert len(m.za) == m.zres

    def test_sigmax_zero_inside(self, rect_mesh3d):
        val = rect_mesh3d.sigmax(np.array([0.0]))
        assert float(np.real(val[0])) == 0.0

    def test_sigmax_nonzero_outside(self, rect_mesh3d):
        val = rect_mesh3d.sigmax(np.array([XW]))
        assert float(np.abs(val[0])) != 0.0

    def test_get_loc_returns_four(self, rect_mesh3d):
        loc = rect_mesh3d.get_loc()
        assert len(loc) == 4

    def test_sigmay_zero_inside(self, rect_mesh3d):
        val = rect_mesh3d.sigmay(np.array([0.0]))
        assert float(np.real(val[0])) == 0.0


# ---------------------------------------------------------------------------
# UniformMesh2D
# ---------------------------------------------------------------------------

class TestUniformMesh2D:
    def test_max_iters_zero(self, uniform_mesh2d):
        assert uniform_mesh2d.max_iters == 0

    def test_shape_matches_rect(self, uniform_mesh2d, rect_mesh2d):
        assert uniform_mesh2d.shape == rect_mesh2d.shape

    def test_rxa_all_ones(self, uniform_mesh2d):
        m = uniform_mesh2d
        assert np.allclose(m.rxa[1:-1], 1.0)
        assert np.allclose(m.rya[1:-1], 1.0)

    def test_dxa_uniform(self, uniform_mesh2d):
        m = uniform_mesh2d
        assert np.allclose(m.dxa[1:], DS)
        assert np.allclose(m.dya[1:], DS)

    def test_get_weights_uniform(self, uniform_mesh2d):
        w = uniform_mesh2d.get_weights()
        assert np.allclose(w, DS * DS)

    def test_get_base_field_identity(self, uniform_mesh2d):
        m = uniform_mesh2d
        u = np.ones(m.shape, dtype=complex)
        assert uniform_mesh2d.get_base_field(u) is u

    def test_refine_by_two_noop(self, uniform_mesh2d):
        m = uniform_mesh2d
        u = np.ones(m.shape, dtype=complex)
        u2 = m.refine_by_two(u, 1e-3)
        assert u2 is u

    def test_refine_base_noop(self, uniform_mesh2d):
        m = uniform_mesh2d
        u = np.ones(m.shape, dtype=complex)
        result = m.refine_base(u, 1e-3)
        assert result is None  # pass returns None

    def test_pvert_xa_equals_indexed(self, uniform_mesh2d):
        m = uniform_mesh2d
        assert np.allclose(m.pvert_xa, m.xa[m.pvert_ix])

    def test_snapto_works(self, uniform_mesh2d):
        xw2, yw2 = uniform_mesh2d.snapto(15.3, 15.3)
        assert abs(xw2 % DS) < 1e-10 or abs(xw2 % DS - DS) < 1e-10

    def test_reinit_changes_xw(self):
        m = UniformMesh2D(XW, YW, DS, DS, Nbc=PML)
        m.reinit(2 * XW, 2 * YW)
        assert m.xw == 2 * XW

    def test_xhg_shape(self, uniform_mesh2d):
        m = uniform_mesh2d
        assert m.xhg.shape[0] == m.xg.shape[0] + 1


# ---------------------------------------------------------------------------
# UniformMesh3D
# ---------------------------------------------------------------------------

class TestUniformMesh3D:
    def test_xy_is_uniform_mesh2d(self, uniform_mesh3d):
        assert isinstance(uniform_mesh3d.xy, UniformMesh2D)

    def test_za_length(self, uniform_mesh3d):
        m = uniform_mesh3d
        assert len(m.za) == m.zres

    def test_sigmax_zero_inside(self, uniform_mesh3d):
        val = uniform_mesh3d.sigmax(np.array([0.0]))
        assert float(np.real(val[0])) == 0.0

    def test_sigmax_nonzero_outside(self, uniform_mesh3d):
        val = uniform_mesh3d.sigmax(np.array([XW]))
        assert float(np.abs(val[0])) != 0.0

    def test_sigmay_zero_inside(self, uniform_mesh3d):
        val = uniform_mesh3d.sigmay(np.array([0.0]))
        assert float(np.real(val[0])) == 0.0

    def test_get_loc_returns_four(self, uniform_mesh3d):
        loc = uniform_mesh3d.get_loc()
        assert len(loc) == 4

    def test_xy_max_iters_zero(self, uniform_mesh3d):
        assert uniform_mesh3d.xy.max_iters == 0
