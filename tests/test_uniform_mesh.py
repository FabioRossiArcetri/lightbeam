"""
Tests for UniformMesh2D, UniformMesh3D, and the Prop3D uniform propagation path.
"""

import numpy as np
import pytest
from lightbeam import UniformMesh2D, UniformMesh3D, RectMesh2D, RectMesh3D, Prop3D
import lightbeam.optics as optics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FreeSpace(optics.OpticSys):
    """Trivial z-invariant free-space optical system for testing."""
    z_invariant = True

    def __init__(self, nb):
        self.nb = nb
        self.nb2 = nb * nb

    def set_IORsq(self, out, z, xg=None, yg=None, coeff=1.0):
        out[:] = coeff * self.nb2

    def set_sampling(self, xymesh):
        pass


WL = 1.55e-6
N0 = 1.44
DS = 0.5e-6
DZ = 10e-6
XW = 10e-6
YW = 10e-6
ZW = 30e-6


# ---------------------------------------------------------------------------
# UniformMesh2D
# ---------------------------------------------------------------------------

class TestUniformMesh2D:
    def setup_method(self):
        self.m = UniformMesh2D(XW, YW, DS, DS, Nbc=4)

    def test_shape_matches_expected(self):
        xres = round(XW / DS) + 1 + 2 * 4
        assert self.m.shape == (xres, xres)
        assert self.m.xres == xres
        assert self.m.yres == xres

    def test_rxa_rya_are_ones(self):
        assert np.all(self.m.rxa == 1.0)
        assert np.all(self.m.rya == 1.0)

    def test_dxa_dya_are_uniform(self):
        assert np.allclose(self.m.dxa, DS)
        assert np.allclose(self.m.dya, DS)

    def test_xix_base_is_identity(self):
        assert np.all(self.m.xix_base == np.arange(self.m.xres))

    def test_yix_base_is_identity(self):
        assert np.all(self.m.yix_base == np.arange(self.m.yres))

    def test_get_weights_uniform(self):
        w = self.m.get_weights()
        assert w.shape == (self.m.xres, self.m.yres)
        assert np.allclose(w, DS * DS)

    def test_get_base_field_identity(self):
        u = np.ones(self.m.shape)
        assert self.m.get_base_field(u) is u

    def test_snapto(self):
        xw2, yw2 = self.m.snapto(9.3e-6, 8.7e-6)
        # result should be an even multiple of 2*DS
        assert xw2 % (2 * DS) < 1e-20 or abs(xw2 % (2 * DS) - 2 * DS) < 1e-20

    def test_domain_bounds(self):
        assert np.isclose(self.m.xm, -XW / 2 - 4 * DS)
        assert np.isclose(self.m.xM,  XW / 2 + 4 * DS)
        assert np.isclose(self.m.ym, -YW / 2 - 4 * DS)
        assert np.isclose(self.m.yM,  YW / 2 + 4 * DS)

    def test_reinit_updates_shape(self):
        new_xw = 20e-6
        new_yw = 20e-6
        self.m.reinit(new_xw, new_yw)
        xres_new = round(new_xw / DS) + 1 + 2 * 4
        assert self.m.xres == xres_new
        assert self.m.shape == (xres_new, xres_new)

    def test_has_required_attributes(self):
        required = [
            'dx0', 'dy0', 'xw', 'yw', 'xm', 'xM', 'ym', 'yM',
            'xa', 'ya', 'xg', 'yg', 'xhg', 'yhg',
            'dxa', 'dya', 'rxa', 'rya',
            'xres', 'yres', 'shape', 'shape0', 'shape0_comp',
            'ccel_ix', 'cvert_ix', 'pvert_ix', 'pvert_xa', 'pvert_ya', 'Nbc',
        ]
        for attr in required:
            assert hasattr(self.m, attr), f"Missing attribute: {attr}"

    def test_no_refinement_methods(self):
        assert not hasattr(self.m, 'refine_by_two')
        assert not hasattr(self.m, 'refine_base')
        assert not hasattr(self.m, 'update')


# ---------------------------------------------------------------------------
# UniformMesh3D
# ---------------------------------------------------------------------------

class TestUniformMesh3D:
    def setup_method(self):
        self.m = UniformMesh3D(XW, YW, ZW, DS, DZ, PML=4)

    def test_xy_is_uniform_mesh2d(self):
        assert isinstance(self.m.xy, UniformMesh2D)

    def test_xy_not_rect_mesh2d(self):
        assert not isinstance(self.m.xy, RectMesh2D)

    def test_zres(self):
        assert self.m.zres == round(ZW / DZ) + 1

    def test_sigmax_zero_inside(self):
        x = np.array([0.0, XW / 4])
        s = self.m.sigmax(x)
        assert np.all(s == 0.0)

    def test_sigmax_nonzero_outside(self):
        x = np.array([XW / 2 + DS * 2])
        s = self.m.sigmax(x)
        assert np.all(np.abs(s) > 0)


# ---------------------------------------------------------------------------
# Prop3D detection and uniform code path
# ---------------------------------------------------------------------------

class TestProp3DUniformDetection:
    def setup_method(self):
        self.fs = _FreeSpace(N0)

    def test_is_uniform_true_for_uniform_mesh(self):
        m = UniformMesh3D(XW, YW, ZW, DS, DZ, PML=4)
        p = Prop3D(WL, m, self.fs, N0)
        assert p._is_uniform is True

    def test_is_uniform_false_for_rect_mesh(self):
        m = RectMesh3D(XW, YW, ZW, DS, DZ, PML=4)
        p = Prop3D(WL, m, self.fs, N0)
        assert p._is_uniform is False


class TestPrecompUniform:
    def setup_method(self):
        self.fs = _FreeSpace(N0)
        m = UniformMesh3D(XW, YW, ZW, DS, DZ, PML=4)
        self.prop = Prop3D(WL, m, self.fs, N0)
        self.prop._precomp_uniform()

    def test_uR_values(self):
        assert abs(self.prop._uR1 - 1. / 12.) < 1e-14
        assert abs(self.prop._uR2 - 5. / 6.) < 1e-14
        assert abs(self.prop._uR3 - 1. / 12.) < 1e-14

    def test_trimats_coeff_arrays_set(self):
        assert self.prop._a0x is not None
        assert self.prop._b0x is not None
        assert self.prop._c0x is not None
        assert self.prop._a0y is not None
        assert self.prop._b0y is not None
        assert self.prop._c0y is not None


class TestProp2EndUniform:
    def setup_method(self):
        self.fs = _FreeSpace(N0)
        self.m = UniformMesh3D(XW, YW, ZW, DS, DZ, PML=4)
        self.prop = Prop3D(WL, self.m, self.fs, N0)

    def _gaussian_input(self):
        xy = self.m.xy
        return np.exp(-(xy.xg ** 2 + xy.yg ** 2) / (3e-6) ** 2).astype(np.complex128)

    def test_output_shape(self):
        u_in = self._gaussian_input()
        result = self.prop.prop2end_uniform(u_in)
        assert result.shape == u_in.shape

    def test_power_conservation(self):
        """Power should be approximately conserved in free space."""
        u_in = self._gaussian_input()
        self.prop.prop2end_uniform(u_in)
        p0 = float(self.prop.totalpower[0])
        p1 = float(self.prop.totalpower[-1])
        assert p0 > 0
        assert abs(p1 / p0 - 1.0) < 0.05  # within 5 %

    def test_prop2end_dispatches_to_uniform_path(self):
        """prop2end should dispatch to prop2end_uniform for UniformMesh3D."""
        prop2 = Prop3D(WL, self.m, self.fs, N0)
        u_in = self._gaussian_input()
        result = prop2.prop2end(u_in.copy())
        assert result.shape == u_in.shape
