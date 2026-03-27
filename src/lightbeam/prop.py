
import numpy as _np
from numpy import pi, s_
from lightbeam.xp import xp, to_cpu, to_device
from lightbeam.mesh import RectMesh3D, RectMesh2D, UniformMesh3D, UniformMesh2D
import lightbeam.optics as optics
from lightbeam.misc import timeit, overlap, normalize,printProgressBar, overlap_nonu, norm_nonu

### to do ###

## performance

# maybe adaptive z stepping
# get a better refinement criterion -- now weighting partially by 2nd deriv. still could use some work

# more efficient ways to store arrays with many repeated values -- sparse data structure?

## readability

def genc(shape):
    return xp.empty(shape,dtype=xp.complex128,order='F')

def genf(shape):
    return xp.empty(shape,dtype=xp.complex128,order='F')

def tri_solve_vec(a, b, c, r, g, u):
    """
    Vectorised Thomas algorithm (pure xp — works on NumPy CPU and CuPy GPU).
    Solves M independent tridiagonal systems simultaneously.
    a, b, c : (N, M) complex – sub-, main-, super-diagonal
    r       : (N, M) complex – right-hand side
    g       : (N, M) complex – scratch array (pre-allocated, passed in)
    u       : (N, M) complex – solution (written in-place)
    """
    N = a.shape[0]
    beta = b[0]
    u[0] = r[0] / beta
    for j in range(1, N):
        g[j] = c[j-1] / beta
        beta = b[j] - a[j] * g[j]
        u[j] = (r[j] - a[j] * u[j-1]) / beta
    for j in range(N - 1):
        k = N - 2 - j
        u[k] -= g[k+1] * u[k+1]

class Prop3D:
    '''beam propagator. employs finite-differences beam propagation with PML as the boundary condition. works on an adaptive mesh'''
    def __init__(self, wl0, _mesh, optical_system:optics.OpticSys, n0):
        
        xymesh = _mesh.xy

        self.wl0 = wl0
        self.k0 = k0 = 2.*pi/wl0
        self.k02 = k02 = k0*k0
        self._mesh = _mesh
        self.n0 = n0

        self.sig = sig = -2.j*k0*n0/_mesh.dz

        self.field = None

        # detect whether the mesh is a uniform (non-adaptive) mesh
        self._is_uniform = isinstance(_mesh, UniformMesh3D)

        self.optical_system = optical_system
        self.optical_system.set_sampling(xymesh)
        self.nb2 = nb2 = optical_system.nb2
        self.n02 = n02 = n0*n0
        
        ## things that will be set during computation

        self.xgrid_cor_facs = [[]]*3
        self.ygrid_cor_facs = [[]]*3

        self.xgrid_cor_mask = []
        self.ygrid_cor_mask = []

        ## precomputing some stuff ##

        Rx,Tupx,Tdox,Ry,Tupy,Tdoy = self.calculate_PML_mats()   

        dx02 = _mesh.xy.dx0**2
        dy02 = _mesh.xy.dy0**2

        K = k02*(nb2-n02)
        n02 = xp.power(n0,2)

        ## coeff matrices of tridiagonal system, updated periodically

        self._a0x = None
        self._b0x = None
        self._c0x = None
        self._a0y = None
        self._b0y = None
        self._c0y = None

        self.a0x_ = None
        self.b0x_ = None
        self.c0x_ = None
        self.a0y_ = None
        self.b0y_ = None
        self.c0y_ = None

        ## same as above but in PML zone
        
        self._apmlx = sig/12. - 0.5/dx02*Tdox - K/48.
        self._bpmlx = 5./6.*sig + Rx/dx02 - 5./24. * K
        self._cpmlx = sig/12. - 0.5/dx02*Tupx - K/48.

        self.apmlx_ = sig/12. + 0.5/dx02*Tdox + K/48.
        self.bpmlx_ = 5./6.*sig - Rx/dx02 + 5./24. * K
        self.cpmlx_ = sig/12. + 0.5/dx02*Tupx + K/48.

        self._apmly = sig/12. - 0.5/dy02*Tdoy - K/48.
        self._bpmly = 5./6.*sig + Ry/dy02 - 5./24. * K
        self._cpmly = sig/12. - 0.5/dy02*Tupy - K/48.

        self.apmly_ = sig/12. + 0.5/dy02*Tdoy + K/48.
        self.bpmly_ = 5./6.*sig - Ry/dy02 + 5./24. * K
        self.cpmly_ = sig/12. + 0.5/dy02*Tupy + K/48.
        
        self.half_dz = _mesh.dz/2.

        self.power = xp.empty((_mesh.zres,))
        self.totalpower = xp.empty((_mesh.zres,))

    def allocate_mats(self):
        sx,sy = self._mesh.xy.xg.shape,self._mesh.xy.yg.T.shape
        _trimatsx = (genc(sx),genc(sx),genc(sx))
        _trimatsy = (genc(sy),genc(sy),genc(sy))

        rmatx,rmaty = genc(sx),genc(sy)
        gx = genc(sx)
        gy = genc(sy)

        fill = self.nb2*self.k02

        IORsq__ = xp.full(sx,fill,dtype=xp.float64)
        _IORsq_ = xp.full(sx,fill,dtype=xp.float64)
        __IORsq = xp.full(sx,fill,dtype=xp.float64)

        return _trimatsx,rmatx,gx,_trimatsy,rmaty,gy,IORsq__,_IORsq_,__IORsq

    def check_z_inv(self):
        return self.optical_system.z_invariant

    def set_IORsq(self,out,z,xg=None,yg=None):
        #premultiply by k02 so we don't have to keep doing it later
        self.optical_system.set_IORsq(out,z,xg,yg,coeff=self.k02)

    def calculate_PML_mats(self):
        '''As per textbook <Beam Propagation Method for Design of Optical Waveguide Devices> , 
        calculate the matrices R, T_j+1, and T_j-1 in the PML zone. We assume that the 
        the PML's refractive index will be constant, equal to the background index.
        '''

        m = self._mesh
        xy = m.xy

        xverts = xy.pvert_xa

        sdox = m.sigmax(xverts-xy.dx0)
        sx = m.sigmax(xverts)
        supx = m.sigmax(xverts+xy.dx0)

        yverts = xy.pvert_ya
        sdoy = m.sigmay(yverts-xy.dy0)
        sy = m.sigmay(yverts)
        supy = m.sigmay(yverts+xy.dy0)

        Qdox = 1./(1.+1.j*sdox*self.nb2)
        Qx = 1./(1.+1.j*sx*self.nb2)
        Qupx = 1./(1.+1.j*supx*self.nb2)

        Tupx = 0.5 * Qx * (Qx+Qupx)
        Tdox = 0.5 * Qx * (Qx+Qdox)
        Rx = 0.25 * Qx * (Qdox+2*Qx+Qupx)       

        Qdoy = 1./(1.+1.j*sdoy*self.nb2)
        Qy = 1./(1.+1.j*sy*self.nb2)
        Qupy = 1./(1.+1.j*supy*self.nb2)

        Tupy= 0.5 * Qy * (Qy+Qupy)
        Tdoy = 0.5 * Qy * (Qy+Qdoy)
        Ry = 0.25 * Qy * (Qdoy+2*Qy+Qupy)  

        return (Rx,Tupx,Tdox,Ry,Tupy,Tdoy)

    def update_grid_cor_facs(self,which='x'):
        xy = self._mesh.xy
        ix = xy.cvert_ix
        if which=='x':
            r = xy.rxa[ix]
            self.xgrid_cor_imask = xp.where(r[1:-1]!=1)[0]
        else:
            r = xy.rya[ix]
            self.ygrid_cor_imask = xp.where(r[1:-1]!=1)[0]
        r2 = r*r

        R1 = (r2 + r -1)/(6*r*(r+1))
        R2 =  (r2 + 3*r + 1)/(6*r)
        R3 = (-r2 + r + 1)/(6*(r+1))

        ## alternative values from paper

        #R1 = (3*r2 - 3*r + 1)/ (6*r*(r+1))
        #R2 = (-r2 + 7*r - 1)/(6*r)
        #R3 = (r2 - 3*r + 3)/(6*(r+1))

        if which=='x':
            self.xgrid_cor_facs[0] = R1
            self.xgrid_cor_facs[1] = R2
            self.xgrid_cor_facs[2] = R3
        else:
            self.ygrid_cor_facs[0] = R1
            self.ygrid_cor_facs[1] = R2
            self.ygrid_cor_facs[2] = R3

    def precomp_trimats(self,which='x'):
        ix = self._mesh.xy.cvert_ix
        s = self.sig    
        nu0 = -self.k02*self.n02

        if which == 'x':
            R1,R2,R3 = self.xgrid_cor_facs
            r = self._mesh.xy.rxa[ix]
            dla = self._mesh.xy.dxa[ix]
            self._a0x = s*R3[1:,None] - 1/(r[1:,None]+1)/(dla[1:,None]*dla[1:,None]) - 0.25*R3[1:,None]*nu0
            self._b0x = s*R2[:,None] + 1/r[:,None]/(dla[:,None]*dla[:,None]) - 0.25*R2[:,None]*nu0
            self._c0x = s*R1[:-1,None] - 1/r[:-1,None]/(r[:-1,None]+1)/(dla[:-1,None]*dla[:-1,None]) - 0.25*R1[:-1,None]*nu0
        else:
            R1,R2,R3 = self.ygrid_cor_facs
            r = self._mesh.xy.rya[ix]
            dla = self._mesh.xy.dya[ix]
            self._a0y = s*R3[1:,None] - 1/(r[1:,None]+1)/(dla[1:,None]*dla[1:,None]) - 0.25*R3[1:,None]*nu0
            self._b0y = s*R2[:,None] + 1/r[:,None]/(dla[:,None]*dla[:,None]) - 0.25*R2[:,None]*nu0
            self._c0y = s*R1[:-1,None] - 1/r[:-1,None]/(r[:-1,None]+1)/(dla[:-1,None]*dla[:-1,None]) - 0.25*R1[:-1,None]*nu0

    def _trimats(self,out,IORsq,which='x'):
        ''' calculate the tridiagonal matrices in the computational zone '''

        ix = self._mesh.xy.cvert_ix
        _IORsq = IORsq[ix]

        if which == 'x':
            R1,R2,R3 = self.xgrid_cor_facs
            r = self._mesh.xy.rxa[ix]
            dla = self._mesh.xy.dxa[ix]
            a,b,c = self._a0x,self._b0x,self._c0x
            
        else:
            R1,R2,R3 = self.ygrid_cor_facs
            r = self._mesh.xy.rya[ix]
            dla = self._mesh.xy.dya[ix]
            a,b,c = self._a0y,self._b0y,self._c0y
            
        _a,_b,_c = out

        s = self.sig

        _a[ix][1:] = a - 0.25*R3[1:,None]*_IORsq[:-1]
        _b[ix]     = b - 0.25*R2[:,None]*_IORsq
        _c[ix][:-1]= c - 0.25*R1[:-1,None]*_IORsq[1:]

        _a[ix][0] = s*R3[0] - 1. / ((r[0]+1) * dla[0]*dla[0]) - 0.25*R3[0]*(_IORsq[0]-self.n02*self.k02)
        _c[ix][-1] = s*R1[-1] - 1/r[-1]/(r[-1]+1)/(dla[-1]*dla[-1]) - 0.25*R1[-1]*(_IORsq[-1]-self.n02*self.k02)

    def rmat_pmlcorrect(self,_rmat,u,which='x'):

        if which == 'x':    
            apml,bpml,cpml = self.apmlx_,self.bpmlx_,self.cpmlx_
        else:
            apml,bpml,cpml = self.apmly_,self.bpmly_,self.cpmly_

        pix = self._mesh.xy.pvert_ix

        temp = xp.empty_like(_rmat[pix])

        temp[1:-1] = apml[1:-1,None]*u[pix-1][1:-1] + bpml[1:-1,None]*u[pix][1:-1] + cpml[1:-1,None]*u[pix+1][1:-1]

        temp[0] = bpml[0]*u[0] + cpml[0]*u[1]
        temp[-1] = apml[-1]*u[-2] + bpml[-1]*u[-1]

        _rmat[pix] = temp

    def rmat(self,_rmat,u,IORsq,which='x'):
        ix = self._mesh.xy.cvert_ix
        _IORsq = IORsq[ix]
        s = self.sig

        if which == 'x':    
            R1,R2,R3 = self.xgrid_cor_facs
            dla = self._mesh.xy.dxa[ix]
            r = self._mesh.xy.rxa[ix]
            a,b,c = self.a0x_,self.b0x_,self.c0x_
        else:
            R1,R2,R3 = self.ygrid_cor_facs
            dla = self._mesh.xy.dya[ix]
            r = self._mesh.xy.rya[ix]
            a,b,c = self.a0y_,self.b0y_,self.c0y_

        N = self.n02*self.k02
        m = s_[1:-1,None]

        _rmat[ix][1:-1] = (a + 0.25*R3[m]*_IORsq[:-2])*u[ix][:-2] + (b + 0.25*R2[m]*_IORsq[1:-1])*u[ix][1:-1] + (c + 0.25*R1[m]*_IORsq[2:])*u[ix][2:]

        _rmat[ix][0] = (s*R2[0] - 1/(r[0]*dla[0]**2 ) + 0.25*R2[0]*(_IORsq[0]-N))*u[0] + (s*R1[0] + 1/r[0]/(r[0]+1)/dla[0]**2 + 0.25*R1[0] * (_IORsq[1]-N) )*u[1]
        _rmat[ix][-1] =  (s*R3[-1] + 1. / ((r[-1]+1) * dla[-1]**2) + 0.25*R3[-1]*(_IORsq[-2]-N))*u[-2] + (s*R2[-1] - 1/(r[-1]*dla[-1]**2) + 0.25*R2[-1]*(_IORsq[-1]-N))*u[-1]

    def rmat_precomp(self,which='x'):
        ix = self._mesh.xy.cvert_ix
        s = self.sig
        n0 = -self.k02 * self.n02
        m = s_[1:-1,None]

        if which == 'x':
            R1,R2,R3 = self.xgrid_cor_facs
            r = self._mesh.xy.rxa[ix]
            dla = self._mesh.xy.dxa[ix]

            self.a0x_ = s*R3[m] + 1/(r[m]+1)/(dla[m]*dla[m]) + 0.25*R3[m]*n0
            self.b0x_ = s*R2[m] - 1/r[m]/(dla[m]*dla[m]) + 0.25*R2[m]*n0
            self.c0x_ = s*R1[m] + 1/r[m]/(r[m]+1)/(dla[m]*dla[m]) + 0.25*R1[m]*n0
        else:
            R1,R2,R3 = self.ygrid_cor_facs
            r = self._mesh.xy.rya[ix]
            dla = self._mesh.xy.dya[ix]

            self.a0y_ = s*R3[m] + 1/(r[m]+1)/(dla[m]*dla[m]) + 0.25*R3[m]*n0
            self.b0y_ = s*R2[m] - 1/r[m]/(dla[m]*dla[m]) + 0.25*R2[m]*n0
            self.c0y_ = s*R1[m] + 1/r[m]/(r[m]+1)/(dla[m]*dla[m]) + 0.25*R1[m]*n0

    def _pmlcorrect(self,_trimats,which='x'):
        ix = self._mesh.xy.pvert_ix
        _a,_b,_c = _trimats 
        
        if which=='x':
            _a[ix] = self._apmlx[:,None]
            _b[ix] = self._bpmlx[:,None]
            _c[ix] = self._cpmlx[:,None]
        else:
            _a[ix] = self._apmly[:,None]
            _b[ix] = self._bpmly[:,None]
            _c[ix] = self._cpmly[:,None]

    def _precomp_uniform(self):
        """Precompute tridiagonal coefficients for a uniform mesh.

        For a uniform grid every step ratio equals 1, so the Shibayama
        grid-correction factors (R1, R2, R3 from Shibayama et al., J. Lightwave
        Technol. 15, 1996) collapse to the constant scalars R1 = R3 = 1/12 and
        R2 = 5/6, evaluated from the general formulas at r = 1:
            R1 = (r² + r − 1) / (6·r·(r+1))  →  1/12
            R2 = (r² + 3r + 1) / (6·r)         →  5/6
            R3 = (−r² + r + 1) / (6·(r+1))     →  1/12
        These are stored on the instance as ``_uR1``, ``_uR2``, ``_uR3`` and
        the per-direction coefficient arrays are computed via the standard methods
        (which read from ``xgrid_cor_facs`` / ``ygrid_cor_facs``).
        """
        # Shibayama grid-correction factors for a uniform grid (r = 1 everywhere)
        self._uR1 = 1./12.   # (1 + 1 - 1) / (6*1*(1+1))
        self._uR2 = 5./6.    # (1 + 3 + 1) / (6*1)
        self._uR3 = 1./12.   # (-1 + 1 + 1) / (6*(1+1))

        # Populate grid_cor_facs with the constant values so that the
        # standard precomp_trimats / rmat_precomp helpers work unchanged.
        self.update_grid_cor_facs('x')
        self.update_grid_cor_facs('y')

        self.precomp_trimats('x')
        self.precomp_trimats('y')

        self.rmat_precomp('x')
        self.rmat_precomp('y')

    def _trimats_uniform(self, out, IORsq, which='x'):
        """Simplified tridiagonal matrix assembly for a uniform mesh.

        Identical in purpose to :meth:`_trimats` but exploits the fact that
        the grid-correction factors are the constant scalars stored in
        ``_uR1``/``_uR2``/``_uR3``, eliminating per-node array indexing.
        """
        ix = self._mesh.xy.cvert_ix
        _IORsq = IORsq[ix]
        R1, R2, R3 = self._uR1, self._uR2, self._uR3
        s = self.sig

        if which == 'x':
            a, b, c = self._a0x, self._b0x, self._c0x
            d2 = self._mesh.xy.dx0 ** 2
        else:
            a, b, c = self._a0y, self._b0y, self._c0y
            d2 = self._mesh.xy.dy0 ** 2

        _a, _b, _c = out

        # Interior nodes (same formula for all; a0*, b0*, c0* are already
        # 2-D arrays with constant rows for the uniform case)
        _a[ix][1:] = a - 0.25 * R3 * _IORsq[:-1]
        _b[ix]     = b - 0.25 * R2 * _IORsq
        _c[ix][:-1]= c - 0.25 * R1 * _IORsq[1:]

        # Boundary nodes — identical formula thanks to r=1, d=const
        _a[ix][0]  = s*R3 - 0.5/d2 - 0.25*R3 * (_IORsq[0]  - self.n02 * self.k02)
        _c[ix][-1] = s*R1 - 0.5/d2 - 0.25*R1 * (_IORsq[-1] - self.n02 * self.k02)

    def rmat_uniform(self, _rmat, u, IORsq, which='x'):
        """Simplified RHS assembly for a uniform mesh.

        Identical in purpose to :meth:`rmat` but uses the constant scalar
        grid-correction factors for the uniform grid.
        """
        ix = self._mesh.xy.cvert_ix
        _IORsq = IORsq[ix]
        R1, R2, R3 = self._uR1, self._uR2, self._uR3
        s = self.sig
        N = self.n02 * self.k02
        m = s_[1:-1, None]

        if which == 'x':
            a, b, c = self.a0x_, self.b0x_, self.c0x_
            d2 = self._mesh.xy.dx0 ** 2
        else:
            a, b, c = self.a0y_, self.b0y_, self.c0y_
            d2 = self._mesh.xy.dy0 ** 2

        # Interior
        _rmat[ix][1:-1] = (
            (a + 0.25*R3 * _IORsq[:-2]) * u[ix][:-2]
            + (b + 0.25*R2 * _IORsq[1:-1]) * u[ix][1:-1]
            + (c + 0.25*R1 * _IORsq[2:]) * u[ix][2:]
        )

        # Boundary (r=1, d²=const → same as general formula with r=1)
        _rmat[ix][0] = (
            (s*R2 - 1./d2 + 0.25*R2 * (_IORsq[0]  - N)) * u[0]
            + (s*R1 + 0.5/d2 + 0.25*R1 * (_IORsq[1]  - N)) * u[1]
        )
        _rmat[ix][-1] = (
            (s*R3 + 0.5/d2 + 0.25*R3 * (_IORsq[-2] - N)) * u[-2]
            + (s*R2 - 1./d2 + 0.25*R2 * (_IORsq[-1] - N)) * u[-1]
        )

    @timeit 
    def prop2end(self,_u,monitor_func=None,ref_val=5.e-6,remesh_every=20,dynamic_n0 = False,fplanewidth=0,writeto=None,xyslice=None,zslice=None):
        
        # Dispatch to the simplified uniform code path when the mesh is uniform.
        if self._is_uniform:
            return self.prop2end_uniform(
                _u,
                xyslice=xyslice,
                zslice=zslice,
                u1_func=monitor_func,
                writeto=writeto,
                dynamic_n0=dynamic_n0,
                fplanewidth=fplanewidth,
            )

        _mesh = self._mesh
        PML = _mesh.PML

        if writeto is not None:
            # then we need to save some field information
            if xyslice is None and zslice is None:
                # save everything
                za_keep = _mesh.za
                shape = (len(za_keep),*_mesh.xg.shape)
            else:
                # save a slice
                za_keep = _mesh.za[zslice]
                shape = (len(za_keep),*_mesh.xg[xyslice].shape)

            self.field = xp.zeros(shape,dtype=xp.complex128)

        #pull xy mesh
        xy = _mesh.xy
        dx,dy = xy.dx0,xy.dy0

        if fplanewidth == 0:
            xa_in = xp.linspace(-_mesh.xw/2,_mesh.xw/2,xy.shape0_comp[0])
            ya_in = xp.linspace(-_mesh.yw/2,_mesh.yw/2,xy.shape0_comp[1])
        else:
            xa_in = xp.linspace(-fplanewidth/2,fplanewidth/2,xy.shape0_comp[0])
            ya_in = xp.linspace(-fplanewidth/2,fplanewidth/2,xy.shape0_comp[1])

        dx0 = xa_in[1]-xa_in[0]
        dy0 = ya_in[1]-ya_in[0]

        # u can either be a field or a function that generates a field.
        # the latter option allows for coarse base grids to be used 
        # without being penalized by forcing the use of a low resolution
        # launch field

        if hasattr(_u, '__array__') or hasattr(_u, '__cuda_array_interface__'):

            _power = overlap(_u,_u)
            print('input power: ',_power)

            # normalize the field, preserving the input power. accounts for grid resolution
            normalize(_u,weight=dx0*dy0,normval=_power)

            #resample the field onto the smaller xy mesh (in the smaller mesh's computation zone)
            u0 = xy.resample_complex(_u,xa_in,ya_in,xy.xa[PML:-PML],xy.ya[PML:-PML])

            _power2 = overlap(u0,u0,dx*dy)

            #now we pad w/ zeros to extend it into the PML zone
            u0 = xp.pad(u0,((PML,PML),(PML,PML)))

            #initial mesh refinement
            xy.refine_base(u0,ref_val)

            weights = xy.get_weights()

            #now resample the field onto the smaller *non-uniform* xy mesh
            u = xy.resample_complex(_u,xa_in,ya_in,xy.xa[PML:-PML],xy.ya[PML:-PML])
            u = xp.pad(u,((PML,PML),(PML,PML)))

            #do another norm to correct for the slight power change you get when resampling. I measure 0.1% change for psflo. should check again
            norm_nonu(u,weights,_power2)
        
        elif callable(_u):
            # must be of the form u(x,y)
            u0 = _u(xy.xg,xy.yg)
            _power = overlap(u0,u0)
            print('input power: ',_power)
            
            # normalize the field, preserving the input power. accounts for grid resolution
            normalize(u0,weight=dx0*dy0,normval=_power)

            # do an initial mesh refinement
            xy.refine_base(u0,ref_val)

            # compute the field on the nonuniform grid
            u = norm_nonu(_u(xy.xg,xy.yg),xy.get_weights(),_power)

        else:
            raise Exception("unsupported type for argument u in prop2end()")

        counter = 0
        total_iters = self._mesh.zres

        print("propagating field...")
        
        __z = 0
        z__ = 0

        #step 0 setup

        self.update_grid_cor_facs('x')
        self.update_grid_cor_facs('y')

        # initial array allocation
        _trimatsx,rmatx,gx,_trimatsy,rmaty,gy,IORsq__,_IORsq_,__IORsq = self.allocate_mats()

        self.precomp_trimats('x')
        self.precomp_trimats('y')

        self.rmat_precomp('x')
        self.rmat_precomp('y')

        self._pmlcorrect(_trimatsx,'x')
        self._pmlcorrect(_trimatsy,'y')

        #get the current IOR dist
        self.set_IORsq(IORsq__,z__)

        #plt.figure(frameon=False)
        #plt.imshow(xy.get_base_field(IORsq__))
        #plt.show()

        print("initial shape: ",xy.shape)
        for i in range(total_iters):       
            if i%20 == 0: 
                printProgressBar(i,total_iters-1)
            u0 = xy.get_base_field(u)
            u0c = xp.conj(u0)
            weights = xy.get_weights()
            
            ## Total power monitor ##
            self.totalpower[i] = overlap_nonu(u,u,weights)
            #print(self.totalpower[i])

            ## Other monitors ##
            if monitor_func is not None:
                monitor_field = norm_nonu(monitor_func(xy.xg,xy.yg),weights)
                self.power[i] = xp.power(overlap_nonu(u,monitor_field,weights),2)

            _z_ = z__ + _mesh.half_dz
            __z = z__ + _mesh.dz
            
            if self.field is not None and counter < len(za_keep) and __z == za_keep[counter]:
                # record the field
                self.field[counter] = u0[xyslice]
                counter += 1

            #avoid remeshing on step 0 
            if (i+1)%remesh_every== 0:

                ## update the effective index
                if dynamic_n0:
                    #update the effective index
                    base = xy.get_base_field(IORsq__)
                    self.n02 = xy.dx0*xy.dy0*float(xp.real(xp.sum(u0c*u0*base)))/self.k02

                oldxm,oldxM = xy.xm,xy.xM
                oldym,oldyM = xy.ym,xy.yM

                oldxw,oldyw = xy.xw,xy.yw

                new_xw,new_yw = oldxw,oldyw
                #expand the grid if necessary
                if _mesh.xwfunc is not None:
                    new_xw = _mesh.xwfunc(__z)
                if _mesh.ywfunc is not None:
                    new_yw = _mesh.ywfunc(__z)

                new_xw, new_yw = xy.snapto(new_xw,new_yw)

                xy.reinit(new_xw,new_yw) #set grid back to base res with new dims

                if (xy.xw > oldxw or xy.yw > oldyw):
                    #now we need to pad u,u0 with zeros to make sure it matches the new space
                    xpad = int((xy.shape0[0]-u0.shape[0])/2)
                    ypad = int((xy.shape0[1]-u0.shape[1])/2)

                    u = xp.pad(u,((xpad,xpad),(ypad,ypad)))
                    u0 = xp.pad(u0,((xpad,xpad),(ypad,ypad)))

                    #pad coord arrays to do interpolation
                    xy.xa_last = xp.hstack( ( xp.linspace(xy.xm,oldxm-dx,xpad) , xy.xa_last , xp.linspace(oldxM + dx, xy.xM,xpad) ) )
                    xy.ya_last = xp.hstack( ( xp.linspace(xy.ym,oldym-dy,ypad) , xy.ya_last , xp.linspace(oldyM + dy, xy.yM,ypad) ) )
                   
                #subdivide into nonuniform grid
                xy.refine_base(u0,ref_val)

                #interp the field to the new grid   
                u = xy.resample_complex(u)

                #give the grid to the optical sys obj so it can compute IORs
                self.optical_system.set_sampling(xy)

                #compute nonuniform grid correction factors R_i
                self.update_grid_cor_facs('x')
                self.update_grid_cor_facs('y')

                # grid size has changed, so now we need to reallocate arrays for at least the next remesh_period iters
                _trimatsx,rmatx,gx,_trimatsy,rmaty,gy,IORsq__,_IORsq_,__IORsq = self.allocate_mats()

                #get the current IOR dist
                self.set_IORsq(IORsq__,z__)
                
                #precompute things that will be reused
                self.precomp_trimats('x')
                self.precomp_trimats('y')

                self.rmat_precomp('x')
                self.rmat_precomp('y')
                
                self._pmlcorrect(_trimatsx,'x')
                self._pmlcorrect(_trimatsy,'y')

            self.set_IORsq(_IORsq_,_z_,)
            self.set_IORsq(__IORsq,__z)

            self.rmat(rmatx,u,IORsq__,'x')
            self.rmat_pmlcorrect(rmatx,u,'x')

            self._trimats(_trimatsx,_IORsq_,'x')
            self._trimats(_trimatsy,__IORsq.T,'y')

            tri_solve_vec(_trimatsx[0],_trimatsx[1],_trimatsx[2],rmatx,gx,u)

            self.rmat(rmaty,u.T,_IORsq_.T,'y')
            self.rmat_pmlcorrect(rmaty,u.T,'y')

            tri_solve_vec(_trimatsy[0],_trimatsy[1],_trimatsy[2],rmaty,gy,u.T)

            z__ = __z
            if (i+2)%remesh_every != 0:
                IORsq__[:,:] = __IORsq
  
        print("final total power",float(self.totalpower[-1]))
        
        if writeto:
            _np.save(writeto,to_cpu(self.field))
        return u,u0

    @timeit 
    def prop2end_uniform(self,u,xyslice=None,zslice=None,u1_func=None,writeto=None,dynamic_n0 = False,fplanewidth=0):
        """Propagate a field to the end of the structure using a uniform mesh.

        This method is called automatically by :meth:`prop2end` when the
        propagator was constructed with a :class:`~lightbeam.mesh.UniformMesh3D`.
        It uses the simplified :meth:`_trimats_uniform` / :meth:`rmat_uniform`
        stencils (constant R1=R3=1/12, R2=5/6) and never re-meshes.
        """
        _mesh = self._mesh
        PML = _mesh.PML

        if not (xyslice is None and zslice is None):
            za_keep = _mesh.za[zslice]
            if hasattr(za_keep, '__len__'):
                minz, maxz = za_keep[0],za_keep[-1]
                shape = (len(za_keep),*_mesh.xg[xyslice].shape)
            else:
                raise Exception('uhh not implemented')
            
            self.field = xp.zeros(shape,dtype=xp.complex128)

        if fplanewidth == 0:
            xa_in = xp.linspace(-_mesh.xw/2,_mesh.xw/2,u.shape[0])
            ya_in = xp.linspace(-_mesh.yw/2,_mesh.yw/2,u.shape[1])
        else:
            xa_in = xp.linspace(-fplanewidth/2,fplanewidth/2,u.shape[0])
            ya_in = xp.linspace(-fplanewidth/2,fplanewidth/2,u.shape[1])

        dx0 = xa_in[1]-xa_in[0]
        dy0 = ya_in[1]-ya_in[0]

        _power = overlap(u,u)
        print('input power: ',_power)

        # normalize the field, preserving the input power. accounts for grid resolution
        normalize(u,weight=dx0*dy0,normval=_power)

        __z = 0

        #pull xy mesh
        xy = _mesh.xy
        dx,dy = xy.dx0,xy.dy0

        #resample the field onto the smaller xy mesh (in the smaller mesh's computation zone)
        u0 = xy.resample_complex(u,xa_in,ya_in,xy.xa[PML:-PML],xy.ya[PML:-PML])

        #now we pad w/ zeros to extend it into the PML zone
        u0 = xp.pad(u0,((PML,PML),(PML,PML)))

        counter = 0
        total_iters = self._mesh.zres

        print("propagating field...")

        z__ = 0

        # One-time setup: precompute constant coefficients for the uniform mesh.
        self._precomp_uniform()

        # initial array allocation
        _trimatsx,rmatx,gx,_trimatsy,rmaty,gy,IORsq__,_IORsq_,__IORsq = self.allocate_mats()

        self._pmlcorrect(_trimatsx,'x')
        self._pmlcorrect(_trimatsy,'y')

        #get the current IOR dist
        self.set_IORsq(IORsq__,z__)

        weights = xy.get_weights()

        print("initial shape: ",xy.shape)
        for i in range(total_iters):       
            if i%20 == 0: 
                printProgressBar(i,total_iters-1)

            ## Total power monitor ##
            self.totalpower[i] = overlap_nonu(u0,u0,weights)

            ## Other monitors ##
            if u1_func is not None:
                lp = norm_nonu(u1_func(xy.xg,xy.yg),weights)
                self.power[i] = xp.power(overlap_nonu(u0,lp,weights),2)

            _z_ = z__ + _mesh.half_dz
            __z = z__ + _mesh.dz
            
            if self.field is not None and (minz<=__z<=maxz):
                ix0,ix1,ix2,ix3 = _mesh.get_loc() 
                mid = int(u0.shape[1]/2)

                self.field[counter][ix0:ix1+1] = u0[:,mid] ## FIX ##
                counter+=1

            self.set_IORsq(_IORsq_,_z_,)
            self.set_IORsq(__IORsq,__z)

            self.rmat_uniform(rmatx,u0,IORsq__,'x')
            self.rmat_pmlcorrect(rmatx,u0,'x')

            self._trimats_uniform(_trimatsx,_IORsq_,'x')
            self._trimats_uniform(_trimatsy,__IORsq.T,'y')

            tri_solve_vec(_trimatsx[0],_trimatsx[1],_trimatsx[2],rmatx,gx,u0)

            self.rmat_uniform(rmaty,u0.T,_IORsq_.T,'y')
            self.rmat_pmlcorrect(rmaty,u0.T,'y')

            tri_solve_vec(_trimatsy[0],_trimatsy[1],_trimatsy[2],rmaty,gy,u0.T)

            z__ = __z
            IORsq__[:,:] = __IORsq
  
        print("final total power",float(self.totalpower[-1]))
        
        if writeto:
            _np.save(writeto,to_cpu(self.field))
        return u0

