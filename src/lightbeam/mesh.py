import numpy as _np
from numpy import s_
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt
from itertools import chain
import math
from lightbeam.xp import xp, to_cpu

## to do

# figure out how to normalize ucrit
# combine the different remeshing options into a single func with a switch argument
# remeshing process is a little inefficient. not sure how to speed up though

TOL=1e-12

class RectMesh2D:
    ''' transverse adapative mesh class '''

    def __init__(self,xw,yw,dx,dy,Nbc=4):

        self.max_iters = 6
        self.Nbc = Nbc
        self.dx0,self.dy0 = dx,dy

        self.xa = None
        self.ya = None

        self.xg = None
        self.yg = None

        self.ccel_ix = s_[Nbc+2:-Nbc-2]

        ###??? idk why these have to overlap but the pml doesn't work any other way
        self.cvert_ix = s_[Nbc:-Nbc]
        self.pvert_ix = xp.hstack((xp.arange(Nbc+1),xp.arange(-Nbc-1,0)))

        self.reinit(xw,yw)
        self.update(self.rfacxa,self.rfacya)

    def reinit(self,xw,yw):
        self.xa_last,self.ya_last = self.xa,self.ya

        dx,dy = self.dx0,self.dy0
        Nbc = self.Nbc

        self.shape0_comp = (int(round(xw/dx)+1),int(round(yw/dy)+1))
        xres,yres = self.shape0_comp[0] + 2*Nbc , self.shape0_comp[1] + 2*Nbc

        self.shape0 = (xres,yres)
        self.shape = (xres,yres)

        self.xw,self.yw = xw,yw

        #anchor point of the adaptive grid
        self.xm = -xw/2-Nbc*dx
        self.ym = -yw/2-Nbc*dy
        self.xM = xw/2+Nbc*dx
        self.yM = yw/2+Nbc*dy

        self.xa0 = xp.linspace(-xw/2-Nbc*dx,xw/2+Nbc*dx,xres)
        self.ya0 = xp.linspace(-yw/2-Nbc*dy,yw/2+Nbc*dy,yres)

        self.xix_base = xp.arange(xres)
        self.yix_base = xp.arange(yres)

        self.dxa = xp.full(xres,dx)
        self.dya = xp.full(yres,dy)

        self.xa = self.xa0 = xp.linspace(-xw/2-Nbc*dx,xw/2+Nbc*dx,xres)
        self.ya = self.ya0 = xp.linspace(-yw/2-Nbc*dy,yw/2+Nbc*dy,yres)

        self.pvert_xa = self.xa0[self.pvert_ix]
        self.pvert_ya = self.ya0[self.pvert_ix]

        self.rfacxa = self.rfacxa0 = xp.full(xres-1,1)
        self.rfacya = self.rfacya0 = xp.full(yres-1,1)

    def snapto(self,xw,yw):
        xwr = 2*math.ceil(xw/2/self.dx0)
        ywr = 2*math.ceil(yw/2/self.dy0)

        #round
        xw = xwr*self.dx0
        yw = ywr*self.dy0
        return xw, yw

    def dxa2xa(self,dxa):
        N = len(dxa)
        out = xp.zeros(N+1)
        xp.cumsum(dxa,out=out[1:])
        return out + self.xm

    def update(self,rfacxa,rfacya):
        self.rfacxa = rfacxa
        self.rfacya = rfacya

        xix_base = xp.empty(len(rfacxa)+1,dtype=int)
        yix_base = xp.empty(len(rfacya)+1,dtype=int)

        xix_base[0] = 0
        yix_base[0] = 0

        xix_base[1:] = xp.cumsum(rfacxa)
        yix_base[1:] = xp.cumsum(rfacya)

        self.xix_base = xix_base[self.xix_base]
        self.yix_base = yix_base[self.yix_base]

        new_dxa = xp.repeat(self.dxa[1:]/rfacxa,rfacxa)
        new_dya = xp.repeat(self.dya[1:]/rfacya,rfacya)

        new_xa = self.dxa2xa(new_dxa)
        new_ya = self.dxa2xa(new_dya)

        self.xa = new_xa
        self.ya = new_ya

        rxa = xp.empty_like(self.xa,dtype=float)
        rxa[1:-1] = new_dxa[1:]/new_dxa[:-1]
        rxa[0] = 1
        rxa[-1] = 1
        self.rxa = rxa

        rya = xp.empty_like(self.ya,dtype=float)
        rya[1:-1] = new_dya[1:]/new_dya[:-1]
        rya[0] = 1
        rya[-1] = 1
        self.rya = rya

        self.dxa = xp.empty_like(self.xa)
        self.dxa[1:] = new_dxa
        self.dxa[0] = self.dxa[1]

        self.dya = xp.empty_like(self.ya)
        self.dya[1:] = new_dya
        self.dya[0] = self.dya[1]

        self.xres,self.yres = len(self.xa),len(self.ya)

        self.xg,self.yg = xp.meshgrid(new_xa,new_ya,indexing='ij')

        #offset grids
        xhg = xp.empty(( self.xg.shape[0] + 1 , self.xg.shape[1] ))
        yhg = xp.empty(( self.yg.shape[0] , self.yg.shape[1] + 1 ))

        xhg[1:-1] = (self.xg[1:] + self.xg[:-1]) * 0.5
        yhg[:,1:-1] = (self.yg[:,1:] + self.yg[:,:-1]) * 0.5

        xhg[0] = self.xg[0] - self.dxa[0]*0.5
        xhg[-1] = self.xg[-1] + self.dxa[-1]*rxa[-1]*0.5

        yhg[:,0] = self.yg[:,0] - self.dya[0]*0.5
        yhg[:,-1] = self.yg[:,-1] + self.dya[-1]*self.rya[-1]*0.5

        self.xhg,self.yhg = xhg,yhg

        self.shape = (len(new_xa),len(new_ya))

    def get_weights(self):
        xhg,yhg = self.xhg,self.yhg
        weights = (xhg[1:] - xhg[:-1]) * (yhg[:,1:] - yhg[:,:-1])
        return weights

    def resample(self,u,xa=None,ya=None,newxa=None,newya=None):
        if xa is None or ya is None:
            out = RectBivariateSpline(to_cpu(self.xa_last),to_cpu(self.ya_last),to_cpu(u))(to_cpu(self.xa),to_cpu(self.ya))
        else:
            out = RectBivariateSpline(to_cpu(xa),to_cpu(ya),to_cpu(u))(to_cpu(newxa),to_cpu(newya))
        return xp.asarray(out)
    
    def resample_complex(self,u,xa=None,ya=None,newxa=None,newya=None):
        reals = xp.real(u)
        imags = xp.imag(u)
        reals = self.resample(reals,xa,ya,newxa,newya)
        imags = self.resample(imags,xa,ya,newxa,newya)
        return reals+1.j*imags

    def plot_mesh(self,reduce_by = 1,show=True):
        i=0
        for x in self.xa:
            if i%reduce_by == 0:
                plt.axhline(y=x,color='k',lw=0.5,alpha=0.5)
            i+=1
        i=0
        for y in self.ya:
            if i%reduce_by == 0:
                plt.axvline(x=y,color='k',lw=0.5,alpha=0.5)
            i+=1
        if show:
            plt.axis('equal')
            plt.show()
    
    def get_base_field(self,u):
        return u[self.xix_base].T[self.yix_base].T

    def _compute_refinement_factor(self,u0,crit_val):
        ''' given some electric field u0, compute values that corresponds to the degree of refinement required
            in the x and y subdivisions in the field. larger refinement factors should imply more grid refinement. 
            crit_val determines the normalization of this refinement factor - this is left as an argument
            to allow the degree of mesh subdivision to be controlled. note that the output refinement factors,
            _rx and _ry, essentially will act as booleans. wherever these factors are larger than 1, the grid will
            marked for subdivision.
        '''
        
        # the default scheme presented here sets the refinement factor to the geometric mean of field amplitude and 
        # field second derivative magnitude. convergence testing shows that this metric leads to faster convergence 
        # over using just field amplitude or second derivative magnitude alone. 
        # evidence for this is entirely empirical and comes from testing that is not 100% comprehensive.

        ix = self.ccel_ix

        # x second derivative estimation
        xdif2 = xp.empty_like(u0,dtype=xp.complex128)
        xdif2[1:-1] = u0[2:]+u0[:-2] - 2*u0[1:-1]
        xdif2[0] = xdif2[-1] = 0

        # y second derivative estimation
        ydif2 = xp.empty_like(u0,dtype=xp.complex128)
        ydif2[:,1:-1] = u0[:,2:]+u0[:,:-2] - 2*u0[:,1:-1]

        ydif2[:,0] = ydif2[:,-1] = 0

        # field amps
        umaxx = xp.sqrt(xp.max(xp.abs(u0),axis=1) * xp.max(xp.abs(xdif2),axis=1))
        umaxx = 0.5*(umaxx[1:]+umaxx[:-1])

        umaxy = xp.sqrt(xp.max(xp.abs(u0),axis=0) * xp.max(xp.abs(ydif2),axis=0))
        umaxy = 0.5*(umaxy[1:]+umaxy[:-1])

        _rx = umaxx[ix]*self.dxa[1:][ix]/crit_val
        _ry = umaxy[ix]*self.dya[1:][ix]/crit_val

        return _rx,_ry


    def refine_by_two(self,u0,crit_val):
        ''' uses a hybrid approach where cells tagged are based on the product of field amplitude
            and second derivative magnitude '''

        ix = self.ccel_ix

        _rx,_ry = self._compute_refinement_factor(u0,crit_val)

        rfacxa = xp.full(u0.shape[0]-1,1,dtype=int)
        rfacya = xp.full(u0.shape[1]-1,1,dtype=int)

        mask = (_rx>1)
        rfacxa[ix][mask] = 2

        mask = (_ry>1)
        rfacya[ix][mask] = 2

        xa_old = self.xa
        ya_old = self.ya

        self.update(rfacxa,rfacya)

        return self.resample_complex(u0,xa_old,ya_old,self.xa,self.ya)

    def refine_base(self,u0,ucrit):
        ''' iteratively apply refine_by_two to fully subdivide the simulation grid. '''
        for i in range(self.max_iters):
            u0 = self.refine_by_two(u0,ucrit)

class UniformMesh2D:
    """
    Strictly uniform transverse mesh for FD-BPM.

    Unlike RectMesh2D, the grid spacing is constant throughout and no adaptive
    refinement is ever performed.  This enables significant simplifications in
    the tridiagonal coefficient assembly: all grid-correction factors collapse
    to the uniform-grid values R1 = R3 = 1/12, R2 = 5/6, and the propagation
    loop in Prop3D detects this class and uses the faster uniform code path.
    """

    def __init__(self, xw, yw, dx, dy, Nbc=4):
        self.max_iters = 0  # no refinement iterations
        self.Nbc = Nbc
        self.dx0, self.dy0 = dx, dy

        self.xa = None
        self.ya = None
        self.xa_last = None
        self.ya_last = None

        self.ccel_ix = s_[Nbc+2:-Nbc-2]
        self.cvert_ix = s_[Nbc:-Nbc]
        self.pvert_ix = xp.hstack((xp.arange(Nbc+1), xp.arange(-Nbc-1, 0)))

        self.reinit(xw, yw)

    def reinit(self, xw, yw):
        dx, dy = self.dx0, self.dy0
        Nbc = self.Nbc

        self.shape0_comp = (int(round(xw/dx)+1), int(round(yw/dy)+1))
        xres, yres = self.shape0_comp[0] + 2*Nbc, self.shape0_comp[1] + 2*Nbc

        self.shape0 = (xres, yres)
        self.shape  = (xres, yres)
        self.xw, self.yw = xw, yw

        self.xm = -xw/2 - Nbc*dx
        self.ym = -yw/2 - Nbc*dy
        self.xM =  xw/2 + Nbc*dx
        self.yM =  yw/2 + Nbc*dy

        self.xa = xp.linspace(self.xm, self.xM, xres)
        self.ya = xp.linspace(self.ym, self.yM, yres)

        # uniform spacing — every cell has the same width
        self.dxa = xp.full(xres, dx)
        self.dya = xp.full(yres, dy)

        # all step ratios are exactly 1 (no non-uniformity)
        self.rxa = xp.ones(xres)
        self.rya = xp.ones(yres)

        self.xres, self.yres = xres, yres

        # identity index maps (no coarsening ever happens)
        self.xix_base = xp.arange(xres)
        self.yix_base = xp.arange(yres)

        self.pvert_xa = self.xa[self.pvert_ix]
        self.pvert_ya = self.ya[self.pvert_ix]

        self.xg, self.yg = xp.meshgrid(self.xa, self.ya, indexing='ij')

        # half-cell offset grids (used by get_weights and Prop3D)
        xhg = xp.empty((xres + 1, yres))
        yhg = xp.empty((xres, yres + 1))

        xhg[1:-1] = (self.xg[1:] + self.xg[:-1]) * 0.5
        yhg[:, 1:-1] = (self.yg[:, 1:] + self.yg[:, :-1]) * 0.5

        xhg[0] = self.xg[0] - dx * 0.5
        xhg[-1] = self.xg[-1] + dx * 0.5

        yhg[:, 0] = self.yg[:, 0] - dy * 0.5
        yhg[:, -1] = self.yg[:, -1] + dy * 0.5

        self.xhg, self.yhg = xhg, yhg

    def snapto(self, xw, yw):
        xwr = 2 * math.ceil(xw / 2 / self.dx0)
        ywr = 2 * math.ceil(yw / 2 / self.dy0)
        return xwr * self.dx0, ywr * self.dy0

    def get_weights(self):
        """Return cell-area weights. For a uniform grid this is a constant array."""
        return xp.full((self.xres, self.yres), self.dx0 * self.dy0)

    def resample(self, u, xa=None, ya=None, newxa=None, newya=None):
        if xa is None or ya is None:
            out = RectBivariateSpline(
                to_cpu(self.xa_last), to_cpu(self.ya_last), to_cpu(u)
            )(to_cpu(self.xa), to_cpu(self.ya))
        else:
            out = RectBivariateSpline(
                to_cpu(xa), to_cpu(ya), to_cpu(u)
            )(to_cpu(newxa), to_cpu(newya))
        return xp.asarray(out)

    def resample_complex(self, u, xa=None, ya=None, newxa=None, newya=None):
        reals = self.resample(xp.real(u), xa, ya, newxa, newya)
        imags = self.resample(xp.imag(u), xa, ya, newxa, newya)
        return reals + 1.j * imags

    def plot_mesh(self, reduce_by=1, show=True):
        i = 0
        for x in self.xa:
            if i % reduce_by == 0:
                plt.axhline(y=x, color='k', lw=0.5, alpha=0.5)
            i += 1
        i = 0
        for y in self.ya:
            if i % reduce_by == 0:
                plt.axvline(x=y, color='k', lw=0.5, alpha=0.5)
            i += 1
        if show:
            plt.axis('equal')
            plt.show()

    def get_base_field(self, u):
        """Identity — no coarsening for uniform mesh."""
        return u


class RectMesh3D:
    def __init__(self,xw,yw,zw,ds,dz,PML=4,xwfunc=None,ywfunc=None):
        '''base is a uniform mesh. can be refined'''
        self.xw,self.yw,self.zw = xw,yw,zw
        self.ds,self.dz = ds,dz
        self.xres,self.yres,self.zres = round(xw/ds)+1+2*PML, round(yw/ds)+1+2*PML, round(zw/dz)+1

        self.xa = xp.linspace(-xw/2-PML*ds,xw/2+PML*ds,self.xres)
        self.ya = xp.linspace(-yw/2-PML*ds, yw/2+PML*ds, self.yres)

        self.xg,self.yg = xp.meshgrid(self.xa,self.ya,indexing='ij')

        self.shape=(self.zres,self.xres,self.yres)

        if xwfunc is None:
            xy_xw = xw
        else:
            xy_xw = 2*math.ceil(xwfunc(0)/2/ds)*ds
        
        if ywfunc is None:
            xy_yw = yw
        else:
            xy_yw = 2*math.ceil(ywfunc(0)/2/ds)*ds
            
        self.xy = RectMesh2D(xy_xw,xy_yw,ds,ds,PML)

        self.za = xp.linspace(0,zw,self.zres)

        self.sigma_max = 5.+0.j #max (dimensionless) conductivity in PML layers
        self.PML = PML
        self.half_dz = dz/2.

        self.xwfunc = xwfunc
        self.ywfunc = ywfunc
    
    def get_loc(self ):

        xy = self.xy
        ix0 = int(xp.searchsorted(self.xa, xy.xm - TOL))
        ix1 = int(xp.searchsorted(self.xa, xy.xM - TOL))
        ix2 = int(xp.searchsorted(self.ya, xy.ym - TOL))
        ix3 = int(xp.searchsorted(self.ya, xy.yM - TOL))
        return ix0,ix1,ix2,ix3

    def sigmax(self,x):
        '''dimensionless, divided by e0 omega'''
        return xp.where(xp.abs(x)>self.xy.xw/2.,xp.power((xp.abs(x) - self.xy.xw/2)/(self.PML*self.xy.dx0),2.)*self.sigma_max,0.+0.j)
    
    def sigmay(self,y):
        '''dimensionless, divided by e0 omega'''
        return xp.where(xp.abs(y)>self.xy.yw/2.,xp.power((xp.abs(y) - self.xy.yw/2)/(self.PML*self.xy.dy0),2.)*self.sigma_max,0.+0.j)


class UniformMesh2D(RectMesh2D):
    """Uniform (non-adaptive) version of RectMesh2D. AMR calls are no-ops."""

    def __init__(self, xw, yw, dx, dy, Nbc=4):
        super().__init__(xw, yw, dx, dy, Nbc)
        self.max_iters = 0

    def get_base_field(self, u):
        return u

    def refine_base(self, u, ucrit):
        pass

    def refine_by_two(self, u, crit_val):
        return u


class UniformMesh3D(RectMesh3D):
    """Uniform (non-adaptive) version of RectMesh3D. Uses UniformMesh2D as xy sub-mesh."""

    def __init__(self, xw, yw, zw, ds, dz, PML=4, xwfunc=None, ywfunc=None):
        super().__init__(xw, yw, zw, ds, dz, PML, xwfunc, ywfunc)
        # Replace the RectMesh2D xy sub-mesh with a UniformMesh2D
        xy_xw = self.xy.xw
        xy_yw = self.xy.yw
        self.xy = UniformMesh2D(xy_xw, xy_yw, ds, ds, PML)

