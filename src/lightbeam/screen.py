# Michael Fitzgerald (mpfitz@ucla.edu)
import numpy as _np
from lightbeam.xp import xp, to_cpu

# CuPy exposes cupy.fft with the same API as numpy.fft
fftfreq  = xp.fft.fftfreq
fft2     = xp.fft.fft2
ifft2    = xp.fft.ifft2
fftshift = xp.fft.fftshift
import matplotlib.pyplot as plt
from matplotlib import animation

def boiling_freq(k,t1):
    return xp.power(k,2/3)/t1

# from Srinath et al. (2015) 28 Dec 2015 | Vol. 23, No. 26 | DOI:10.1364/OE.23.033335 | OPTICS EXPRESS 33335
class PhaseScreenGenerator(object):
    def __init__(self, D, p, vy, vx, T, r0, wl0, wl, rs=None, seed=None, alpha_mag=1.,filter_func = None,filter_scale=None):
        # set up random number generator
        if rs is None:
            rs = _np.random.RandomState(seed=seed)
            self.seed = seed

        if filter_scale is None:
            filter_scale = D/2
        
        if filter_func is None:
            filter_func = lambda x,y: xp.ones_like(x)

        self.rs = rs

        # set up array dimensions
        a = 2.**_np.ceil(_np.log2(D/p)) / (D/p)
        self.N = int(a*D/p) # number of pixels on a side of a screen
        S = self.N*p # [m] screen size

        # frequency array
        fy = fftfreq(self.N, d=p)
        fx = fftfreq(self.N, d=p)

        fy = xp.outer(fy, xp.ones(self.N))
        fx = xp.outer(xp.ones(self.N), fx)

        ff = xp.sqrt(fx*fx+fy*fy)
        
        # turbulent power spectrum
        with _np.errstate(divide='ignore'):
            self.P =   2.*xp.pi/S * self.N * r0**(-5./6.) * (fy*fy + fx*fx)**(-11./12.) * xp.sqrt(0.00058) * (wl0/wl)
            self.P[0,0] = 0. # no infinite power

            if filter_func is not None:
                self.P *= xp.sqrt(filter_func(filter_scale*fx,filter_scale*fy))

        # set phase scale
        theta = -2.*xp.pi*T*(fy*vy+fx*vx)
        self.alpha = alpha_mag*xp.exp(1j*theta)#*xp.exp(-T*boiling_freq(ff,0.27)) # |alpha|=1 is pure frozen flow 

        # white noise scale factor
        self.beta = xp.sqrt(1.-xp.abs(self.alpha)**2)*self.P

        self.last_phip = None

        self.t = 0

    def generate(self):
        # generate white noise on CPU for reproducibility, then move to device
        w_cpu = self.rs.randn(self.N, self.N)
        w = xp.asarray(w_cpu)   # one H→D transfer (small array)
        wp = fft2(w)             # on-device FFT

        # get FT of phase
        if self.last_phip is None:
            # first timestep, generate power
            phip = self.P*wp
        else:
            phip = self.alpha*self.last_phip + self.beta*wp
        self.last_phip = phip    # stays on device between calls
    
        # get phase
        phi = ifft2(phip).real   # on-device

        return phi               # device array; caller uses to_cpu() if plotting
    
    def reset(self):
        self.last_phip = None 
        self.rs = _np.random.RandomState(self.seed)

    def get_layer(self,t):
        # need to generate from self.t up until t in steps of T
        pass

def make_ani(out,t,dt,D=10,p=0.1,wl0=1,wl=1,vx=4,vy=0,r0=0.25,alpha=1,seed=345698,delay=20):
    psgen = PhaseScreenGenerator(D, p, vy, vx, dt, r0, wl0, wl, seed=seed,alpha_mag=alpha)

    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure()
    ax = plt.axes(xlim=(0, 128), ylim=(0,128))

    im=plt.imshow(psgen.generate())

    # initialization function: plot the background of each frame
    def init():
        im.set_data(xp.zeros((128,128)))
        return [im]

    # animation function.  This is called sequentially
    def animate(i):
        im.set_data(psgen.generate())
        return [im]

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                frames=int(t/dt), interval=20, blit=True)

    anim.save(out+'.mp4', fps=60, extra_args=['-vcodec', 'libx264'])

    plt.show()

if __name__=='__main__':

    import zernike as zk
    #make_ani("turb_ani_test_alpha0pt999",5,0.01,vx=0,alpha=0.99)

    r0s = [0.169,0.1765,0.185,0.196,0.215,0.245,0.295]
    SRs = [0.1,0.2,0.3,0.4,0.5,0.6,0.7]

    rms2s=[]
    rms3s=[]

    for sr,r0 in zip(SRs,r0s):

        D = 10. # [m]  telescope diameter
        p = 10/256 # [m/pix] sampling scale

        # set wind parameters
        vy, vx = 0., 10. # [m/s] wind velocity vector
        T = 0.01 # [s]  sampling interval

        # set turbulence parameters
        #r0 = 0.185 # [m]
        wl0 = 1 #[um]
        wl = 1 #[um]

        seed = 123456
        psgen = PhaseScreenGenerator(D, p, vy, vx, T, r0, wl0, wl, seed=seed)

        xa = ya = xp.linspace(-1,1,256)
        xg , yg = xp.meshgrid(xa,ya)

        dA = (D/256)**2

        z2 = zk.Zj_cart(2)(xg,yg)
        z3 = zk.Zj_cart(3)(xg,yg)

        #plt.imshow(z2)
        #plt.show()

        print(xp.sum(z2*z2) * dA / xp.pi/25)
        s = psgen.generate()


        PV2s = []
        PV3s = []
        c2s = [] 
        c3s = []
        pe2s = []
        pe3s = []

        for i in range(1000):
            s = psgen.generate()
            #plt.imshow(s)
            #plt.show()

            c2 = xp.sum(s*z2)*dA/xp.pi/25
            c3 = xp.sum(s*z3)*dA/xp.pi/25

            c2s.append(c2)
            c3s.append(c3)

            PV2s.append(4*c2/(2*xp.pi))
            PV3s.append(4*c3/(2*xp.pi))

            pe2 = xp.std(c2*z2)/(2*xp.pi)
            pe3 = xp.std(c3*z3)/(2*xp.pi) 

            pe2s.append(pe2)
            pe3s.append(pe3)

        print(xp.std(xp.array(c2s)))
        plt.plot(xp.arange(1000)*0.01,c2s)
        plt.show()

        rms2 = xp.sqrt(xp.mean(xp.power(xp.array(PV2s),2)))
        rms3 = xp.sqrt(xp.mean(xp.power(xp.array(PV3s),2)))

        plt.plot(xp.arange(100),pe2s,color='steelblue',label='x tilt')
        plt.plot(xp.arange(100),pe3s,color='indianred',label='y tilt')
        #plt.axhline(y=rms2,color='steelblue',ls='dashed')
        #plt.axhline(y=rms3,color='indianred',ls='dashed')
        plt.xlabel("timestep")
        plt.ylabel("RMS wf error of TT modes")

        plt.legend(frameon=False)
        plt.show()

        rms2s.append(rms2)
        rms3s.append(rms3)

    plt.plot(SRs,rms2s,label="x tilt",color='steelblue')
    plt.plot(SRs,rms3s,label="y tilt",color='indianred')

    plt.xlabel("Strehl ratio")
    plt.ylabel("rms mode amplitude")
    plt.legend(frameon=False)
    plt.show()

    """
    plt.plot(xp.arange(len(coeffs2)),coeffs2,color='steelblue',label="x tilt")
    plt.plot(xp.arange(len(coeffs3)),coeffs3,color='indianred',label="y tilt")

    rms2 = xp.sqrt(xp.mean(xp.power(xp.array(coeffs2),2)))
    rms3 = xp.sqrt(xp.mean(xp.power(xp.array(coeffs3),2)))

    plt.axhline(y=rms2,color='steelblue',ls='dashed')
    plt.axhline(y=rms3,color='indianred',ls='dashed')
    plt.xlabel("timestep")
    plt.ylabel("zernike mode amplitude")
    plt.ylim(-8,8)

    plt.legend(frameon=False)

    plt.show()
    """

    """
    last = psgen.generate()
    out = xp.zeros_like(last)

    for i in range(8000):
        cur = psgen.generate()
        out += xp.power(cur-last,2)
        last = cur

    out/=8000
    plt.imshow(out)
    plt.show()
    print(xp.mean(out))
    """



    #fits.writeto('screens_test.fits', screens, overwrite=True)




    ## show results
    #import pylab
    #fig = pylab.figure(0)
    #fig.clear()
    #ax = fig.add_subplot(111)
    #for screen in screens:
    #    ax.cla()
    #    ax.imshow(screen)
    #    pylab.draw()
    #    pylab.show()
