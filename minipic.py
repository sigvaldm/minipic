import numpy as np
import matplotlib.pylab as plt

class Solver:

    def __init__(self, Ng, dx, finite_difference=False):

        self.Ng = Ng
        L = dx*Ng

        kdx = 2*np.pi/L
        # Ngd = (Ng//2+1)*kdx
        # Ngd = Ng*kdx
        Ngd = Ng*1.1/1.1
        Ngd[-1] = Ngd[-1]//2+1
        kdx[0:-1] /= np.sqrt(2) # Why the fuck must this be here?
        Ngd *= kdx
        # print(kdx, Ngd)

        if len(Ng)==1:
            ks = np.mgrid[0:Ngd[0]:kdx[0]]
        elif len(Ng)==2:
            ks = np.mgrid[0:Ngd[0]:kdx[0], 0:Ngd[1]:kdx[1]]

            kx, ky = ks
            print(kx[:,0])
            print(ky[0,:])

            ks = np.sqrt(ks[0]**2+ks[1]**2)
            print(ks)
        else:
            ks = np.mgrid[0:Ngd[0]:kdx[0], 0:Ngd[1]:kdx[1], 0:Ngd[2]:kdx[2]]
            ks = np.sqrt(ks[0]**2+ks[1]**2+ks[2]**2)


        # ks = ks[0]


        # ks = np.sqrt(ks[0]**2)
        # ks = np.sqrt(ks[0]**2)
        # ks[0] *= 2*np.pi/L[0]
        # ks[1] *= 2*np.pi/L[1]

        # ks = 0
        # for n,l in zip(Ng, L):
        #     ks += (2*np.pi/l)*np.arange(n//2+1)
        #     print(ks)

        # if len(Ng)==1:
        #     ks[0] = 
        # ks[0,0] = 1 # to avoid divide-by-zero

        # print(ks**(-2))

        self.K_sq_inv = ks**(-2)
        print(self.K_sq_inv**(-0.5))

        # if not finite_difference:
        #     self.K_sq_inv = ks**(-2)
        # else:
        #     arg = 0.5*dx*ks
        #     self.K_sq_inv = (ks*np.sin(arg)/arg)**(-2)

        if len(Ng)==1:
            self.K_sq_inv[0] = 0
        elif len(Ng)==2:
            self.K_sq_inv[0,0] = 0 # Quasi-neutrality
        else:
            self.K_sq_inv[0,0,0] = 0
        # self.K_sq_inv[:] = 1

    def solve(self, rho):
        # plt.plot(rho[0,:]); plt.show()
        spectrum = np.fft.rfftn(rho, rho.shape, norm="ortho")
        # plt.plot(np.abs(spectrum[0,:])); plt.show()
        # spectrum = np.fft.fftn(rho)
        print(self.K_sq_inv)
        spectrum *= self.K_sq_inv
        phi = np.fft.irfftn(spectrum, rho.shape, norm="ortho")
        # phi = np.fft.ifftn(spectrum, rho.shape)
        # print(rho.shape, spectrum.shape, self.K_sq_inv.shape, phi.shape)
        return phi

def grad(phi, dx):
    E = np.zeros(phi.shape)
    E[0]    = phi[1]  - phi[-1]
    E[1:-1] = phi[2:] - phi[:-2]
    E[-1]   = phi[0]  - phi[-2]
    E /= 2*dx
    return E

def accel(x, v, a):
    "Error O(dt^3) in pos and vel but only O(dt^2) in energy"
    N = len(a)
    j = x.astype(int)
    ai = a[j] + (x-j)*(a[(j+1)%N] - a[j])
    energy = 0.5*sum(v*(v+ai))
    v += ai
    return energy


def accel_accurate_energy(x, vb, vc, a):
    """
    Error O(dt^3) in pos, vel and energy
    va - v at (n+0.5)
    vb - v at (n-0.5)
    vc - v at (n-1.5)
    vn - v at n
    """

    N = len(a)
    j = x.astype(int)
    ai = a[j] + (x-j)*(a[(j+1)%N] - a[j])

    va = vb + ai
    vn = (3/8)*va + (3/4)*vb - (1/8)*vc
    energy = 0.5*sum(vn**2)

    return energy, va, vb

# Potential energy from particles
# Faster field computations:
#   0.5*dx*sum(rho*phi)
#   0.5*dx*sum(E**2)
def potential_energy(x, phi, q):

    N = len(phi)
    j = x.astype(int)
    phi_i = phi[j] + (x-j)*(phi[(j+1)%N] - phi[j])
    energy = 0.5*q*sum(phi_i)
    return energy

def move(x, v, L):
    x += v
    x %= L

def distr(x, N):
    j = x.astype(int)
    rho = np.zeros(N)
    np.add.at(rho, j      , 1-(x-j))
    np.add.at(rho, (j+1)%N, x-j)
    return rho
