import numpy as np
import matplotlib.pylab as plt
import numba as nb
from copy import deepcopy

parallel=False

def make_grid(domain_size, num_cells, sparse=False):

    axes = []
    for l, n in zip(domain_size, num_cells):
        axes.append(np.linspace(0, l, n, endpoint=False))

    return np.meshgrid(*axes, indexing='ij', sparse=sparse)

class Solver:

    def __init__(self, Ng, dx, finite_difference=False):

        Ng = deepcopy(Ng)
        L = dx*Ng

        kdx = 2*np.pi/L
        Ng[-1] = Ng[-1]//2+1
        kdx[0:-1] /= np.sqrt(2) # Why the fuck must this be here?

        ks = make_grid(Ng*kdx, Ng)
        ks = np.sqrt(sum([a**2 for a in ks]))

        ks.ravel()[0]=1 # To avoid divide-by-zero
        self.K_sq_inv = ks**(-2)
        self.K_sq_inv.ravel()[0] = 0 # Quasi-neutrality

    def solve(self, rho):
        spectrum = np.fft.rfftn(rho)
        spectrum *= self.K_sq_inv
        phi = np.fft.irfftn(spectrum, rho.shape)
        return phi

# def grad(phi, dx):
#     E = np.zeros(phi.shape)
#     E[0]    = phi[1]  - phi[-1]
#     E[1:-1] = phi[2:] - phi[:-2]
#     E[-1]   = phi[0]  - phi[-2]
#     E /= 2*dx
#     return E

def grad(phi, dx):
    if len(phi.shape)==1:
        E = np.zeros((1,*phi.shape))
        E[0,0]    = phi[1]  - phi[-1]
        E[0,1:-1] = phi[2:] - phi[:-2]
        E[0,-1]   = phi[0]  - phi[-2]
        E /= 2*dx[0]
    elif len(phi.shape)==2:
        E = np.zeros((2,*phi.shape))
        E[0,0,:]    = phi[1,:]  - phi[-1,:]
        E[0,1:-1,:] = phi[2:,:] - phi[:-2,:]
        E[0,-1,:]   = phi[0,:]  - phi[-2,:]
        E[1,:,0]    = phi[:,1]  - phi[:,-1]
        E[1,:,1:-1] = phi[:,2:] - phi[:,:-2]
        E[1,:,-1]   = phi[:,0]  - phi[:,-2]
        E[0] /= 2*dx[0]
        E[1] /= 2*dx[1]
    else:
        E = np.zeros((3,*phi.shape))
        E[0,0,:,:]    = phi[1,:,:]  - phi[-1,:,:]
        E[0,1:-1,:,:] = phi[2:,:,:] - phi[:-2,:,:]
        E[0,-1,:,:]   = phi[0,:,:]  - phi[-2,:,:]
        E[1,:,0,:]    = phi[:,1,:]  - phi[:,-1,:]
        E[1,:,1:-1,:] = phi[:,2:,:] - phi[:,:-2,:]
        E[1,:,-1,:]   = phi[:,0,:]  - phi[:,-2,:]
        E[2,:,:,0]    = phi[:,:,1]  - phi[:,:,-1]
        E[2,:,:,1:-1] = phi[:,:,2:] - phi[:,:,:-2]
        E[2,:,:,-1]   = phi[:,:,0]  - phi[:,:,-2]
        E[0] /= 2*dx[0]
        E[1] /= 2*dx[1]
        E[2] /= 2*dx[2]
    return E

def accel(x, v, a):
    "Error O(dt^3) in pos and vel but only O(dt^2) in energy"
    N = len(a)
    ai = np.interp(x, np.arange(N), a, period=N)
    energy = 0.5*np.sum(v*(v+ai))
    v += ai
    return energy

@nb.njit(fastmath=True, parallel=parallel)
def nb_accel(xs, vs, a):
    "Error O(dt^3) in pos and vel but only O(dt^2) in energy"
    N = len(a)
    b = np.zeros(N+1)
    b[:-1] = a
    b[-1] = a[0]
    energy = 0.
    for i in range(len(xs)):
        x = xs[i]
        j = int(x)
        ai = (x-j)*b[j+1] + (1-x+j)*b[j]
        energy += vs[i]*(vs[i]+ai)
        vs[i] += ai
    energy *= 0.5
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

@nb.njit(fastmath=True, parallel=parallel)
def nb_move(xs, vs, L):
    for i in nb.prange(len(xs)):
        xs[i] += vs[i]
        xs[i] %= L
    # xs += vs
    # xs %= L

def distr(x, N):
    j = x.astype(int)
    rho  = np.bincount(j, 1-(x-j))
    rho += np.bincount((j+1)%N, x-j)
    return rho

@nb.njit(fastmath=True, parallel=parallel)
def nb_distr(xs, N):
    rho = np.zeros(N+1)
    for i in nb.prange(len(xs)):
        x = xs[i]
    # for x in xs:
        j = int(x)
        rho[j] += 1-(x-j)
        rho[j+1] += x-j
    rho[0] += rho[-1]
    return rho[:-1]
