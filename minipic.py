import numpy as np
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
    x = x.ravel()
    N = len(a[0])
    ai = np.interp(x, np.arange(N), a[0], period=N).reshape(-1,1)
    energy = 0.5*np.sum(v*(v+ai))
    v += ai
    return energy

def nb_accel(xs, vs, a):
    dim = a.ndim-1
    if dim == 3:
        return nb_accel_3D(xs, vs, a)
    elif dim == 2:
        return nb_accel_2D(xs, vs, a)
    else:
        return nb_accel_1D(xs, vs, a)

@nb.njit(fastmath=True, parallel=parallel, cache=True)
def nb_accel_3D(xs, vs, a):
    "Error O(dt^3) in pos and vel but only O(dt^2) in energy"
    N = a.shape
    b = np.zeros((3,N[1]+1,N[2]+1,N[3]+1))
    b[:,:-1,:-1,:-1] = a
    b[:,-1,:,:] = b[:,0,:,:]
    b[:,:,-1,:] = b[:,:,0,:]
    b[:,:,:,-1] = b[:,:,:,0]

    energy = 0.
    for p in nb.prange(xs.shape[0]):

        i = int(xs[p,0])
        j = int(xs[p,1])
        k = int(xs[p,2])
        x = xs[p,0]-i
        y = xs[p,1]-j
        z = xs[p,2]-k
        xc = 1-x
        yc = 1-y
        zc = 1-z

        for d in nb.prange(3):
            ai = zc*( yc*( xc*b[d,i,j,k]
                          +x *b[d,i+1,j,k])
                     +y *( xc*b[d,i,j+1,k]
                          +x *b[d,i+1,j+1,k])) \
                +z *( yc*( xc*b[d,i,j,k+1]
                          +x *b[d,i+1,j,k+1])
                     +y *( xc*b[d,i,j+1,k+1]
                          +x *b[d,i+1,j+1,k+1]))

            energy += vs[p,d]*(vs[p,d]+ai)
            vs[p,d] += ai

    energy *= 0.5
    return energy

@nb.njit(fastmath=True, parallel=parallel, cache=True)
def nb_accel_2D(xs, vs, a):
    "Error O(dt^3) in pos and vel but only O(dt^2) in energy"
    N = a.shape
    b = np.zeros((2,N[1]+1,N[2]+1))
    b[:,:-1,:-1] = a
    b[:,-1,:] = b[:,0,:]
    b[:,:,-1] = b[:,:,0]

    energy = 0.
    for p in nb.prange(xs.shape[0]):

        i = int(xs[p,0])
        j = int(xs[p,1])
        x = xs[p,0]-i
        y = xs[p,1]-j
        xc = 1-x
        yc = 1-y

        for d in nb.prange(2):
            ai = yc*( xc*b[d,i,j]
                     +x *b[d,i+1,j]) \
                +y *( xc*b[d,i,j+1]
                     +x *b[d,i+1,j+1])

            energy += vs[p,d]*(vs[p,d]+ai)
            vs[p,d] += ai

    energy *= 0.5
    return energy

@nb.njit(fastmath=True, parallel=parallel, cache=True)
def nb_accel_1D(xs, vs, a):
    "Error O(dt^3) in pos and vel but only O(dt^2) in energy"
    N = a.shape
    b = np.zeros((1,N[1]+1))
    b[:,:-1] = a
    b[:,-1] = b[:,0]

    energy = 0.
    for p in nb.prange(xs.shape[0]):

        i = int(xs[p,0])
        x = xs[p,0]-i
        xc = 1-x

        ai = xc*b[0,i] + x*b[0,i+1]

        energy += vs[p,0]*(vs[p,0]+ai)
        vs[p,0] += ai

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

@nb.njit(fastmath=True, parallel=parallel, cache=True)
def nb_move(xs, vs, L):
    # Unrolled loops perform better
    # xs += vs
    # xs %= L
    if xs.shape[1]==3:
        for p in nb.prange(xs.shape[0]):
            xs[p,0] += vs[p,0]
            xs[p,0] %= L[0]
            xs[p,1] += vs[p,1]
            xs[p,1] %= L[1]
            xs[p,2] += vs[p,2]
            xs[p,2] %= L[2]
    elif xs.shape[1]==2:
        for p in nb.prange(xs.shape[0]):
            xs[p,0] += vs[p,0]
            xs[p,0] %= L[0]
            xs[p,1] += vs[p,1]
            xs[p,1] %= L[1]
    elif xs.shape[1]==1:
        for p in nb.prange(xs.shape[0]):
            xs[p,0] += vs[p,0]
            xs[p,0] %= L[0]

def distr(x, N):
    x = x.ravel()
    j = x.astype(int)
    rho  = np.bincount(j, 1-(x-j))
    rho += np.bincount((j+1)%N, x-j)
    return rho

def nb_distr(xs, N):
    if xs.shape[1] == 3:
        return nb_distr_3D(xs, N)
    elif xs.shape[1] == 2:
        return nb_distr_2D(xs, N)
    elif xs.shape[1] == 1:
        return nb_distr_1D(xs, N)

@nb.njit(fastmath=True, parallel=parallel, cache=True)
def nb_distr_3D(xs, N):

    # Allocate an extra slab in each dimension for periodic contributions
    rho = np.zeros((N[0]+1, N[1]+1, N[2]+1))

    # Do not parallelize this loop
    for p in range(len(xs)):

        i = int(xs[p,0])
        j = int(xs[p,1])
        k = int(xs[p,2])
        x = xs[p,0]-i
        y = xs[p,1]-j
        z = xs[p,2]-k
        xc = 1-x
        yc = 1-y
        zc = 1-z

        rho[i  , j  , k  ] += xc * yc * zc
        rho[i+1, j  , k  ] += x  * yc * zc
        rho[i  , j+1, k  ] += xc * y  * zc
        rho[i+1, j+1, k  ] += x  * y  * zc
        rho[i  , j  , k+1] += xc * yc * z
        rho[i+1, j  , k+1] += x  * yc * z
        rho[i  , j+1, k+1] += xc * y  * z
        rho[i+1, j+1, k+1] += x  * y  * z

    # Wrap periodic contributions in place
    rho[0,:,:] += rho[-1,:,:]
    rho[:,0,:] += rho[:,-1,:]
    rho[:,:,0] += rho[:,:,-1]
    return rho[:-1,:-1,:-1]

@nb.njit(fastmath=True, parallel=parallel, cache=True)
def nb_distr_2D(xs, N):

    # Allocate an extra slab in each dimension for periodic contributions
    rho = np.zeros((N[0]+1, N[1]+1))

    # Do not parallelize this loop
    for p in range(len(xs)):

        i = int(xs[p,0])
        j = int(xs[p,1])
        x = xs[p,0]-i
        y = xs[p,1]-j
        xc = 1-x
        yc = 1-y

        rho[i  , j  ] += xc * yc
        rho[i+1, j  ] += x  * yc
        rho[i  , j+1] += xc * y
        rho[i+1, j+1] += x  * y

    # Wrap periodic contributions in place
    rho[0,:] += rho[-1,:]
    rho[:,0] += rho[:,-1]
    return rho[:-1,:-1]

@nb.njit(fastmath=True, parallel=parallel, cache=True)
def nb_distr_1D(xs, N):

    # Allocate an extra slab in each dimension for periodic contributions
    rho = np.zeros(N[0]+1)

    # Do not parallelize this loop
    for p in range(xs.shape[0]):

        i = int(xs[p,0])
        x = xs[p,0]-i
        xc = 1-x

        rho[i]   += xc
        rho[i+1] += x

    # Wrap periodic contributions in place
    rho[0] += rho[-1]
    return rho[:-1]
