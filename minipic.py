from __future__ import print_function, division
import numpy as np
import matplotlib.pylab as plt

class Solver(object):

    def __init__(self, Ng, dx, finite_difference=False):
        self.Ng = Ng
        L = dx*Ng
        k = np.array([n*2*np.pi/L for n in range(Ng//2+1)])

        k[0] = 1 # to avoid divide-by-zero

        if not finite_difference:
            self.K_sq_inv = k**(-2)
        else:
            arg = 0.5*dx*k
            self.K_sq_inv = (k*np.sin(arg)/arg)**(-2)

        self.K_sq_inv[0] = 0 # Quasi-neutrality

    def solve(self, rho):
        spectrum = np.fft.rfft(rho)
        spectrum *= self.K_sq_inv
        phi = np.fft.irfft(spectrum, self.Ng)
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
    rho  = np.bincount(j, 1-(x-j))
    rho += np.bincount((j+1)%N, x-j)
    return rho
