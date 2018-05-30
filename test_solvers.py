import minipic as mp
import numpy as np
import matplotlib.pylab as plt
import copy
from numpy import linalg as la


N = [2**n for n in range(1,11)]

L = 2*np.pi

errors = []
for i in N:

    dx = L/i
    x = np.arange(0,L,dx)

    rho   = np.cos(x) # Test source
    phi_a = np.cos(x) # phi - analytical solution
    E_a   = np.sin(x) # E-field - analytical solution

    psolver = mp.PoissonSolver(i) # New solver
    solver  = mp.Solver(i, dx, True) # Original solver


    phi1 = psolver.solve(rho)
    E1   = psolver.grad(phi1)

    phi2 = solver.solve(rho)
    E2 = -mp.grad(phi2, dx)

    error = la.norm(E1-E2, np.inf)
    errors.append(error)

    # plt.plot(E_a, label='Exact')
    # plt.plot(E1, label='Spectral')
    # plt.plot(E2, label='FE')
    # plt.legend(loc='upper right')
    # plt.show()

N = np.array(N)
plt.loglog(N, errors)
plt.grid()
plt.show()
