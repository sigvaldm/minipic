from minipic import *
import numpy as np

dim = 2

L = np.array([1]*dim)
Ng = np.array([10]*dim)
dx = L/Ng
grid = make_grid(L, Ng, sparse=True)

mode = 2
k = mode*np.pi/L[dim-1]


phi = np.sin(k*grid[dim-1]) + np.zeros(Ng)
E = -grad(phi, dx)

x_e = np.linspace(0, L[dim-1], 100)
E_e = -k*np.cos(k*x_e)
plt.plot(x_e, E_e, label='E exact')

x = grid[-1].flatten()
if dim==1:
    plt.plot(x, E[0,:], 'o', label='E')
elif dim==2:
    plt.plot(x, E[1,0,:], 'o', label='E')

plt.legend()
plt.show()
