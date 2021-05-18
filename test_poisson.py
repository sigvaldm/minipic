from minipic import *
import numpy as np

dim = 2

L = np.array([1]*dim)
Ng = np.array([5]*dim)
dx = L/Ng
grid = make_grid(L, Ng, True)

mode = 2
k = mode*np.pi/L[0]

x = grid[0]

rho = np.zeros(Ng) # To get right shape
rho += (k**2)*np.sin(k*grid[dim-1])
phi_e = np.zeros(Ng)
phi_e += np.sin(k*grid[dim-1])

solver = Solver(Ng, dx)
print(Ng)
phi = solver.solve(rho)

if dim==1:
    plt.plot(x, phi_e[:], label='phi exact')
    plt.plot(x, phi[:], '--', label='phi')
elif dim==2:
    plt.plot(x, phi_e[0,:], label='phi exact')
    plt.plot(x, phi[0,:], '--', label='phi')
else:
    plt.plot(x, phi_e[0,:,0], label='phi exact')
    plt.plot(x, phi[0,:,0], '--', label='phi')

plt.legend()
plt.show()
