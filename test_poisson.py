from minipic import *
import numpy as np

dim = 3

if dim==1:
    L = np.array([1])
    Ng = np.array([100])
    dx = L/Ng
    x = np.mgrid[0:L[0]:dx[0]]
elif dim==2:
    L = np.array([1, 1])
    Ng = np.array([100, 100])
    dx = L/Ng
    x,y = np.mgrid[0:L[0]:dx[0], 0:L[1]:dx[1]]
else:
    L = np.array([1, 1, 1])
    Ng = np.array([100, 100, 100])
    dx = L/Ng
    x,y,z = np.mgrid[0:L[0]:dx[0], 0:L[1]:dx[1], 0:L[2]:dx[2]]

mode = 2
k = mode*np.pi/L[0]

# rho = k**2*np.sin(k*x)
# phi_e = np.sin(k*x)

rho = (k**2)*np.sin(k*y)
phi_e = np.sin(k*y)

solver = Solver(Ng, dx)
phi = solver.solve(rho)

# plt.plot(x[:,0], rho[:,0], label='rho')

if dim==1:
    plt.plot(x[:], phi_e[:], label='phi exact')
    plt.plot(x[:], phi[:], '--', label='phi')
elif dim==2:
    plt.plot(x[:,0], phi_e[0,:], label='phi exact')
    plt.plot(x[:,0], phi[0,:], '--', label='phi')
else:
    plt.plot(x[:,0,0], phi_e[0,:,0], label='phi exact')
    plt.plot(x[:,0,0], phi[0,:,0], '--', label='phi')

plt.legend()
plt.show()
