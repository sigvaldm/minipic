import minipic as mp
import numpy as np
import matplotlib.pylab as plt

"""
" INITIAL CONDITIONS
"""

Ng = 32
Np = Ng*4

Nt = 150
dt = 0.2
L = 2*np.pi
dx = L/Ng
x = np.arange(0,L,dx)
q = -1.0
m = 1.0
mul = (L/Np)*(m/q**2)
q *= mul
m *= mul
solver = mp.Solver(Ng, dx, True)

pos = np.linspace(0, Ng, Np, endpoint=False)
# pos = np.random.uniform(0, Ng, Np)
pos = pos + 0.01*np.cos(2*np.pi*pos/Ng)
pos %= Ng

plt.hist(pos, bins=Ng)
plt.show()

vel = np.zeros(pos.shape) # cold
# vel = velTh * np.random.randn(Np) + velDrift # warm

KE = np.zeros(Nt)
PE = np.zeros(Nt)
KE[0] = 0.5*m*sum(vel**2)

rho = (q/dx)*mp.distr(pos, Ng)
phi = solver.solve(rho)
E = -mp.grad(phi, dx)

plt.plot(rho-np.average(rho), label='rho')
plt.plot(phi, label='phi')
plt.plot(E, label='E')
plt.legend(loc="lower right")
plt.show()
#
a = E*(q/m)*(dt**2/dx)
mp.accel(pos, vel, 0.5*a)
rho -= np.average(rho)
PE[0] = 0.5*dx*sum(rho*phi)

"""
" TIME LOOP
"""

for n in range(1,Nt):

    mp.move(pos, vel, Ng)
    rho = (q/dx)*mp.distr(pos, Ng)
    phi = solver.solve(rho)
    E = -mp.grad(phi, dx)
    a = E*(q/m)*(dt**2/dx)
    KE[n] = m*mp.accel(pos, vel, a)
    rho -= np.average(rho)
    PE[n] = 0.5*dx*sum(rho*phi)


plt.plot(KE, label='Kinetic Energy')
plt.plot(PE, label='Potential Energy')
plt.plot(KE+PE, label='Total Energy')
plt.legend(loc='lower right')
plt.show()

# plt.plot(rho, label='rho')
# plt.plot(phi, label='phi')
# plt.plot(E, label='E')
# plt.legend(loc='upper right')
# plt.show()
