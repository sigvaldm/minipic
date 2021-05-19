import minipic as mp
import numpy as np
import matplotlib.pylab as plt
from tasktimer import TaskTimer

"""
" INITIAL CONDITIONS
"""

Ng = np.array([1024])
# Np = Ng*4
Np = int(1e4)

Nt = 150
dt = 0.2
L = np.array([2*np.pi])
dx = L/Ng
x = np.arange(0,L[0],dx[0])
q = -1.0
m = 1.0
mul = (L[0]/Np)*(m/q**2)
q *= mul
m *= mul
solver = mp.Solver(Ng, dx)

pos = np.linspace(0, Ng[0], Np, endpoint=False)
# pos = np.random.uniform(0, Ng, Np)
pos = pos + 0.01*np.cos(2*np.pi*pos/Ng[0])
pos %= Ng[0]

vel = np.zeros(pos.shape) # cold
# vel = velTh * np.random.randn(Np) + velDrift # warm

KE = np.zeros(Nt)
PE = np.zeros(Nt)
KE[0] = 0.5*m*sum(vel**2)

rho = (q/dx)*mp.distr(pos, Ng[0])
phi = solver.solve(rho)
E = -mp.grad(phi, dx)[0]

#
a = E*(q/m)*(dt**2/dx)
mp.accel(pos, vel, 0.5*a)
rho -= np.average(rho)
PE[0] = 0.5*dx*sum(rho*phi)

"""
" TIME LOOP
"""

timer = TaskTimer()

for n in timer.iterate(range(1,Nt)):

    timer.task('Move')
    mp.move(pos, vel, Ng[0])

    timer.task('Distribute')
    rho = (q/dx)*mp.distr(pos, Ng[0])

    timer.task('Poisson-solve')
    phi = solver.solve(rho)

    timer.task('E-field gradient')
    E = -mp.grad(phi, dx)[0]

    timer.task('Accelerate')
    a = E*(q/m)*(dt**2/dx)
    KE[n] = (dx/dt)**2*m*mp.accel(pos, vel, a)
    rho -= np.average(rho)

    timer.task('Potential Energy')
    PE[n] = 0.5*dx*sum(rho*phi)

print(timer)


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
