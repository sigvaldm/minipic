import minipic as mp
import numpy as np
import matplotlib.pylab as plt
from tasktimer import TaskTimer

"""
" INITIAL CONDITIONS
"""

Ng = 32**3
Np = Ng*64

Nt = 45
dt = 0.2
L = 2*np.pi
dx = L/Ng
x = np.arange(0,L,dx)
qe = -1.0
qi = 1.0
me = 1.0
mi = 10.0
mul = (L/Np)*(me/qe**2)
qe *= mul
me *= mul
qi *= mul
mi *= mul
solver = mp.Solver(Ng, dx, True)

pos_e = np.linspace(0, Ng, Np, endpoint=False)
pos_i = np.linspace(0, Ng, Np, endpoint=False)
# pos = np.random.uniform(0, Ng, Np)
pos_e = pos_e + 0.01*np.cos(2*np.pi*pos_e/Ng)
pos_e %= Ng

vel_e = np.zeros(pos_e.shape) # cold
vel_i = np.zeros(pos_i.shape) # cold
# vel = velTh * np.random.randn(Np) + velDrift # warm

PE = np.zeros(Nt)
KE_i = np.zeros(Nt)
KE_e = np.zeros(Nt)
KE_e[0] = 0.5*me*sum(vel_e**2)
KE_i[0] = 0.5*mi*sum(vel_i**2)

rho = (qe/dx)*mp.nb_distr(pos_e, Ng) + (qi/dx)*mp.nb_distr(pos_i, Ng)
phi = solver.solve(rho)
E = -mp.grad(phi, dx)

#
a = E*(dt**2/dx)
mp.nb_accel(pos_e, vel_e, 0.5*(qe/me)*a)
mp.nb_accel(pos_i, vel_i, 0.5*(qi/mi)*a)
# rho -= np.average(rho)
PE[0] = 0.5*dx*sum(rho*phi)

timer = TaskTimer()

"""
" TIME LOOP
"""

for n in timer.iterate(range(1,Nt)):

    timer.task('Move')
    mp.nb_move(pos_e, vel_e, Ng)
    mp.nb_move(pos_i, vel_i, Ng)

    timer.task('Distribute')
    rho = (qe/dx)*mp.nb_distr(pos_e, Ng) + (qi/dx)*mp.nb_distr(pos_i, Ng)

    timer.task('Solve phi')
    phi = solver.solve(rho)

    timer.task('Solve E')
    E = -mp.grad(phi, dx)

    timer.task('Accelerate')
    a = E*(dt**2/dx)
    KE_e[n] = (dx/dt)**2*me*mp.nb_accel(pos_e, vel_e, (qe/me)*a)
    KE_i[n] = (dx/dt)**2*mi*mp.nb_accel(pos_i, vel_i, (qi/mi)*a)
    # rho -= np.average(rho)

    timer.task('Potential energy')
    PE[n] = 0.5*dx*sum(rho*phi)

print(timer)


plt.plot(KE_e, label='Kinetic Energy (electrons)')
plt.plot(KE_i, label='Kinetic Energy (ions)')
plt.plot(PE, label='Potential Energy')
plt.plot(KE_e+KE_i+PE, label='Total Energy')
plt.legend(loc='lower right')
plt.show()

# plt.plot(rho, label='rho')
# plt.plot(phi, label='phi')
# plt.plot(E, label='E')
# plt.legend(loc='upper right')
# plt.show()
