import minipic as mp
import numpy as np
import matplotlib.pylab as plt
from tasktimer import TaskTimer

"""
" INITIAL CONDITIONS
"""

Ng = np.array([32, 32])
Np = np.prod(Ng)*64

T = 3*np.pi
dt = 0.05
Nt = int(np.ceil(T/dt))
L = np.array([2*np.pi]*len(Ng))
dx = L/Ng
dv = np.prod(dx)

qe = -1.0
qi = 1.0
me = 1.0
mi = 10.0
mul = (np.prod(L)/Np)*(me/qe**2)
qe *= mul
me *= mul
qi *= mul
mi *= mul
solver = mp.Solver(Ng, dx)

pos_e = np.random.rand(Np, len(Ng))*Ng
pos_i = np.random.rand(Np, len(Ng))*Ng
pos_e[:,0] += 1e-5*np.cos(2*np.pi*pos_e[:,0]/Ng[0])
pos_e[:,0] %= Ng[0]

vel_e = np.zeros(pos_e.shape)
vel_i = np.zeros(pos_i.shape)

PE = np.zeros(Nt)
KE_i = np.zeros(Nt)
KE_e = np.zeros(Nt)
KE_e[0] = 0.5*me*np.sum(vel_e**2)
KE_i[0] = 0.5*mi*np.sum(vel_i**2)

"""
" TIME LOOP
"""

timer = TaskTimer()

for n in timer.iterate(range(1,Nt)):

    timer.task('Distribute')
    rho = (qe/dv)*mp.nb_distr(pos_e, Ng) + (qi/dv)*mp.nb_distr(pos_i, Ng)

    timer.task('Poisson-solve')
    phi = solver.solve(rho)

    timer.task('E-field gradient')
    E = -mp.grad(phi, dx)

    timer.task('Accelerate')
    a = E*(dt**2/dx[0]) * (1-0.5*(n==1))
    KE_e[n] = (dx[0]/dt)**2*me*mp.nb_accel(pos_e, vel_e, (qe/me)*a)
    KE_i[n] = (dx[0]/dt)**2*mi*mp.nb_accel(pos_i, vel_i, (qi/mi)*a)

    timer.task('Potential energy')
    PE[n] = 0.5*dv*np.sum(rho*phi)

    timer.task('Move')
    mp.nb_move(pos_e, vel_e, Ng)
    mp.nb_move(pos_i, vel_i, Ng)


print(timer)

plt.plot(KE_e, label='Kinetic Energy (electrons)')
plt.plot(KE_i, label='Kinetic Energy (ions)')
plt.plot(PE, label='Potential Energy')
plt.plot(KE_e+KE_i+PE, label='Total Energy')
plt.legend(loc='lower right')
plt.show()
