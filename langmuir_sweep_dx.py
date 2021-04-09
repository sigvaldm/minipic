import minipic as mp
import numpy as np
import matplotlib.pylab as plt
import copy

"""
" INITIAL CONDITIONS
"""

Ngs = [2**n for n in range(2,16)]

errors = []
dxs = []
for Ng in Ngs:
    print("Ng={}".format(Ng))
    Np = Ng*2

    Nt = 500
    dt = 0.001
    L = 2*np.pi
    dx = L/Ng
    dxs.append(dx)
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

    vel = np.zeros(pos.shape) # cold
    # vel = velTh * np.random.randn(Np) + velDrift # warm
    velold = copy.deepcopy(vel)

    KE = np.zeros(Nt)
    PE = np.zeros(Nt)
    KE[0] = 0.5*m*sum(vel**2)

    rho = (q/dx)*mp.distr(pos, Ng)
    phi = solver.solve(rho)
    E = -mp.grad(phi, dx)

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
        # KE[n] = (dx/dt)**2*m*mp.accel(pos, vel, a)
        KEn, vel, velold = mp.accel_accurate_energy(pos, vel, velold, a)
        KE[n] = (dx/dt)**2*m*KEn
        rho -= np.average(rho)
        # PE[n] = 0.5*dx*sum(rho*phi)
        # PE[n] = 0.5*dx*sum(E**2)
        PE[n] = mp.potential_energy(pos, phi, q)


    TE = KE + PE
    TE[1] = TE[0] # Invalid datapoint

    # plt.plot(KE, label='Kinetic Energy')
    # plt.plot(PE, label='Potential Energy')
    # plt.plot(TE, label='Total Energy')
    # plt.legend(loc='lower right')
    # plt.show()

    offset = int(len(TE)/2)
    offset = 0
    error = np.max(np.abs(TE[offset:]-TE[offset]))/TE[offset]
    errors.append(error)

    # plt.plot(rho, label='rho')
    # plt.plot(phi, label='phi')
    # plt.plot(E, label='E')
    # plt.legend(loc='upper right')
    # plt.show()

dxs = np.array(dxs)
plt.loglog(dxs, errors)
plt.loglog(dxs, errors[0]*(dxs/dxs[0])**1, '--')
plt.loglog(dxs, errors[0]*(dxs/dxs[0])**2, '--')
plt.loglog(dxs, errors[0]*(dxs/dxs[0])**3, '--')
plt.grid()
plt.show()
