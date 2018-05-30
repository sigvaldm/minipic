import minipic as mp
import numpy as np
import matplotlib.pylab as plt
import copy

"""
" INITIAL CONDITIONS
"""

spectral_grad = True

dts = [2**(-n) for n in range(15)]

errors = []
for dt in dts:
    Ng = 1024
    Np = Ng*8

    Nt = 150
    # dt = 0.2
    L = 2*np.pi
    dx = L/Ng
    x = np.arange(0,L,dx)
    q = -1.0
    m = 1.0
    mul = (L/Np)*(m/q**2)
    q *= mul
    m *= mul

    if spectral_grad:
        solver = mp.PoissonSolver(Ng)
    else:
        solver = mp.Solver(Ng, dx, False)

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
    if spectral_grad:
        E = solver.grad(phi)
    else:
        E = -mp.grad(phi, dx)

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
        if spectral_grad:
            E = solver.grad(phi)
        else:
            E = -mp.grad(phi, dx)

        a = E*(q/m)*(dt**2/dx)
        # KE[n] = (dx/dt)**2*m*mp.accel(pos, vel, a)
        KEn, vel, velold = mp.accel_accurate_energy(pos, vel, velold, a)
        KE[n] = (dx/dt)**2*m*KEn
        rho -= np.average(rho)
        PE[n] = 0.5*dx*sum(rho*phi)


    TE = KE + PE
    TE[1] = TE[0] # Invalid datapoint

    # plt.plot(KE, label='Kinetic Energy')
    # plt.plot(PE, label='Potential Energy')
    # plt.plot(TE, label='Total Energy')
    # plt.legend(loc='lower right')
    # plt.show()

    error = np.max(np.abs(TE-TE[0]))/TE[0]
    errors.append(error)

    # plt.plot(rho, label='rho')
    # plt.plot(phi, label='phi')
    # plt.plot(E, label='E')
    # plt.legend(loc='upper right')
    # plt.show()

dts = np.array(dts)
plt.loglog(dts, errors)
plt.loglog(dts, errors[0]*(dts/dts[0])**1, '--')
plt.loglog(dts, errors[0]*(dts/dts[0])**2, '--')
plt.loglog(dts, errors[0]*(dts/dts[0])**3, '--')
plt.grid()
plt.show()
