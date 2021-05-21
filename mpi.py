from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

x = np.linspace(0, 1, 5, dtype=float) + rank
res = np.zeros(x.shape, dtype=float)

print('{} x   ={}'.format(rank, x))

comm.Reduce(x, res, op=MPI.SUM, root=0)
print('{} res ={}'.format(rank, res))

comm.Bcast(res, root=0)
print('{} res2={}'.format(rank, res))
