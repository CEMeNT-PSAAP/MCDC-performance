from mpi4py import MPI
import numpy as np

N = int(1E10)

chunk_size = int(1E8)
N_chunk = int(N / chunk_size)

data = np.zeros(N, dtype=np.float64)
buff = np.zeros(N, dtype=np.float64)

for i in range(N_chunk):
    start = i * chunk_size
    end = start + chunk_size
    print(i, N_chunk, start, end)
    MPI.COMM_WORLD.Reduce(data[start:end], buff[start:end], MPI.SUM, 0)
