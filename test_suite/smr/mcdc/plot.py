import matplotlib.pyplot as plt
import h5py

with h5py.File('output.h5')as f:
    fission_total = f['tallies/mesh_tally_0/density/mean'][:]
    t = f['tallies/mesh_tally_0/grid/t'][:]
    t_mid = 0.5 * (t[:-1] + t[1:])

plt.plot(t_mid, fission_total)
plt.show()
