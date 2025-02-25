import numpy as np
import mcdc
import sys

method = sys.argv[1]
if method not in ["analog"]:
    print("[ERROR] Unsupported method: %s" % method)
    exit()


# =============================================================================
# Set model
# =============================================================================
# Homogeneous, mono-energetic, infinite medium system

# Materials
m = mcdc.material(
    capture=np.array([1.0 / 3.0]),
    scatter=np.array([[1.0 / 3.0]]),
    fission=np.array([1.0 / 3.0]),
    nu_p=np.array([2.3]),
)

# Surfaces
s1 = mcdc.surface("plane-x", x=-1e10, bc="reflective")
s2 = mcdc.surface("plane-x", x=1e10, bc="reflective")

# Cells
mcdc.cell(+s1 & -s2, m)

# =============================================================================
# Set source
# =============================================================================
# Isotropic pulse at x=t=0
# (By default, t = 0)

mcdc.source(point=[0.0, 0.0, 0.0], isotropic=True)

# =============================================================================
# Set tally, setting, and run mcdc
# =============================================================================

# Tally
mcdc.tally.mesh_tally(
    scores=["flux"],
    x=np.linspace(-20.5, 20.5, 202),
    t=np.linspace(0.0, 20.0, 21),
)

# Setting
mcdc.setting(N_particle=1e5, N_batch=30)

# Run
mcdc.run()
