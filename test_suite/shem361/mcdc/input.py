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
# The infinite homogenous medium is modeled with reflecting slab

# Materials
with np.load("MGXS-SHEM361.npz") as data:
    SigmaC = data["SigmaC"] * 1.5  # /cm
    SigmaS = data["SigmaS"]
    SigmaF = data["SigmaF"]
    nu_p = data["nu_p"]
    nu_d = data["nu_d"]
    chi_p = data["chi_p"]
    chi_d = data["chi_d"]
    G = data["G"]
    speed = data["v"]
    lamd = data["lamd"]
#
m = mcdc.material(
    capture=SigmaC,
    scatter=SigmaS,
    fission=SigmaF,
    nu_p=nu_p,
    chi_p=chi_p,
    nu_d=nu_d,
    chi_d=chi_d,
    decay=lamd,
    speed=speed,
)

# Surfaces
s1 = mcdc.surface("plane-x", x=-1e10, bc="reflective")
s2 = mcdc.surface("plane-x", x=1e10, bc="reflective")

# Cells
c = mcdc.cell(+s1 & -s2, m)

# =============================================================================
# Set source
# =============================================================================
# At highest group

energy = np.zeros(G)
energy[-1] = 1.0
source = mcdc.source(energy=energy)

# =============================================================================
# Set tally, setting, and run mcdc
# =============================================================================

# Tally
time_grid = np.insert(np.logspace(-8, 1, 100), 0, 0.0)
mcdc.tally.mesh_tally(
    scores=["flux"], t=time_grid, g="all"
)
mcdc.tally.mesh_tally(
    scores=["density"],
    t=time_grid,
)

# Setting
mcdc.setting(N_particle=1e3, N_batch=30, active_bank_buff=10000)

# Run
mcdc.run()
