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
# The infinite 2D SMR pincell

# Materials
fuel = mcdc.material(
    [
        ["O16", 0.04585265389377734],
        ["O17", 1.7419604031574338e-05],
        ["O18", 9.19424166352541e-05],
        ["U235", 0.0007217486041189947],
        ["U238", 0.02224950230720295],
    ]
)
gas = mcdc.material(
    [
        ["He3", 4.808864272483583e-10],
        ["He4", 0.00024044273273775193],
    ]
)
cladding = mcdc.material(
    [
        ["Zr90", 0.021826659699624183],
        ["Zr91", 0.004759866313504049],
        ["Zr92", 0.007275553233208061],
        ["Zr94", 0.007373126250329802],
        ["Zr96", 0.0011878454258298875],
        ["Nb93", 0.00042910080334290177],
        ["O16", 5.7790773120342736e-05],
        ["O17", 2.195494260303957e-08],
        ["O18", 1.15880388345964e-07],
    ]
)
water = mcdc.material(
    nuclides=[
        ["B10", 1.0323440206972448e-05],
        ["B11", 4.1762534601163005e-05],
        ["H1", 0.050347844752850625],
        ["H2", 7.842394716362082e-06],
        ["O16", 0.025117935412784034],
        ["O17", 9.542402714463945e-06],
        ["O18", 5.03657582849965e-05],
    ]
)

# Surfaces
cy1 = mcdc.surface("cylinder-z", center=[0.0, 0.0], radius=0.405765)
cy2 = mcdc.surface("cylinder-z", center=[0.0, 0.0], radius=0.41402)
cy3 = mcdc.surface("cylinder-z", center=[0.0, 0.0], radius=0.47498)
#
pitch = 1.92
x1 = mcdc.surface("plane-x", x=-pitch / 2, bc="reflective")
x2 = mcdc.surface("plane-x", x=pitch / 2, bc="reflective")
y1 = mcdc.surface("plane-y", y=-pitch / 2, bc="reflective")
y2 = mcdc.surface("plane-y", y=pitch / 2, bc="reflective")

# Cells
mcdc.cell(-cy1, fuel)
mcdc.cell(+cy1 & -cy2, gas)
mcdc.cell(+cy2 & -cy3, cladding)
mcdc.cell(+cy3 & +x1 & -x2 & +y1 & -y2, water)

# =============================================================================
# Set source
# =============================================================================
# 14.1 MeV, isotropic, at center

mcdc.source(
    point=[0.0, 0.0, 0.0],
    energy=np.array([[1e6 - 1, 1e6 + 1], [1.0, 1.0]]),
    isotropic=True,
)

# =============================================================================
# Set tally, setting, and run mcdc
# =============================================================================

# Tally
time_grid = np.insert(np.logspace(-8, 2, 100), 0, 0.0)
mcdc.tally.mesh_tally(
    scores=["flux"],
    E=np.load("MGXS-SHEM361.npz")['E'],
    t=time_grid,
)
mcdc.tally.mesh_tally(
    scores=["density"],
    t=time_grid,
)

# Setting
mcdc.setting(N_particle=1e3, active_bank_buff=10000)

# Run
mcdc.run()
