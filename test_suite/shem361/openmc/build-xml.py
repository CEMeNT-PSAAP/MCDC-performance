import openmc
import numpy as np
import h5py
import sys

N = int(sys.argv[1])

# ===============================================================================
# Data
# ===============================================================================

# Load material data
with np.load("MGXS-SHEM361.npz") as data:
    SigmaT = data["SigmaT"]
    SigmaC = data["SigmaC"]
    SigmaS = data["SigmaS"]
    nuSigmaF_p = data["nuSigmaF_p"]
    SigmaF = data["SigmaF"]
    nu_p = data["nu_p"]
    nu_d = data["nu_d"]
    chi_p = data["chi_p"]
    chi_d = data["chi_d"]
    G = data["G"][()]
    J = data["J"][()]
    E = data["E"]
    v = data["v"]
    lamd = data["lamd"]
SigmaT += SigmaC * 0.5
SigmaC *= 1.5
SigmaA = SigmaC + SigmaF

# Make a prompt spectrum independent of inducing neutron energy
norm = np.sum(chi_p)
chi_p = np.sum(chi_p, axis=1) / norm

# Flip
SigmaT = np.flip(SigmaT)
SigmaC = np.flip(SigmaC)
SigmaA = np.flip(SigmaA)
SigmaS = np.flip(SigmaS)
SigmaF = np.flip(SigmaF)
nu_p = np.flip(nu_p)
chi_p = np.flip(chi_p)
v = np.flip(v)
chi_d = np.flip(chi_d, 0)
nu_d = np.flip(nu_d, 1)

# Transpose
chi_d = np.transpose(chi_d)
SigmaS = np.transpose(SigmaS)

# ===========================================================================
# Set Library
# ===========================================================================

groups = openmc.mgxs.EnergyGroups(E)

xsdata = openmc.XSdata("mat", groups, num_delayed_groups=J)
xsdata.order = 0

xsdata.set_inverse_velocity(1.0 / v, temperature=294.0)

xsdata.set_total(SigmaT, temperature=294.0)
xsdata.set_absorption(SigmaA, temperature=294.0)
xsdata.set_scatter_matrix(np.expand_dims(SigmaS, 2), temperature=294.0)
xsdata.set_decay_rate(lamd, temperature=294.0)

xsdata.set_prompt_nu_fission(nu_p * SigmaF, temperature=294.0)
xsdata.set_delayed_nu_fission(nu_d * SigmaF, temperature=294.0)
xsdata.set_chi_prompt(chi_p, temperature=294.0)
xsdata.set_chi_delayed(chi_d, temperature=294.0)
mg_cross_sections_file = openmc.MGXSLibrary(groups, J)
mg_cross_sections_file.add_xsdata(xsdata)
mg_cross_sections_file.export_to_hdf5("mgxs.h5")

# ===========================================================================
# Exporting to OpenMC materials.xml file
# ===========================================================================

materials = {}
materials["mat"] = openmc.Material(name="mat")
materials["mat"].set_density("macro", 1.0)
materials["mat"].add_macroscopic("mat")
materials_file = openmc.Materials(materials.values())
materials_file.cross_sections = "mgxs.h5"
materials_file.export_to_xml()

# ===========================================================================
# Exporting to OpenMC geometry.xml file
# ===========================================================================

# Instantiate ZCylinder surfaces
surf_Z1 = openmc.ZPlane(surface_id=1, z0=-1e10, boundary_type="reflective")
surf_Z2 = openmc.ZPlane(surface_id=2, z0=1e10, boundary_type="reflective")

# Instantiate Cells
cell_F = openmc.Cell(cell_id=1, name="F")

# Use surface half-spaces to define regions
cell_F.region = +surf_Z1 & -surf_Z2

# Register Materials with Cells
cell_F.fill = materials["mat"]

# Instantiate Universes
root = openmc.Universe(universe_id=0, name="root universe", cells=[cell_F])

# Instantiate a Geometry, register the root Universe, and export to XML
geometry = openmc.Geometry(root)
geometry.export_to_xml()

# ===========================================================================
# Exporting to OpenMC settings.xml file
# ===========================================================================

# Instantiate a Settings object, set all runtime parameters, and export to XML
settings_file = openmc.Settings()
settings_file.run_mode = "fixed source"
settings_file.particles = N
settings_file.batches = 30
settings_file.output = {"tallies": False}
settings_file.cutoff = {"time_neutron": 10}
settings_file.energy_mode = "multi-group"

# Create an initial uniform spatial source distribution over fissionable zones
lower_left = (-1, -1, -1)
upper_right = (1, 1, 1)
uniform_dist = openmc.stats.Box(lower_left, upper_right)
energy = openmc.stats.Uniform(E[-2], E[-1])
settings_file.source = openmc.IndependentSource(space=uniform_dist, energy=energy)
settings_file.export_to_xml()


# ===========================================================================
# Exporting to OpenMC tallies.xml file
# ===========================================================================

# Create a mesh filter that can be used in a tally
time_filter = openmc.TimeFilter(np.insert(np.logspace(-8, 1, 100), 0, 0.0))
energy_filter = openmc.EnergyFilter(E)

# Now use the mesh filter in a tally and indicate what scores are desired
tally1 = openmc.Tally(name="TD spectrum")
tally1.filters = [time_filter, energy_filter]
tally1.scores = ["flux"]

# Instantiate a Tallies collection and export to XML
tallies = openmc.Tallies([tally1])
tallies.export_to_xml()
