import openmc
import numpy as np
import h5py, sys

N = int(sys.argv[1])

# ===========================================================================
# Set Library
# ===========================================================================

SigmaC = 1.0 / 3.0
SigmaF = 1.0 / 3.0
nu = 2.3
SigmaA = SigmaC + SigmaF
SigmaS = 1.0 / 3.0
SigmaT = SigmaA + SigmaS
v = 1.0

groups = openmc.mgxs.EnergyGroups([0.0, 2e7])

xsdata = openmc.XSdata("mat", groups)
xsdata.order = 0

xsdata.set_inverse_velocity([1.0 / v], temperature=294.0)

xsdata.set_total([SigmaT], temperature=294.0)
xsdata.set_absorption([SigmaA], temperature=294.0)
xsdata.set_scatter_matrix(np.ones((1, 1, 1)) * SigmaS, temperature=294.0)

xsdata.set_nu_fission([nu * SigmaF], temperature=294.0)
mg_cross_sections_file = openmc.MGXSLibrary(groups)
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
surf_Z1 = openmc.XPlane(surface_id=1, x0=-1e10, boundary_type="reflective")
surf_Z2 = openmc.XPlane(surface_id=2, x0=1e10, boundary_type="reflective")

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
settings_file.cutoff = {"time_neutron": 20}
settings_file.energy_mode = "multi-group"

# Create an initial uniform spatial source distribution over fissionable zones
delta_dist = openmc.stats.Point()
isotropic = openmc.stats.Isotropic()
settings_file.source = openmc.IndependentSource(space=delta_dist, angle=isotropic)
settings_file.export_to_xml()


# ===========================================================================
# Exporting to OpenMC tallies.xml file
# ===========================================================================

# Create a mesh filter that can be used in a tally
mesh = openmc.RectilinearMesh()
mesh.x_grid = np.linspace(-20.5, 20.5, 202)
mesh.y_grid = np.linspace(-1E10, 1E10, 2)
mesh.z_grid = np.linspace(-1E10, 1E10, 2)
time = np.linspace(0.0, 20.0, 21)
time_mesh_filter = openmc.TimedMeshFilter(mesh, time)
time_filter = openmc.TimeFilter(time)

# Now use the mesh filter in a tally and indicate what scores are desired
tally1 = openmc.Tally(name="flux")
tally1.estimator = "tracklength"
tally1.filters = [time_mesh_filter]
tally1.scores = ["flux"]

# Instantiate a Tallies collection and export to XML
tallies = openmc.Tallies([tally1])
tallies.export_to_xml()
