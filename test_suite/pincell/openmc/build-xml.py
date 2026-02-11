import openmc
import numpy as np
import sys

N = int(sys.argv[1])

###############################################################################
# Create materials for the problem

# Materials
fuel = openmc.Material()
fuel.add_nuclide('U235', 0.0001654509603995036)
fuel.add_nuclide('U238', 0.022801089905717036)
fuel.add_nuclide('O16', 0.04593308173223308)
#
moderator = openmc.Material()
moderator.add_nuclide('H1', 0.05129627050184732)
moderator.add_nuclide('O16', 0.024622209840886707)
moderator.add_nuclide('B10', 4.103701640147785e-05)
#
materials = openmc.Materials([fuel, moderator])
materials.export_to_xml()

###############################################################################
# Define problem geometry

cylinder = openmc.ZCylinder(r=0.45720, name='Fuel OR')
pitch = 1.25984
box = openmc.model.RectangularPrism(pitch, pitch, boundary_type='reflective')
#
fuel_cell = openmc.Cell(fill=fuel, region=-cylinder)
moderator_cell = openmc.Cell(fill=moderator, region=+cylinder & -box)
#
geometry = openmc.Geometry([fuel_cell, moderator_cell])
geometry.export_to_xml()

###############################################################################
# Define problem settings

settings = openmc.Settings()
settings.run_mode = "fixed source"
settings.batches = 30
settings.particles = N
settings.cutoff = {"time_neutron": 1.0}
space = openmc.stats.Point()  # At the origin (0, 0, 0)
energy = openmc.stats.delta_function(14.1e6)  # At 14.1 MeV
settings.source = openmc.IndependentSource(space=space, energy=energy)
settings.export_to_xml()

###############################################################################
# Define tallies

# Create a mesh filter that can be used in a tally
time_filter = openmc.TimeFilter(np.insert(np.logspace(-8, 2, 50), 0, 0.0))
with np.load("MGXS-SHEM361.npz") as data:
    E = data["E"]
energy_filter = openmc.EnergyFilter(E)

# Now use the mesh filter in a tally and indicate what scores are desired
tally = openmc.Tally(name="TD spectrum")
tally.filters = [time_filter, energy_filter]
tally.scores = ['flux']

# Instantiate a Tallies collection and export to XML
tallies = openmc.Tallies([tally])
tallies.export_to_xml()
