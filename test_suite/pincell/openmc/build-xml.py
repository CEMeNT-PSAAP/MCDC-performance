from math import log10

import numpy as np
import openmc, sys

N = int(sys.argv[1])

###############################################################################
# Create materials for the problem

fuel = openmc.Material()
fuel.set_density('sum', 1.0)
fuel.add_nuclide("U235", 0.0007217486041189947)
fuel.add_nuclide("U238", 0.02224950230720295)
fuel.add_nuclide("O16", 0.04585265389377734)
fuel.add_nuclide("O17", 1.7419604031574338e-05)
fuel.add_nuclide("O18", 9.19424166352541e-05)

gas = openmc.Material()
gas.set_density('sum', 1.0)
gas.add_nuclide("He3", 4.808864272483583e-10)
gas.add_nuclide("He4", 0.00024044273273775193)

clad = openmc.Material()
clad.set_density('sum', 1.0)
clad.add_nuclide("Zr90", 0.021826659699624183)
clad.add_nuclide("Zr91", 0.004759866313504049)
clad.add_nuclide("Zr92", 0.007275553233208061)
clad.add_nuclide("Zr94", 0.007373126250329802)
clad.add_nuclide("Zr96", 0.0011878454258298875)
clad.add_nuclide("Nb93", 0.00042910080334290177)
clad.add_nuclide("O16", 5.7790773120342736e-05)
clad.add_nuclide("O17", 2.195494260303957e-08)
clad.add_nuclide("O18", 1.15880388345964e-07)

water = openmc.Material(name='Water')
water.set_density('sum', 1.0)
water.add_nuclide("B10", 1.0323440206972448e-05)
water.add_nuclide("B11", 4.1762534601163005e-05)
water.add_nuclide("H1", 0.050347844752850625)
water.add_nuclide("H2", 7.842394716362082e-06)
water.add_nuclide("O16", 0.025117935412784034)
water.add_nuclide("O17", 9.542402714463945e-06)
water.add_nuclide("O18", 5.03657582849965e-05)

# Collect the materials together and export to XML
materials = openmc.Materials([fuel, gas, clad, water])
materials.export_to_xml()

###############################################################################
# Define problem geometry

# Create cylindrical surfaces
cy1 = openmc.ZCylinder(r=0.405765)
cy2 = openmc.ZCylinder(r=0.41402)
cy3 = openmc.ZCylinder(r=0.47498)

# Create a region represented as the inside of a rectangular prism
pitch = 1.92
box = openmc.model.RectangularPrism(pitch, pitch, boundary_type='reflective')

# Create cells, mapping materials to regions
fuel = openmc.Cell(fill=fuel, region=-cy1)
gas = openmc.Cell(fill=gas, region=-cy2 & +cy1)
clad = openmc.Cell(fill=clad, region=-cy3 & +cy2)
water = openmc.Cell(fill=water, region=+cy3 & -box)

# Create a geometry and export to XML
geometry = openmc.Geometry([fuel, gas, clad, water])
geometry.export_to_xml()

###############################################################################
# Define problem settings

# Indicate how many particles to run
settings = openmc.Settings()
settings.run_mode = 'fixed source'
settings.particles = N
settings.batches = 30

# Create an initial uniform spatial source distribution over fissionable zones
delta_dist = openmc.stats.Point()
energy = openmc.stats.Uniform(1E6-1, 1E6+1)
settings.source = openmc.IndependentSource(space=delta_dist,energy=energy)
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
