# -*- coding: utf-8 -*-
"""
FTE34806 - Modelling of Biobased Production Systems
MSc Biosystems Engineering, WUR
@authors:   -- add your team names here --

Evaluation of the grass growth model
"""

import numpy as np
import matplotlib.pyplot as plt

from mbps.models.grass import Grass

# -- Define the required variables
# Simulation time
tsim = np.linspace(0.0, 365.0, 365 + 1)  # [d]
dt = 1  # [d]
# Initial conditions
# TODO: Define sensible values for the initial conditions
x0 = {'Ws': 10e-8, 'Wg': 10e-8}  # [kgC m-2]
# Model parameters (as provided by Mohtar et al. 1997 p.1492-1493)
# TODO: Define values for the model parameters
p = {'a': 40.0,  # [m2 kgC-1] structural specific leaf area
     'alpha': 2E-9,  # [kgCO2 J-1] leaf photosynthetic efficiency
     'beta': 0.05,
     'gama': 0.0,
     'k': 0.5,
     'm': 0.1,
     'M': 0.02,
     'mu_m': 0.5,
     'P0': 0.432,
     'phi': 0.9,
     'Tmax': 42,
     'Tmin': 0,
     'Topt': 20,
     'Y': 0.75,
     'z': 1.33
     }
# Parameters adjusted manually to obtain growth
# TODO: If needed, adjust the values for 2 or 3 parameters to obtain growth
# p[???] = ???
# p[???] = ???

# Disturbances (assumed constant for this test)
# 2-column arrays: Column 1 for time. Column 2 for the constant value.
# PAR [J m-2 d-1], environment temperature [°C], and
# water availability index [-]
# TODO: Fill in sensible constant values for T and I0.
d = {'I0': np.array([tsim, np.full((tsim.size,), 20.0)]).T,
     'T': np.array([tsim, np.full((tsim.size,), 23.0)]).T,
     'WAI': np.array([tsim, np.full((tsim.size,), 10.0)]).T
     }

# Controlled inputs
u = {'f_Gr': 5e-8, 'f_Hr':5e-9}  # [kgDM m-2 d-1]

# Initialize grass module
grass = Grass(tsim, dt, x0, p)

# Run simulation
tspan = (tsim[0], tsim[-1])
y_grass = grass.run(tspan, d, u)

# Retrieve simulation results
# assuming 0.4 kgC/kgDM (Mohtar et al. 1997, p. 1492)
# TODO: Retrieve the simulation results
t_grass = y_grass['t']
WsDM = y_grass['Ws']
WgDM = y_grass['Wg']

# -- Plot
# TODO: Make a plot for WsDM & WgDM vs. t
plt.figure(1)
plt.plot(t_grass, WsDM)
plt.plot(t_grass, WgDM)
plt.show()
