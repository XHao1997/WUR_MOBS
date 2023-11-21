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
tsim = np.linspace(0.0, 365.0, 365+1) # [d]
dt = 1 # [d]
# Initial conditions
# TODO: Define sensible values for the initial conditions
x0 = {'Ws':1E-4,'Wg':1E-6} # [kgC m-2]
# Model parameters (as provided by Mohtar et al. 1997 p.1492-1493)
# TODO: Define values for the model parameters
p = {'a':40.0,          # [m2 kgC-1] structural specific leaf area
     'alpha':2E-9,      # [kgCO2 J-1] leaf photosynthetic efficiency
     'beta':0.05,       # [d-1] senescence rate
     'k':0.5,           # [-] extinction coefficient of canopy
     'm':0.1,           # [-] leaf transmission coefficient
     'M':0.02,          # [d-1] maintenance respiration coefficient
     'mu_m':0.5,        # [d-1] max. structural specific growth rate
     'P0':0.432,        # [kgCO2 m-2 d-1] max photosynthesis parameter
     'phi':0.9,         # [-] photoshynth. fraction for growth
     'Tmin':0.0,        # [°C] maximum temperature for growth
     'Topt':20.0,       # [°C] minimum temperature for growth
     'Tmax':42.0,       # [°C] optimum temperature for growth
     'Y':0.75,          # [-] structure fraction from storage
     'z':1.33           # [-] bell function power
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
d = {'I0':np.array([tsim, np.full((tsim.size,), 800.0*(100**2))]).T,
     'T':np.array([tsim, np.full((tsim.size,), 20.0)]).T,
     'WAI':np.array([tsim, np.full((tsim.size,),1.0)]).T
     }

# Controlled inputs
u = {'f_Gr':0, 'f_Hr':0}            # [kgDM m-2 d-1]

# Initialize grass module
grass = Grass(tsim,dt,x0,p)

# Run simulation
tspan = (tsim[0],tsim[-1])
y_grass = grass.run(tspan,d,u)

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


