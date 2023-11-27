# -*- coding: utf-8 -*-
"""
FTE34806 - Modelling of Biobased Production Systems
MSc Biosystems Engineering, WUR
@authors:   Daniel Reyes Lastiri, Stefan Maranus,
            Rachel van Ooteghem, Tim Hoogstad

Tutorial for the logistic growth model

This comment section is required for the beginning of any Python file.
It provides a general description of the file's contents.
"""

''' Explanation
Comment sections are created with triple quotation marks.
In the course, we use these sections to explain details of the code.
The script file starts by importing the functions used in the file.
We suggest to import first Python functions, then import our own functions.
'''

# Import Python functions
import numpy as np
import matplotlib.pyplot as plt
# Import our own functions
from mbps.models.log_growth import LogisticGrowth

# Simulation time array

# TODO: Define the array for simulation time, using the function 'linspace'.
# The array ranges from 0 to 10, with a step size of 1.0 [d].
# How many elements should this array have?
tsim = np.linspace(0, 10, 101)  # [d]

# Initialize model object

# TODO: Define the following variables:
# model integration time step size of 1.0 [d]
# dictionary of initial conditions for m = 1.0 [gDM m-2]
# dictionary of parameters for r = 1.2 [d-1] and K = 100 [gDM m-2]
dt = 0.1  # [d] time-step size
x0 = {'m': 1.0}  # [gDM m-2] initial conditions
p = {'r': 1.2, 'K': 100}  # [d-1], [gDM m-2] model parameters
lg = LogisticGrowth(tsim, dt, x0, p)

# Run model
tspan = (tsim[0], tsim[-1])
y = lg.run(tspan)


# anaytical solution
m0, r, K = x0['m'], p['r'], p['K']
m_anl = K/(1 + (K-m0)/m0*np.exp(-r*tsim))

# Plot results
plt.figure(1)
plt.plot(y['t'], y['m'],label='euler')
plt.plot(y['t_rk'], y['m_rk'],label='rk')
plt.plot(tsim, m_anl,linestyle='dashed',color='red',label='analytical')
plt.xlabel(r'$time\ [d]$')
plt.ylabel(r'$mass\ [gDM\ m^{-2}]$')
plt.show()
plt.figure(2)
plt.plot(y['t'], y['m']-m_anl,label='euler')
plt.plot(y['t_rk'], y['m_rk']-m_anl,label='rk')
plt.xlabel(r'$time\ [d]$')
plt.ylabel(r'$mass\ [gDM\ m^{-2}]$')
plt.show()