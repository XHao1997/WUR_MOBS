# -*- coding: utf-8 -*-
"""
FTE34806 - Modelling of Biobased Production Systems
MSc Biosystems Engineering, WUR
@authors:   -- Write your team names here --

Tutorial for the uncertainty analysis
"""
import numpy as np
import matplotlib.pyplot as plt

from mbps.functions.uncertainty import fcn_plot_uncertainty

plt.style.use('ggplot')

# Random number generator. A seed is specified to allow for reproducibility.
rng = np.random.default_rng(seed=12)

# Simulation time array
tsim = np.linspace(0, 50, 50+1) # [d]

# Monte Carlo simulations
n_sim = 1000 # number of simulations
# Initialize array of outputs, shape (len(tsim), len(n_sim))
m_arr = np.full((tsim.size,n_sim), np.nan)
# Run simulations
for j in range(n_sim):
    # TODO: Specify the parameter values or sample from a normal distribution.
    # Sample random parameters
    r = rng.normal(0.05, 0.005)
    K = rng.normal(2, 0.2)
    # TODO: define the growth model function
    m = K*(1-np.exp(-r*tsim))
    # Store current simulation 'j' in the corresponding array column. 
    m_arr[:,j] = m
    
# Plot results
plt.figure(1)
# TODO: Make a plot for the first 12 simulations (the first 12 columns in y_arr)
plt.plot(tsim, m_arr[:, :20])
plt.xlabel(r'$time\ [d]$')
plt.ylabel('biomass ' + r'$[kgDM\ m^{-2}]$')
# plt.show()
plt.figure(2)
ax2 = plt.gca()
ax2 = fcn_plot_uncertainty(ax2, tsim, m_arr, ci=[0.50,0.68,0.95])
plt.xlabel(r'$time\ [d]$')
plt.ylabel('biomass ' + r'$[kgDM\ m^{-2}]$')
plt.show()