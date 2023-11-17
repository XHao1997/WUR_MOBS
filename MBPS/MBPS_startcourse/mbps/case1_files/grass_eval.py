# -*- coding: utf-8 -*-
"""
FTE34806 - Modelling of Biobased Production Systems
MSc Biosystems Engineering, WUR
@authors:   -- add your team names here --

Evaluation of the grass growth model
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mbps.models.grass import Grass

# Disturbances
# PAR [J m-2 d-1], env. temperature [°C], and water availability index [-]
# TODO: Specify the corresponding dates to read weather data (see csv file).
t_ini = "20160101"
t_end = "20161231"
print(os.getcwd())
data_weather = pd.read_csv(
    '../../data/temp.csv',  # to move up one directory from current directory
    usecols=['YYYYMMDD', 'TG', 'Q', 'RH'],  # columns to use
    index_col=0,  # column with row names from used columns, 0-indexed
)
data_grass = pd.read_csv("../../data/temp_real.csv", usecols=['X', 'Y'])
t_grass_data = np.arange(0, 252, 0.1)

print(data_weather.head())
plt.style.use('ggplot')

# Grass data
# TODO: define numpy arrays with measured grass data in the Netherlands
t_grass_data = np.arange(0, 252, 0.1)
y_int = np.interp(t_grass_data, data_grass['X'], data_grass['Y'])
m_grass = np.zeros_like(t_grass_data)
for i in range(t_grass_data.shape[0]):
    m_grass[i] = np.trapz(y_int[:i], np.array(t_grass_data[:i]))

# Simulation time
tsim = np.linspace(0.0, 365.0, 365 + 1)  # [d]
dt = 1  # [d]

# Initial conditions
# TODO: define sensible values for the initial conditions
# x0 = {???, ???}  # [kgC m-2]

# Model parameters, as provided by Mohtar et al. (1997)
# TODO: define the varameper values in the dictionary p
# p = { ??? }

# Model parameters adjusted manually to obtain growth
# # TODO: (Further) adjust 2-3 parameter values to match measured growth behaviour
# p[???] = ???
# p[???] = ???
#
#
# # # Retrieve relevant arrays from pandas dataframe
# T = data_weather.loc[t_ini:t_end,'TG'].values    # [0.1 °C] Env. temperature
# I_gl = data_weather.loc[t_ini:t_end,'Q'].values # [J cm-2 d-1] Global irradiance
# # Aply the necessary conversions of units
#
# T = T/10    # [???] to [???] Env. temperature
# I0 = I_gl/(100**2)  # [J/cm2] to [W/m2] Global irradiance to PAR
# # Dictionary of disturbances (2D arrays, with col 1 for time, and col 2 for d)
# d = {'T':np.array([tsim, T]).T,
#      'I0':np.array([tsim, I0]).T,
#      'WAI':np.array([tsim, np.full((tsim.size,),1.0)]).T
#      }
#
# # Controlled inputs
# u = {'f_Gr':0, 'f_Hr':0}            # [kgDM m-2 d-1]
#
# # Initialize module
# # TODO: Call the module Grass to initialize an instance
# grass = Grass(tsim, dt, x0, p)
#
# # Run simulation
# # TODO: Call the method run to generate simulation results
# tspan = (tsim[0], tsim[-1])
# y_grass = grass.run(???, ???, ???)

# # Retrieve simulation results
# # assuming 0.4 kgC/kgDM (Mohtar et al. 1997, p. 1492)
# # TODO: Retrieve the simulation results
# t_grass = y_grass['t']
# WsDM = ???
# WgDM = ???
#
# # Plot
# # TODO: Make a plot for WsDM, WgDM and grass measurement data.
# plt.figure(1)
#

fig, ax = plt.subplots()
ax.set_xlabel(r'day', fontsize=10)
ax.set_ylabel(u'grass mass ($\mathregular{kgCm^{-2}}$)', fontsize=10)
ax.set_title('grass yield per day')
ax.scatter(t_grass_data, m_grass, s=1)
plt.tight_layout()
plt.show()
