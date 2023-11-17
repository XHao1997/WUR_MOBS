# -*- coding: utf-8 -*-
"""
FTE34806 - Modelling of Biobased Production Systems
MSc Biosystems Engineering, WUR
@authors:   -- add your team names here --

Sensitivity analysis of the grass growth model
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mbps.models.grass import Grass
from mbps.classes.module import Module

plt.style.use('ggplot')

# TODO: Define the required variables for the grass module

# Simulation time
tsim = np.linspace(0.0, 365.0, 365 + 1)  # [d]
dt = 1  # [d]
# Initial conditions
x0 = {'Ws': 1E-2, 'Wg': 6E-2}  # [kgC m-2]
# Model parameters (as provided by Mohtar et al. 1997 p.1492-1493)
p = {'a': 40.0,  # [m2 kgC-1] structural specific leaf area
     'alpha': 2e-9,  # [kgCO2 J-1] leaf photosynthetic efficiency
     'beta': 0.05,
     'gama': 0.01,
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
# Model parameters adjusted manually to obtain growth
# p[???] = ???
# p[???] = ???

# Disturbances
# PAR [J m-2 d-1], environment temperature [°C], and
# water availability index [-]
t_ini = '20160101'
t_end = '20161231'
data_weather = pd.read_csv(
    '../../data/temp.csv',  # to move up one directory from current directory
    usecols=['YYYYMMDD', 'TG', 'Q', 'RH'],  # columns to use
    index_col=0,  # column with row names from used columns, 0-indexed
)
# Retrieve relevant arrays from pandas dataframe
T = data_weather.loc[t_ini:t_end, 'TG'].values  # [0.1 °C] Env. temperature
I_gl = data_weather.loc[t_ini:t_end, 'Q'].values  # [J cm-2 d-1] Global irradiance
# Aply the necessary conversions of units
T = T / 10  # [???] to [???] Env. temperature
I0 = I_gl * 60 * 60 * 24  # [???] to [???] Global irradiance to PAR
# print(I_gl.shape)
# print(I0.shape)
# print(I_gl.shape)
# print(tsim.shape)
# Dictionary of disturbances (2D arrays, with col 1 for time, and col 2 for d)
d = {'T': np.array([tsim, T]).T,
     'I0': np.array([tsim, I0]).T,
     'WAI': np.array([tsim, np.full((tsim.size,), 1.0)]).T
     }

# Controlled inputs
u = {'f_Gr': 0, 'f_Hr': 0}  # [kgDM m-2 d-1]

# Initialize grass module
grass = Grass(tsim, dt, x0, p)

# Normalized sensitivities
ns = grass.ns(x0, p, d=d, u=u, y_keys=('Wg',))
# print(ns.head())
# Calculate mean NS through time
# TODO: use the ns DataFrame to calculate mean NS per parameter
ns_mean = np.mean(ns['Wg'])
print('Mean NS per parameter: ', ns_mean)

# -- Plots
# TODO: Make the necessary plots (example provided below)
# plt.figure(1)
# plt.plot(grass.t, ns['Wg'], label=ns.columns.levels[1])
# # plt.plot(grass.t, ns['Wg','a','+'], label='a +')
#
# plt.legend()
# plt.title('All parameters')
# plt.xlabel(r'$time\ [d]$')
# plt.ylabel('normalized sensitivity [-]')

plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.plot(grass.t, ns['Wg', 'a', '+'], label=r'a +')
plt.plot(grass.t, ns['Wg', 'alpha', '+'], label=r'$\alpha +$')
plt.plot(grass.t, ns['Wg', 'm', '+'], label=r'm +')
plt.plot(grass.t, ns['Wg', 'mu_m', '+'], label=r'$\mu_m$ +')
plt.plot(grass.t, ns['Wg', 'P0', '+'], label=r'P0 +')
plt.plot(grass.t, ns['Wg', 'phi', '+'], label=r'$\phi +$')
plt.plot(grass.t, ns['Wg', 'Y', '+'], label=r'Y +')
plt.legend()
plt.xlabel(r'$time\ [d]$')
plt.ylabel('normalized sensitivity [-]')

plt.subplot(1, 2, 2)
plt.plot(grass.t, ns['Wg', 'a', '-'], label=r'a -', linestyle='--')
plt.plot(grass.t, ns['Wg', 'alpha', '-'], label=r'$\alpha -$', linestyle='--')
plt.plot(grass.t, ns['Wg', 'm', '-'], label=r'm -', linestyle='--')
plt.plot(grass.t, ns['Wg', 'mu_m', '-'], label=r'$\mu_m$ -', linestyle='--')
plt.plot(grass.t, ns['Wg', 'P0', '-'], label=r'P0 -', linestyle='--')
plt.plot(grass.t, ns['Wg', 'phi', '-'], label=r'$\phi -$', linestyle='--')
plt.plot(grass.t, ns['Wg', 'Y', '-'], label=r'Y -', linestyle='--')
# plt.plot(grass.t, ns['Wg','a','+'], label='a +')
# plt.plot(grass.t, ns['Wg','beta','-'], label='beta -', linestyle='--')
# plt.plot(grass.t, ns['Wg','beta','+'], label='beta +')
plt.legend()
plt.xlabel(r'$time\ [d]$')

plt.tight_layout()

plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.plot(grass.t, ns['Wg', 'gama', '+'], label=r'$\gamma +$')
plt.plot(grass.t, ns['Wg', 'beta', '+'], label=r'$\beta +$')
plt.plot(grass.t, ns['Wg', 'k', '+'], label=r'K +')
plt.plot(grass.t, ns['Wg', 'M', '+'], label=r'M +')
plt.plot(grass.t, ns['Wg', 'Topt', '+'], label=r'$T_{opt}$ +')
plt.plot(grass.t, ns['Wg', 'z', '+'], label=r'z +')

plt.legend()
plt.xlabel(r'$time\ [d]$')
plt.ylabel('normalized sensitivity [-]')

plt.subplot(1, 2, 2)
plt.plot(grass.t, ns['Wg', 'gama', '-'], label=r'$\gamma -$', linestyle='--')
plt.plot(grass.t, ns['Wg', 'beta', '-'], label=r'$\beta -$', linestyle='--')
plt.plot(grass.t, ns['Wg', 'k', '-'], label='K -', linestyle='--')
plt.plot(grass.t, ns['Wg', 'M', '-'], label='M -', linestyle='--')
plt.plot(grass.t, ns['Wg', 'Topt', '-'], label=r'$T_{opt}$ -', linestyle='--')
plt.plot(grass.t, ns['Wg', 'z', '-'], label='z -', linestyle='--')
# plt.plot(grass.t, ns['Wg','a','+'], label='a +')
# plt.plot(grass.t, ns['Wg','beta','-'], label='beta -', linestyle='--')
# plt.plot(grass.t, ns['Wg','beta','+'], label='beta +')
plt.legend()
plt.xlabel(r'$time\ [d]$')

plt.tight_layout()
# plt.figure(3)
# plt.plot(grass.t, ns['Wg','gama'])
# plt.plot(grass.t, ns['Wg','beta'])
# plt.plot(grass.t, ns['Wg','k'])
# plt.plot(grass.t, ns['Wg','M'])
# plt.plot(grass.t, ns['Wg','Topt'])
# plt.plot(grass.t, ns['Wg','z'])

# plt.plot(grass.t, ns['Wg','a','+'], label='a +')
# plt.plot(grass.t, ns['Wg','beta','-'], label='beta -', linestyle='--')
# plt.plot(grass.t, ns['Wg','beta','+'], label='beta +')
# plt.legend()
# plt.xlabel(r'$time\ [d]$')
# plt.ylabel('normalized sensitivity [-]')


plt.tight_layout()
plt.show()
