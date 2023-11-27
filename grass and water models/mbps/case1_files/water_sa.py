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

from mbps.models.water import Water
from mbps.classes.module import Module

plt.style.use('ggplot')

# TODO: Define the required variables for the grass module

# Simulation time
tsim = np.linspace(0.0, 365.0, 365+1) # [d]
dt = 1 # [d]


# Initial conditions
x0 = {'L1': 54,      #[mm] Water level in layer1
      'L2': 80,      #[mm] Water level in layer2
      'L3': 144,      #[mm] Water level in layer3
      'DSD': 1}     #[d] Days since damp


# Model parameters (as provided by Mohtar et al. 1997 p.1492-1493)
p = {'S': 10,                   # [mm d-1] parameter of precipitation retention
     'alpha': 1.29E-6,          # [mm J-1] Priestley-Taylor parameter
     'gamma': 0.68,             # [mbar °C-1] Psychrometric constant
     'alb': 0.23,               # [-] Albedo of soil
     'kcrop': 0.9,              # [-] Evapotranspiration coefficient
     'WAIc': 0.75,              # [-] Critical water value for water availability index
     'theta_fc1': 0.36,         # [-] Field capacity of soil layer 1
     'theta_fc2': 0.32,         # [-] Field capacity of soil layer 2
     'theta_fc3': 0.24,         # [-] Field capacity of soil layer 3
     'theta_pwp1': 0.21,        # [-] Permanent wilting point of soil layer 1
     'theta_pwp2': 0.17,        # [-] Permanent wilting point of soil layer 2
     'theta_pwp3': 0.10,        # [-] Permanent wilting point of soil layer 3
     'D1': 150,                 # [mm] Depth of Soil layer 1
     'D2': 250,                 # [mm] Depth of soil layer 2
     'D3': 600,                 # [mm] Depth of soil layer 3
     'krf1': 0.25,              # [-] Rootfraction layer 1
     'krf2': 0.5,               # [-] Rootfraction layer 2
     'krf3': 0.25,              # [-] Rootfraction layer 3
     'mlc': 0.2                 # [-] Fraction of soil covered by mulch
     }





# Disturbances
t_ini = '20170101'
t_end = '20180101'

# Daily data
data_weather = pd.read_csv(
    '../../data/etmgeg_260.csv', # .. to move up one directory from current directory
    skipinitialspace=True, # ignore spaces after comma separator
    header = 47-3, # row with column names, 0-indexed, excluding spaces
    usecols = ['YYYYMMDD', 'TG', 'Q', 'RH'], # columns to use
    index_col = 0, # column with row names from used columns, 0-indexed
    )


# Retrieve relevant arrays from pandas dataframe
data_LAI = pd.read_csv('../../data/LAI.csv') # Dummy LAI from grass evaluation
data_LAI = data_LAI.iloc[0:366, :]
T = data_weather.loc[t_ini:t_end,'TG'].values      # [0.1 °C] Env. temperature
I_glb = data_weather.loc[t_ini:t_end,'Q'].values  # [J cm-2 d-1] Global irr.
f_prc = data_weather.loc[t_ini:t_end,'RH'].values # [0.1 mm d-1] Precipitation
f_prc[f_prc<0.0] = 0 # correct data that contains -0.1 for very low values

# Aply the necessary conversions of units
T = T / 10
I_glb = I_glb * 100 *100
f_prc = f_prc / 10
# print(np.array([data_LAI.iloc[:,0].values, data_LAI.iloc[:,1]]).T.shape)
d = {'I_glb' : np.array([tsim, I_glb]).T,
    'T' : np.array([tsim, T]).T,
    'f_prc': np.array([tsim, f_prc]).T,
    'LAI' : np.array([data_LAI.iloc[:,0].values, data_LAI.iloc[:,1]]).T
     }


# Controlled inputs
u = {'f_Irg':0}            # [mm d-1]

# Initialize module
water = Water(tsim, dt, x0, p)


# Normalized sensitivities
ns1 = water.ns(x0, p, d=d, u=u, y_keys=('L1',))
ns2 = water.ns(x0, p, d=d, u=u, y_keys=('L2',))
ns3 = water.ns(x0, p, d=d, u=u, y_keys=('L3',))


# Calculate mean NS through time
# TODO: use the ns DataFrame to calculate mean NS per parameter
ns_mean1 = np.mean(ns1['L1'])
# print('Mean NS per parameter for Layer 1: ', ns_mean1)
ns_mean2 = np.mean(ns2['L2'])
# print('Mean NS per parameter for Layer 2: ', ns_mean2)
ns_mean3 = np.mean(ns3['L3'])
# print('Mean NS per parameter for Layer 3: ', ns_mean3)


# -- Plots
# TODO: Make the necessary plots (example provided below)
# plt.figure(1)
# plt.plot(grass.t, ns['L1'], label=ns.columns.levels[1])
# # plt.plot(grass.t, ns['L1','a','+'], label='a +')
#
# plt.legend()
# plt.title('All parameters')
# plt.xlabel(r'$time\ [d]$')
# plt.ylabel('normalized sensitivity [-]')

plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.plot(water.t, ns1['L1','S', '+'], label=r'S +')
plt.plot(water.t, ns1['L1','alpha', '+'], label=r'$\alpha +$')
plt.plot(water.t, ns1['L1','gamma', '+'], label=r'gamma +')
plt.plot(water.t, ns1['L1','alb', '+'], label=r'alb +')
plt.plot(water.t, ns1['L1','kcrop', '+'], label=r'kcrop +')
plt.plot(water.t, ns1['L1','WAIc', '+'], label=r'WAIc +')
plt.plot(water.t, ns1['L1','theta_fc1', '+'], label=r'theta_fc1 +')
plt.plot(water.t, ns1['L1','theta_pwp1', '+'], label=r'theta_pwp1 +')
plt.plot(water.t, ns1['L1','D1', '+'], label=r'D1 +')
plt.plot(water.t, ns1['L1','krf1', '+'], label=r'krf1 +')
plt.plot(water.t, ns1['L1','mlc', '+'], label=r'mlc +')
plt.legend()
plt.xlabel(r'$time\ [d]$')
plt.ylabel('normalized sensitivity [-]')

plt.subplot(1, 2, 2)
plt.plot(water.t, ns1['L1','S', '-'], label=r'S -')
plt.plot(water.t, ns1['L1','alpha', '-'], label=r'$\alpha -$')
plt.plot(water.t, ns1['L1','gamma', '-'], label=r'gamma -')
plt.plot(water.t, ns1['L1','alb', '-'], label=r'alb -')
plt.plot(water.t, ns1['L1','kcrop', '-'], label=r'kcrop -')
plt.plot(water.t, ns1['L1','WAIc', '-'], label=r'WAIc -')
plt.plot(water.t, ns1['L1','theta_fc1', '-'], label=r'theta_fc1 -')
plt.plot(water.t, ns1['L1','theta_pwp1', '-'], label=r'theta_pwp1 -')
plt.plot(water.t, ns1['L1','D1', '-'], label=r'D1 -')
plt.plot(water.t, ns1['L1','krf1', '-'], label=r'krf1 -')
plt.plot(water.t, ns1['L1','mlc', '-'], label=r'mlc -')
plt.legend()
plt.xlabel(r'$time\ [d]$')


plt.tight_layout()

# plt.figure(figsize=(16, 8))
# plt.subplot(1, 2, 1)
# plt.plot(grass.t, ns['Wg','gama', '+'], label=r'$\gamma +$')
# plt.plot(grass.t, ns['Wg','beta', '+'], label=r'$\beta +$')
# plt.plot(grass.t, ns['Wg','k', '+'], label=r'K +')
# plt.plot(grass.t, ns['Wg','M', '+'], label=r'M +')
# plt.plot(grass.t, ns['Wg','Topt', '+'], label=r'$T_{opt}$ +')
# plt.plot(grass.t, ns['Wg','z', '+'], label=r'z +')
#
# plt.legend()
# plt.xlabel(r'$time\ [d]$')
# plt.ylabel('normalized sensitivity [-]')
#
# plt.subplot(1, 2, 2)
# plt.plot(grass.t, ns['Wg','gama', '-'], label=r'$\gamma -$', linestyle='--')
# plt.plot(grass.t, ns['Wg','beta', '-'], label=r'$\beta -$', linestyle='--')
# plt.plot(grass.t, ns['Wg','k', '-'], label='K -', linestyle='--')
# plt.plot(grass.t, ns['Wg','M', '-'], label='M -', linestyle='--')
# plt.plot(grass.t, ns['Wg','Topt', '-'], label=r'$T_{opt}$ -', linestyle='--')
# plt.plot(grass.t, ns['Wg','z', '-'], label='z -', linestyle='--')
# # plt.plot(grass.t, ns['Wg','a','+'], label='a +')
# # plt.plot(grass.t, ns['Wg','beta','-'], label='beta -', linestyle='--')
# # plt.plot(grass.t, ns['Wg','beta','+'], label='beta +')
# plt.legend()
# plt.xlabel(r'$time\ [d]$')
#
# plt.tight_layout()
#
#
# # Graph for in word file
# plt.figure(figsize=(8, 14))
# plt.subplot(2, 1, 1)
# plt.plot(grass.t, ns['Wg','a', '+'], label='a +')
# plt.plot(grass.t, ns['Wg','beta', '+'], label=r'$\beta +$')
# plt.plot(grass.t, ns['Wg','Y', '+'], label=r'Y +')
# plt.plot(grass.t, ns['Wg','M', '+'], label=r'M +')
# plt.plot(grass.t, ns['Wg','phi', '+'], label=r'$\phi +$')
#
# plt.legend()
# plt.ylabel('normalized sensitivity [-]')
#
# plt.subplot(2, 1, 2)
# plt.plot(grass.t, ns['Wg','a', '-'], label='a -', linestyle='--')
# plt.plot(grass.t, ns['Wg','beta', '-'], label=r'$\beta -$', linestyle='--')
# plt.plot(grass.t, ns['Wg','Y', '-'], label='Y -', linestyle='--')
# plt.plot(grass.t, ns['Wg','M', '-'], label='M -', linestyle='--')
# plt.plot(grass.t, ns['Wg','phi', '-'], label=r'$\phi -$', linestyle='--')
#
# plt.legend()
# plt.xlabel(r'$time\ [d]$')
# plt.ylabel('normalized sensitivity [-]')
#
# plt.tight_layout()
plt.show()