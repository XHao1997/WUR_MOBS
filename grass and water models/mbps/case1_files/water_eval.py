# -*- coding: utf-8 -*-
"""
FTE34806 - Modelling of Biobased Production Systems
MSc Biosystems Engineering, WUR
@authors:   -- write your team names here --

Evaluation of the soil water model

NOTE: To change the simulation from 1[d] to 1[hr] time step:
    1) change tsim
        tsim = np.linspace(0, 365, 24*365+1)
    2) change dt
        dt = 1/24
    3) add hour in t_ini and t_end, e.g.:
        t_ini = '20170101 1'
        t_end = '20180101 1'
    4) comment out daily weather data, and uncomment hourly weather data
    5) change temperature string from 'TG' to 'T':
        T = data_weather.loc[t_ini:t_end,'T'].values
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from mbps.models.water import Water

plt.style.use('ggplot')

# Simulation time
# TODO: Define the simulation time and integration time step 
tsim = np.linspace(0, 365, 366)
dt = 1

# Initial conditions
# Define the dictionary of initial conditions
x0 = {'L1': 54,      #[mm] Water level in layer1
      'L2': 80,      #[mm] Water level in layer2
      'L3': 144,      #[mm] Water level in layer3
      'DSD': 1}     #[d] Days since damp


# Castellaro et al. 2009, and assumed values for soil types and layers
# TODO: Define the dictionary of values for model parameters
p = {'S': 10,                # [mm d-1] parameter of precipitation retention
     'alpha': 1.29E-6,            # [mm J-1] Priestley-Taylor parameter
     'gamma': 0.68,            # [mbar °C-1] Psychrometric constant
     'alb': 0.23,              # [-] Albedo of soil
     'kcrop': 0.9,            # [-] Evapotranspiration coefficient
     'WAIc': 0.75,             # [-] Critical water value for water availability index
     'theta_fc1': 0.36,        # [-] Field capacity of soil layer 1
     'theta_fc2': 0.32,        # [-] Field capacity of soil layer 2
     'theta_fc3': 0.24,        # [-] Field capacity of soil layer 3
     'theta_pwp1': 0.21,       # [-] Permanent wilting point of soil layer 1
     'theta_pwp2': 0.17,       # [-] Permanent wilting point of soil layer 2
     'theta_pwp3': 0.10,       # [-] Permanent wilting point of soil layer 3
     'D1': 150,               # [mm] Depth of Soil layer 1
     'D2': 250,               # [mm] Depth of soil layer 2
     'D3': 600,               # [mm] Depth of soil layer 3
     'krf1': 0.25,             # [-] Rootfraction layer 1
     'krf2': 0.5,             # [-] Rootfraction layer 2
     'krf3': 0.25,             # [-] Rootfraction layer 3
     'mlc': 0.2               # [-] Fraction of soil covered by mulch
     }
# Disturbances (assumed constant for test)
# environment temperature [°C], global irradiance [J m-2 d-1], 
# precipitation [mm d-1], leaf area index [-]
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

# Hourly data
# data_weather = pd.read_csv(
#     '../data/uurgeg_260_2011-2020.csv',
#     skipinitialspace=True, # ignore spaces after comma separator
#     header = 31-3, # row with column names, 0-indexed, excluding spaces
#     usecols = ['YYYYMMDD', 'HH', 'T', 'Q', 'RH'], # columns to use
#     parse_dates = [[0,1]], # Combine first two columns as index
#     index_col = 0, # column with row names, from used & parsed columns, 0-indexed
#     )

data_LAI = pd.read_csv('../../data/LAI.csv') # Dummy LAI from grass evaluation
data_LAI = data_LAI.iloc[0:366, :]
T = data_weather.loc[t_ini:t_end,'TG'].values      # [0.1 °C] Env. temperature
I_glb = data_weather.loc[t_ini:t_end,'Q'].values  # [J cm-2 d-1] Global irr. 
f_prc = data_weather.loc[t_ini:t_end,'RH'].values # [0.1 mm d-1] Precipitation
f_prc[f_prc<0.0] = 0 # correct data that contains -0.1 for very low values

# TODO: Apply the necessary conversions
T = T / 10
I_glb = I_glb * 100 *100
f_prc = f_prc / 10
print(np.array([data_LAI.iloc[:,0].values, data_LAI.iloc[:,1]]).T.shape)
d = {'I_glb' : np.array([tsim, I_glb]).T, 
    'T' : np.array([tsim, T]).T,
    'f_prc': np.array([tsim, f_prc]).T,
    'LAI' : np.array([data_LAI.iloc[:,0].values, data_LAI.iloc[:,1]]).T
     }

# Controlled inputs
u = {'f_Irg':0}            # [mm d-1]

# Initialize module
water = Water(tsim, dt, x0, p)

# Run simulation
tspan = (tsim[0], tsim[-1])
y_water = water.run(tspan, d, u)

# Retrieve simulation results
# TODO: Retrive variables from the dictionary of model outputs
t_water = y_water['t']
L1 = y_water['L1']
L2 = y_water['L2']
L3 = y_water['L3']

Fpe = water.f['f_Pe']
Ftr1 = water.f['f_Tr1']
Ftr2 = water.f['f_Tr2']
Ftr3 = water.f['f_Tr3']
Fev = water.f['f_Ev']
Fdr1 = water.f['f_Dr1']
Fdr2 = water.f['f_Dr2']
Fdr3 = water.f['f_Dr3']
f_Irg = water.f['f_Irg']
print(f_Irg.shape)
# Plot
# TODO: Make plots for the state variables (as L and theta),
# and the volumetric flows.
# Include lines for the pwp and fc for each layer.
plt.figure(1)
plt.plot(t_water, L1, label='L1')
plt.plot( t_water, L2, label='L2',linestyle='--')
plt.plot(t_water, L3, label='L3', marker= 'x', linestyle='-')
#plt.plot(t_water, p['theta_fc1']*p['D1'], label='L1')
#plt.plot( t_water, p['theta_fc1']*p['D1'], label='L2',linestyle='--')
#plt.plot(t_water, p['theta_fc1']*p['D1'], label='L3', marker= 'x', linestyle='-')
plt.legend()
plt.figure(2)
plt.subplot(3, 1,1)

plt.plot(tsim, Fpe, label='Fpe')
plt.plot( tsim, -Fev, label='Fev')
plt.plot(tsim, f_Irg, label='Firg')
plt.plot(tsim, -Ftr1, label='Ftr1')
plt.plot( tsim, -Fdr1, label='Fdr1')
plt.ylim(-10,10)
plt.legend()

plt.subplot(3, 1,2)
plt.plot(tsim, -Ftr2, label='Ftr2')
plt.plot( tsim, -Fdr2, label='Fdr2')
plt.ylim(-10,0)
plt.legend()

plt.subplot(3, 1,3)
plt.plot(tsim, -Ftr3, label='Ftr3')
plt.plot( tsim, -Fdr3, label='Fdr3')
plt.ylim(-10, 0)
plt.legend()
plt.show()
