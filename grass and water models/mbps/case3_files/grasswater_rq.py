# -*- coding: utf-8 -*-
"""
FTE34806 - Modelling of Biobased Production Systems
MSc Biosystems Engineering, WUR
@authors:   Daniel Reyes Lastiri, Stefan Maranus,
            Rachel van Ooteghem, Tim Hoogstad

Evaluation of the grass & water model
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mbps.models.grass_sol import Grass
from mbps.models.water_sol import Water
from mbps.functions.slide_win import slide_win
import os

plt.style.use('ggplot')

# Simulation time
tsim = np.linspace(0, 365, 365 + 1)  # [d]

# Weather data (disturbances shared across models)
t_ini = '20040101'
t_end = '20041231'
t_weather = np.linspace(0, 365, 365 + 1)

data_weather = pd.read_csv(
    '../../data/etmgeg_1995_2020_deelen.csv',  # .. to move up one directory from current directory
    skipinitialspace=True,  # ignore spaces after comma separator
    header=10,  # row with column names, 0-indexed, excluding spaces
    usecols=['YYYYMMDD', 'TG', 'Q', 'RH'],  # columns to use
    index_col=0,  # column with row names from used columns, 0-indexed
)

# ---- Grass sub-model
# Step size
dt_grs = 1  # [d]

# Initial conditions
# TODO: Specify suitable initial conditions for the grass sub-model
x0_grs = {'Ws': 1e-3, 'Wg': 1.20e-2}  # [kgC m-2]

# Model parameters (as provided by Mohtar et al. 1997 p.1492-1493)
p_grs = {'a': 40.0,  # [m2 kgC-1] structural specific leaf area
         'alpha': 2E-9,  # [kgCO2 J-1] leaf photosynthetic efficiency
         'beta': 0.05,  # [d-1] senescence rate
         'k': 0.5,  # [-] extinction coefficient of canopy
         'm': 0.1,  # [-] leaf transmission coefficient
         'M': 0.02,  # [d-1] maintenance respiration coefficient
         'mu_m': 0.5,  # [d-1] max. structural specific growth rate
         'P0': 0.432,  # [kgCO2 m-2 d-1] max photosynthesis parameter
         'phi': 0.9,  # [-] photoshynth. fraction for growth
         'Tmin': 0.0,  # [°C] maximum temperature for growth
         'Topt': 20.0,  # [°C] minimum temperature for growth
         'Tmax': 42.0,  # [°C] optimum temperature for growth
         'Y': 0.75,  # [-] structure fraction from storage
         'z': 1.33  # [-] bell function power
         }
# Model parameters adjusted manually to obtain growth
# TODO: Adjust a few parameters to obtain growth.
# Satrt by using the modifications from Case 1.
# If needed, adjust further those or additional parameters
p_grs['alpha'] = 9.337e-9
p_grs['beta'] = 0.01
p_grs['T_min'] = 4.5

# Disturbances
# PAR [J m-2 d-1], environment temperature [°C], leaf area index [-]
T = data_weather.loc[t_ini:t_end, 'TG'].values  # [0.1 °C] Env. temperature
I_gl = data_weather.loc[t_ini:t_end, 'Q'].values  # [J cm-2 d-1] Global irr.

T = T / 10  # [0.1 °C] to [°C] Environment temperature
I0 = 0.45 * I_gl * 1E4 / dt_grs  # [J cm-2 d-1] to [J m-2 d-1] Global irr. to PAR

d_grs = {'T': np.array([t_weather, T]).T,
         'I0': np.array([t_weather, I0]).T,
         }

# Initialize module
grass = Grass(tsim, dt_grs, x0_grs, p_grs)

# ---- Water sub-model
dt_wtr = 1  # [d]

# Initial conditions
# TODO: Specify suitable initial conditions for the soil water sub-model
x0_wtr = {'L1': 54, 'L2': 80, 'L3': 144, 'DSD': 1}  # 3*[mm], [d]

# Castellaro et al. 2009, and assumed values for soil types and layers
p_wtr = {'S': 10,  # [mm d-1] parameter of precipitation retention
         'alpha': 1.29E-6,  # [mm J-1] Priestley-Taylor parameter
         'gamma': 0.68,  # [mbar °C-1] Psychrometric constant
         'alb': 0.23,  # [-] Albedo (assumed constant crop & soil)
         'kcrop': 0.90,  # [mm d-1] Evapotransp coefficient, range (0.85-1.0)
         'WAIc': 0.75,  # [-] WDI critical, range (0.5-0.8)
         'theta_fc1': 0.36,  # [-] Field capacity of soil layer 1
         'theta_fc2': 0.32,  # [-] Field capacity of soil layer 2
         'theta_fc3': 0.24,  # [-] Field capacity of soil layer 3
         'theta_pwp1': 0.21,  # [-] Permanent wilting point of soil layer 1
         'theta_pwp2': 0.17,  # [-] Permanent wilting point of soil layer 2
         'theta_pwp3': 0.10,  # [-] Permanent wilting point of soil layer 3
         'D1': 150,  # [mm] Depth of Soil layer 1
         'D2': 250,  # [mm] Depth of soil layer 2
         'D3': 600,  # [mm] Depth of soil layer 3
         'krf1': 0.25,  # [-] Rootfraction layer 1 (guess)
         'krf2': 0.50,  # [-] Rootfraction layer 2 (guess)
         'krf3': 0.25,  # [-] Rootfraction layer 2 (guess)
         'mlc': 0.2,  # [-] Fraction of soil covered by mulching
         }

# Disturbances
# global irradiance [J m-2 d-1], environment temperature [°C],
# precipitation [mm d-1], leaf area index [-].
T = data_weather.loc[t_ini:t_end, 'TG'].values  # [0.1 °C] Env. temperature
I_glb = data_weather.loc[t_ini:t_end, 'Q'].values  # [J cm-2 d-1] Global irr.
f_prc = data_weather.loc[t_ini:t_end, 'RH'].values  # [0.1 mm d-1] Precipitation
f_prc[f_prc < 0.0] = 0  # correct data that contains -0.1 for very low values

T = T / 10  # [0.1 °C] to [°C] Environment temperature
I_glb = I_glb * 1E4 / dt_wtr  # [J cm-2 d-1] to [J m-2 d-1] Global irradiance
f_prc = f_prc / 10 / dt_wtr  # [0.1 mm d-1] to [mm d-1] Precipitation

d_wtr = {'I_glb': np.array([t_weather, I_glb]).T,
         'T': np.array([t_weather, T]).T,
         'f_prc': np.array([t_weather, f_prc]).T,
         }

# Initialize module
water = Water(tsim, dt_wtr, x0_wtr, p_wtr)

# ---- Run simulation
# Initial disturbance
d_grs['WAI'] = np.array([[0], [1] * 1]).T

# Iterator
# (stop at second-to-last element, and store index in Fortran order)
it = np.nditer(tsim[:-1], flags=['f_index'])
list_WgDM = []
go_harvest = False
m_harvest = []
d_harvest = []
harvest_day = None
harvest_times = 0
irr_mass = 0
for ti in it:
    # Index for current time instant
    idx = it.index
    # Integration span
    tspan = (tsim[idx], tsim[idx + 1])
    # print('Integrating', tspan)

    # harvest_to_mass = 0
    u_grs = {'f_Gr': 0, 'f_Hr': 0}  # [kgDM m-2 d-1]
    u_wtr = {'f_Irg': irr_mass}  # [mm d-1]

    go_harvest, harvest_mass, go_final_harvest, harvest_day = slide_win(list_WgDM, 7, 7e-3)
    if go_harvest:
        list_WgDM = []
        if ti<180 and ti>100 and harvest_times!=3:
            print('harvest')
            print("harvest mass: ", harvest_mass)
            print("harvest day",harvest_day)
            harvest_mass = harvest_mass

            # harvest_mass = list_WgDM[-1]*0.7*0.4
            u_grs = {'f_Gr': 0, 'f_Hr': abs(harvest_mass*0.4)}  # [kgDM m-2 d-1]

            m_harvest.append(abs(harvest_mass))
            d_harvest.append(ti)
            harvest_times +=1
            list_WgDM = []
        elif idx>200 and idx<240:
            print('final harvest')
            print("harvest mass: ", harvest_mass)
            print("harvest day",harvest_day)
            harvest_mass = abs((WgDM[idx]-0.02))
            u_grs = {'f_Gr': 0, 'f_Hr': harvest_mass*0.4}  # [kgDM m-2 d-1]
            u_wtr = {'f_Irg': 0}  # [mm d-1]
            m_harvest.append(abs(harvest_mass))
            d_harvest.append(ti)
            list_WgDM = []

    y_grs = grass.run(tspan, d_grs, u_grs)
    # Retrieve grass model outputs for water model
    d_wtr['LAI'] = np.array([y_grs['t'], y_grs['LAI']])
    # Run water model
    y_wtr = water.run(tspan, d_wtr, u_wtr)
    irr_mass = y_wtr['DSD'][1]*1.2*(idx<200)
    # print("parameter for irr", y_wtr['DSD'])

    # Retrieve water model outputs for grass model
    d_grs['WAI'] = np.array([y_wtr['t'], y_wtr['WAI']])
    WgDM = grass.y['Wg']/0.4
    if idx>90:
        list_WgDM.append(WgDM[int(ti)])

print(np.sum((m_harvest)))
print(m_harvest)
print(d_harvest)
# Retrieve simulation results
t_grs, t_wtr = grass.t, water.t
WsDM, WgDM, LAI = grass.y['Ws'] / 0.4, grass.y['Wg'] / 0.4, grass.y['LAI']
L1, L2, L3 = water.y['L1'], water.y['L2'], water.y['L3'],
WAI = water.y['WAI']

# ---- Plots
plt.figure(1)
plt.plot(t_grs, WsDM, label='WsDM')
plt.plot(t_grs, WgDM, label='WgDM')
plt.legend()
plt.xlabel(r'$time\ [d]$')
plt.ylabel(r'$grass\ biomass\ [kgDM\ m^{-2}]$')
plt.show()