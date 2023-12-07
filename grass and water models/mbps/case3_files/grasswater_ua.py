# -*- coding: utf-8 -*-
"""
FTE34806 - Modelling of Biobased Production Systems
MSc Biosystems Engineering, WUR
@authors:   -- write your team names here --

Tutorial for the uncertainty analysis of the logistic growth model.
This tutorial covers first the calibration of the logistic growth model,
then the identification of the uncertainty in the parameter estimates,
and their propagation through the model.
"""
import numpy as np
import matplotlib.pyplot as plt
from mbps.functions.uncertainty import fcn_plot_uncertainty
import pandas as pd
from mbps.models.grass_sol import Grass
from mbps.models.water_sol import Water
import os

plt.style.use('ggplot')

# Random number generator. A seed is specified to allow for reproducibility.
rng = np.random.default_rng(seed=12)

# -- Calibration --
# Grass data, Wageningen 1984 (Bouman et al. 1996)
# Cummulative yield [kgDM m-2]
tdata = np.array([87, 121, 155, 189, 217, 246, 273, 304])
mdata = np.array([0.05, 0.21, 0.54, 0.88, 0.99, 1.02, 1.04, 1.13])

# Simulation time array
tsim = np.linspace(0, 365, int(365/5) + 1)  # [d]
tspan = (tsim[0], tsim[-1])


# -*- coding: utf-8 -*-
"""
FTE34806 - Modelling of Biobased Production Systems
MSc Biosystems Engineering, WUR
@authors:   Daniel Reyes Lastiri, Stefan Maranus,
            Rachel van Ooteghem, Tim Hoogstad

Evaluation of the grass & water model
"""


plt.style.use('ggplot')

# Simulation time
tsim = np.linspace(0, 365, 365 + 1)  # [d]

# Weather data (disturbances shared across models)
t_ini = '20160101'
t_end = '20161231'
t_weather = np.linspace(0, 365, 365 + 1)
# curent_dir = os.getcwd()
# data_path = curent_dir+'/data/etmgeg_260.csv'
# print(data_path)
data_weather = pd.read_csv(
    '../..//data/etmgeg_260.csv',  # .. to move up one directory from current directory
    skipinitialspace=True,  # ignore spaces after comma separator
    header=47 - 3,  # row with column names, 0-indexed, excluding spaces
    usecols=['YYYYMMDD', 'TG', 'Q', 'RH'],  # columns to use
    index_col=0,  # column with row names from used columns, 0-indexed
)

# Grass data. (Organic matter assumed equal to DM) [gDM m-2]
# Groot and Lantinga (2004)
t_data = np.array([107, 114, 122, 129, 136, 142, 149, 156])
m_data = np.array([156., 198., 333., 414., 510., 640., 663., 774.])
m_data = m_data / 1E3

# ---- Grass sub-model
# Step size
dt_grs = 1  # [d]

# Initial conditions
# TODO: Specify suitable initial conditions for the grass sub-model
x0_grs = {'Ws': 1e-2, 'Wg': 3e-2}  # [kgC m-2]

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
p_grs['alpha'] = 9E-09
p_grs['beta'] = 0.02
p_grs['Tmin'] = 4.5

# Disturbances
# PAR [J m-2 d-1], environment temperature [°C], leaf area index [-]
T = data_weather.loc[t_ini:t_end, 'TG'].values  # [0.1 °C] Env. temperature
I_gl = data_weather.loc[t_ini:t_end, 'Q'].values  # [J cm-2 d-1] Global irr.

T = T / 10  # [0.1 °C] to [°C] Environment temperature
I0 = 0.45 * I_gl * 1E4 / dt_grs  # [J cm-2 d-1] to [J m-2 d-1] Global irr. to PAR

d_grs = {'T': np.array([t_weather, T]).T,
         'I0': np.array([t_weather, I0]).T,
         }



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



# ---- Run simulation
# Initial disturbance
def fcn_ua(p):
    # Initialize module
    # Initialize module
    p_grs['alpha'],p_grs['Tmin']  = p

    grass = Grass(tsim, dt_grs, x0_grs, p_grs)
    water = Water(tsim, dt_wtr, x0_wtr, p_wtr)
    d_grs['WAI'] = np.array([[0, 1, 2, 3, 4], [1., ] * 5]).T
    # Iterator
    # (stop at second-to-last element, and store index in Fortran order)
    it = np.nditer(tsim[:-1], flags=['f_index'])
    for ti in it:
        # Index for current time instant
        idx = it.index
        # Integration span
        tspan = (tsim[idx], tsim[idx + 1])
        # Controlled inputs
        u_grs = {'f_Gr': 0, 'f_Hr': 0}  # [kgDM m-2 d-1]
        u_wtr = {'f_Irg': 0}  # [mm d-1]
        # Run grass model
        y_grs = grass.run(tspan, d_grs, u_grs)
        # Retrieve grass model outputs for water model
        d_wtr['LAI'] = np.array([y_grs['t'], y_grs['LAI']])
        # Run water model
        y_wtr = water.run(tspan, d_wtr, u_wtr)
        # Retrieve water model outputs for grass model
        d_grs['WAI'] = np.array([y_wtr['t'], y_wtr['WAI']])

    # Retrieve simulation results
    t_grs, t_wtr = grass.t, water.t
    WsDM, WgDM, LAI = grass.y['Ws'] / 0.4, grass.y['Wg'] / 0.4, grass.y['LAI']
    L1, L2, L3 = water.y['L1'], water.y['L2'], water.y['L3'],
    WAI = water.y['WAI']
    return WgDM


# -- Uncertainty Analysis --
# Monte Carlo simulations
n_sim = 1000  # number of simulations
# Initialize array of outputs, shape (len(tsim), len(n_sim))
m_arr = np.full((tsim.size, n_sim), np.nan)
# Run simulations
for j in range(n_sim):
    # TODO: Fill in the Monte Carlo simulations
    # TODO: Specify the parameter values or sample from a normal distribution.
    # Sample random parameters
    T_min = rng.normal(4.5, 0)
    alpha = rng.normal(9e-9, 9e-9*5e-2)
    m = fcn_ua([alpha,T_min])
    # Store current simulation 'j' in the corresponding array column.
    m_arr[:, j] = m

# Plot results
# TODO: Plot the confidence intervals using 'fcn_plot_uncertainty'
plt.subplots()
plt.plot(tsim, m_arr[:, :1000], linewidth=0.5)
plt.xlabel(r'$time\ [d]$')
plt.ylabel('biomass ' + r'$[kgDM\ m^{-2}]$')
plt.show()

plt.subplots()
ax2 = plt.gca()
ax2 = fcn_plot_uncertainty(ax2, tsim, m_arr, ci=[0.50, 0.68, 0.95])
plt.xlabel(r'$time\ [d]$')
plt.ylabel('biomass ' + r'$[kgDM\ m^{-2}]$')
plt.show()


# plt.show()

# plt.show()
# References

# Bouman, B.A.M., Schapendonk, A.H.C.M., Stol, W., van Kralingen, D.W.G.
#  (1996) Description of the growth model LINGRA as implemented in CGMS.
#  Quantitative Approaches in Systems Analysis no. 7
#  Fig. 3.4, p. 35
