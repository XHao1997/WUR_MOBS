import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from mbps.functions.calibration import fcn_residuals, fcn_accuracy
from mbps.models.grass_sol import Grass
from mbps.models.water_sol import Water

plt.style.use('ggplot')

# Simulation time
tsim = np.linspace(0, 365, 365 + 1)  # [d]

# Weather data (disturbances shared across models)
t_ini = '19850101'
t_end = '19860101'
t_weather = np.linspace(1, 365, 365 + 1)
data_weather = pd.read_csv(
    '../../data/etmgeg_1984_1985_DeBilt.csv',  # .. to move up one directory from current directory
    skipinitialspace=True,  # ignore spaces after comma separator
    header=10,  # row with column names, 0-indexed, excluding spaces
    usecols=['YYYYMMDD', 'TG', 'Q', 'RH'],  # columns to use
    index_col=0,  # column with row names from used columns, 0-indexed
)

# -- Grass sub-model --
# Step size
dt_grs = 1  # [d]

# Initial conditions
# TODO: Specify suitable initial conditions for the grass sub-model
x0_grs = {'Ws': 5.2e-2, 'Wg': 1.20e-2}  # [kgC m-2]
# Model parameters (as provided by Mohtar et al. 1997 p.1492-1493)
p_grs = {'a': 40.0, 'alpha': 4e-9, 'beta': 0.025, 'k': 0.5, 'm': 0.1, 'M': 0.02, 'mu_m': 0.5, 'P0': 0.432, 'phi': 0.9,
         'Tmin': 0.0, 'Topt': 20.0, 'Tmax': 42.0, 'Y': 0.75, 'z': 1.33}
# p0 = [p_grs['alpha'], p_wtr['kcrop'], p_grs['Y'], p_grs['Tmin'], p_grs['beta']]

# Model parameters adjusted manually to obtain growth
p_grs['alpha'] = 9e-9
p_grs['beta'] = 0.01
p_grs['Tmin'] = 4
p_grs['M'] = 0.04
# TODO: Adjust a few parameters to obtain growth.
# Satrt by using the modifications from Case 1.
# If needed, adjust further those or additional parameters

# Disturbances
# PAR [J m-2 d-1], environment temperature [°C], leaf area index [-]
T = data_weather.loc[t_ini:t_end, 'TG'].values  # [0.1 °C] Env. temperature
I_gl = data_weather.loc[t_ini:t_end, 'Q'].values  # [J cm-2 d-1] Global irr.

T = T / 10  # [0.1 °C] to [°C] Environment temperature
I0 = 0.45 * I_gl * 1E4 / dt_grs  # [J cm-2 d-1] to [J m-2 d-1] Global irr. to PAR

d_grs = {'T': np.array([t_weather, T]).T,
         'I0': np.array([t_weather, I0]).T,
         'WAI': np.array([t_weather, np.full((t_weather.size,), 1.0)]).T
         }

# Initialize module
grass = Grass(tsim, dt_grs, x0_grs, p_grs)

# -- Water sub-model --
dt_wtr = 1  # [d]

# Initial conditions
# TODO: Specify suitable initial conditions for the soil water sub-model
x0_wtr = {'L1': 0.36 * 150, 'L2': 0.32 * 250, 'L3': 0.24 * 600, 'DSD': 1}  # 3*[mm], [d]
# x0_wtr = {'L1': 54, 'L2': 80, 'L3': 144, 'DSD': 1}  # 3*[mm], [d]

# Castellaro et al. 2009, and assumed values for soil types and layers
p_wtr = {'alpha': 1.29E-6,  # [mm J-1] Priestley-Taylor parameter
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
         'S': 10,  # [mm d-1] parameter of precipitation retention
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

# Grass data. (Organic matter assumed equal to DM) [gDM m-2]
# Groot and Lantinga (2004)
t_data = np.array([107, 114, 122, 129, 136, 142, 149, 156])
m_data = np.array([156., 198., 333., 414., 510., 640., 663., 774.])
m_data = m_data / 1E3
# Grass data. (Organic matter assumed equal to DM) [gDM m-2]
# Groot and Lantinga (2004)
t_data = np.array([112.78, 118.19, 125.63, 131.72, 137.81])
m_data = np.array([1388.88, 1944.44, 2453.70, 3333.33, 3888.88])
m_data = m_data / 1E4


def fnc_y(p0):
    # Reset initial conditions
    grass.x0 = x0_grs.copy()
    water.x0 = x0_wtr.copy()

    # Model parameters
    grass.p['alpha'] = p0[0]
    water.p['WAIc'] = p0[1]
    # grass.p['phi'] = p0[2]
    # grass.p['Y'] = p0[2]
    grass.p['Tmin'] = p0[2]
    # grass.p['beta'] = p0[3]

    # grass.p['beta'] = p0[3]
    # water.p['D3'] = p0[5]

    # Run simulation
    # Initial disturbance
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

    return grass.y['Wg'] / 0.4


p0 = [p_grs['alpha'], p_wtr['WAIc'], p_grs['Tmin']]
# p0 = [p_grs['alpha'], p_grs['beta']]

bnds = ((4e-10, 0.5,  0, 0.01), (4e-2, 1,  10, 0.05))
bnds = ((4e-10, 0.5,  0), (4e-2, 1,  10))

# bnds = ((4e-10,  0.01), (4e-8, 0.05))

y_ls = least_squares(fcn_residuals, p0, bounds=bnds, args=(fnc_y, grass.t, t_data, m_data),
                     kwargs={'plot_progress': True})
# Calibration accuracy
y_calib_acc = fcn_accuracy(y_ls)

# Run calibrated simulation
# TODO: Retrieve the parameter estimates from
# the output of the least_squares function
p_hat = y_ls['x']

# TODO: Run the model output simulation function with the estimated parameters
# (this is the calibrated model output)
WgDM_hat = fnc_y(p_hat)

# -- Plot results --
# TODO: Make one figure comparing the calibrated model against
# the measured data
plt.figure('Calibration alpha & mu_m')
plt.plot(grass.t, WgDM_hat, label='Calibrated model')
plt.plot(t_data, m_data, label='Measured data')
plt.legend()
plt.show()
