# -*- coding: utf-8 -*-
"""
FTE34806 - Modelling of Biobased Production Systems
MSc Biosystems Engineering, WUR
@authors:   -- Fill in your team names --

Tutorial: Soil water model analysis.
1. Slope of saturation vapour pressure.
2. Reference evapotranspiration.
"""
# TODO: import the required packages
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# TODO: specify the matplotlib style


# Measured saturation vapour pressure
# TODO: define arrays for the measured data T and Pvs.
# Specify the units in comments
T_data = np.array([10, 20, 30, 40])
Pvs_data = np.array([[12.28, 23.39, 42.46, 73.84]])
tsim = np.arange(T_data[0],T_data[-1],0.001)
# Air temperature [K]
# TODO: define an array for sesnible values of T
T = np.arange(1, 42, 1)
tspan = (1, 42)

# Model parameteres and variables
alpha = 1.291  # [mm MJ-1 (m-2)] Priestley-Taylor parameter
gamma = 0.68  # [mbar °C-1] Psychrometric constant
Irr_gl = 18.0  # [MJ m-2 d-2] Global irradiance
alb = 0.23  # [-] albedo (crop)
Rn = 0.408 * Irr_gl * 1 - (alb)  # [MJ m-2 d-1] Net radiation


# Model equations
# TODO: Define variables for the model
# Exercise 1. Pvs, Delta
# Exercise 2. ET0
def fcn_vap_pressure(t, y):
    dydt = 5304 * np.exp(21.3 - (5304 / t)) / (np.power(t,2))
    print(np.exp(21.3 - (5304 / t)))
    return dydt


def fuc_ET0(delta):
    return alpha * Rn * delta / (delta + gamma)


y0 = np.array([10])
sol = solve_ivp(fcn_vap_pressure, tspan, y0, method='RK45',t_eval=tsim,rtol=1E-8, atol=1E-8)
plt.plot(sol.t.ravel(), sol.y.ravel())

plt.show()
print(sol)

# Relative error
# TODO: Calculate the average relative error
# Tip: The numpy functions np.isin() or np.where() can help you retrieve the
# modelled values for Pvs at the corresponding value for T_data.


# Figures
# TODO: Make the plots
# Exercise 1. Pvs vs. T and Delta vs. T,
# Exercise 2. ET0 vs. T
