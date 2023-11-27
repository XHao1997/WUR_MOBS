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

# TODO: specify the matplotlib style
plt.style.use('ggplot')

# Measured saturation vapour pressure
# TODO: define arrays for the measured data T and Pvs.
# Specify the units in comments
T_data = np.array([10, 20, 30, 40])
Pvs_data = np.array([12.28, 23.39, 42.46, 73.84])

# Air temperature [K]
# TODO: define an array for sesnible values of T
T = np.linspace(10, 40, 50)

# Model parameteres and variables
alpha = 1.291   # [mm MJ-1 (m-2)] Priestley-Taylor parameter
gamma = 0.68    # [mbar °C-1] Psychrometric constant
Irr_gl = 18.0   # [MJ m-2 d-2] Global irradiance
alb = 0.23      # [-] albedo (crop)
Rn = 0.408*Irr_gl*1-(alb)  # [MJ m-2 d-1] Net radiation

# Model equations
# TODO: Define variables for the model
# Exercise 1. Pvs, Delta
def pvs(T):
    pvs = np.exp(21.3-5304/(T+273.15)) - 2.3
    dpvs = (5304/(T+273.15)**2)*np.exp(21.3-5304/(T+273.15))
    return pvs, dpvs
# Exercise 2. ET0
def ET(T):
    p,delta = pvs(T)
    # H0 = (1 - lambda) * R / 58.3
    ET0 = alpha * Rn * (delta/(delta + gamma))
    return ET0
# P_model = pvs(T_data)
P_test, dP_test = pvs(T)
ET0 = ET(T)
# error = (P_model-Pvs_data)
# error = sum(error)/len(Pvs_data)
# Relative error
# TODO: Calculate the average relative error
# Tip: The numpy functions np.isin() or np.where() can help you retrieve the
# modelled values for Pvs at the corresponding value for T_data.
# print(error)

# Figures
# TODO: Make the plots
# Exercise 1. Pvs vs. T and Delta vs. T,
plt.figure(1)
plt.plot(T, P_test, label='P')
plt.plot( T, dP_test, label='delta_P', marker='', linestyle='--')
plt.plot(T_data, Pvs_data, label='Pvs_data', marker='x', linestyle='')
plt.legend()
# Exercise 2. ET0 vs. T
plt.figure(2)

plt.plot( T, ET0, label='ET0 vs. T', marker='', linestyle='-')
plt.legend()
plt.show()