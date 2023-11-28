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
from scipy.optimize import least_squares

from mbps.models.log_growth import LogisticGrowth
from mbps.functions.calibration import fcn_residuals, fcn_accuracy
from mbps.functions.uncertainty import fcn_plot_uncertainty

plt.style.use('ggplot')

# Random number generator. A seed is specified to allow for reproducibility.
rng = np.random.default_rng(seed=12)

# -- Calibration --
# Grass data, Wageningen 1984 (Bouman et al. 1996)
# Cummulative yield [kgDM m-2]
tdata = np.array([87, 121, 155, 189, 217, 246, 273, 304])
mdata = np.array([0.05, 0.21, 0.54, 0.88, 0.99, 1.02, 1.04, 1.13])

# Simulation time array
tsim = np.linspace(0, 365, 365 + 1)  # [d]
tspan = (tsim[0], tsim[-1])

# Initialize and run reference object
dt = 1.0  # [d] time step size
x0 = {'m': 0.033}  # [gDM m-2] initial conditions
p = {'r': 0.01, 'K': 1.0}  # [d-1], [kgDM m-2] model parameters (ref. values)
lg = LogisticGrowth(tsim, dt, x0, p)
y = lg.run(tspan)


# Define function to simulate model as a function of estimated array 'p0'.
def fcn_y(p0):
    # Reset initial conditions
    lg.x0 = x0.copy()
    # Reassign parameters from array to object
    lg.p['r'] = p0[0]
    lg.p['K'] = p0[1]
    # Simulate the model
    y = lg.run(tspan)
    # Retrieve result (model output of interest)
    m = y['m']
    return m


def fcn_analytical_sol(K, r, t=tsim, m0=x0['m']):
    return m0 * K / (m0 + np.exp(-r * t) * (K - m0))


# Run calibration function
p0 = np.array([p['r'], p['K']])  # Initial guess
y_ls = least_squares(fcn_residuals, p0,
                     args=(fcn_y, lg.t, tdata, mdata),
                     kwargs={'plot_progress': True})

# Calibration accuracy
y_calib_acc = fcn_accuracy(y_ls)

# Simulate model with initial and estimated parameters
p_hat_arr = y_ls['x']
m_hat = fcn_y(p_hat_arr)

# Plot calibration results
plt.figure(1)
plt.plot(lg.t, m_hat, label=r'$\hat{m}$')
plt.plot(tdata, mdata, label=r'$m_{data}$', linestyle='None', marker='o')
plt.xlabel('time ' + r'$[d]$')
plt.ylabel('cummulative mass ' + r'$[kgDM\ m^{-2}]$')
plt.legend()

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
    m0 = x0['m']
    r = 0.086
    K = rng.normal(1.404, 0.070)
    # TODO: define the growth model function
    m = fcn_analytical_sol(K,r)
    # Store current simulation 'j' in the corresponding array column.
    m_arr[:, j] = m

# Plot results
# TODO: Plot the confidence intervals using 'fcn_plot_uncertainty'
plt.subplots()
plt.plot(tsim, m_arr[:, :12])
plt.xlabel(r'$time\ [d]$')
plt.ylabel('biomass ' + r'$[kgDM\ m^{-2}]$')
# plt.show()

plt.subplots()
ax2 = plt.gca()
ax2 = fcn_plot_uncertainty(ax2, tsim, m_arr, ci=[0.50, 0.68, 0.95])
plt.xlabel(r'$time\ [d]$')
plt.ylabel('biomass ' + r'$[kgDM\ m^{-2}]$')
# plt.show()


# plt.show()

plt.show()
# References

# Bouman, B.A.M., Schapendonk, A.H.C.M., Stol, W., van Kralingen, D.W.G.
#  (1996) Description of the growth model LINGRA as implemented in CGMS.
#  Quantitative Approaches in Systems Analysis no. 7
#  Fig. 3.4, p. 35
