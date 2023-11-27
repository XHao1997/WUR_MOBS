# -*- coding: utf-8 -*-
"""
FTE34806 - Modelling of Biobased Production Systems
MSc Biosystems Engineering, WUR
@authors:   Daniel Reyes Lastiri, Stefan Maranus,
            Rachel van Ooteghem, Tim Hoogstad

Tutorial for the calibration of the logistic growth model

Exercise 2.1. Calibration based on the object-oriented Module.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

from mbps.models.log_growth import LogisticGrowth
from mbps.functions.calibration import fcn_residuals, fcn_accuracy

plt.style.use('ggplot')

# Simulation time array
tsim = np.linspace(0.0, 365.0, 365+1) # [d]
tspan = (tsim[0], tsim[-1])

# Initialize reference object
dt = 1.0                # [d] time-step size
x0 = {'m':0.01}         # [kgDM m-2] initial conditions
p = {'r':0.01,'K':1.0}  # [d-1], [kgDM m-2] model parameters (initial guess)
lg = LogisticGrowth(tsim, dt, x0, p)

# Grass data, Wageningen 1984 (Bouman et al. 1996)
tdata = np.array([87, 121, 155, 189, 217, 246, 273, 304])
mdata = np.array([0.05, 0.21, 0.54, 0.88, 0.99, 1.02, 1.04, 1.13])

# Define function to simulate model as a function of estimated array 'p0'.
def fcn_y(p0):
    # Reset initial conditions
    lg.x0 = x0.copy()
    # Reassign parameters from array to object
    # TODO: assign first element of array p0 to r, and second element to K
    lg.p['r'] = p0[0]
    lg.p['K'] = p0[1]
    # Simulate the model
    # TODO: call the method 'run' from object lg
    y = lg.run(tspan)
    # Retrieve result

    # TODO: retrieve from dictionary y, the model output of interest
    m = y['m_rk']
    return m

# Run calibration function
# TODO: Look in the documentation of 'fcn_residuals'
# to find the missing required arguments, and fill them in.
p0 = np.array([p['r'], p['K']]) # Initial guess
y_lsq = least_squares(fcn_residuals, p0,
                     args=(fcn_y, lg.t, tdata, mdata),
                     kwargs={'plot_progress':True})

# Calibration accuracy
y_acc = fcn_accuracy(y_lsq)

# Simulate model with estimated parameters
# TODO: Run the model output function (fcn_y)
# to generate the calibrated model output (estimated mass mhat)
p_hat = y_lsq['x']
mhat = fcn_y(p_hat, t_sim)

# Plot results
# TODO: Make a figure with mass data (markers, no line), and
# the calibrated model output m_hat.
plt.figure(figsize=(16, 8))
plt.plot(t_sim, mhat, label=r'm_hat', linestyle='-')
plt.plot(t_data, m_data, label=r'mdata', marker='x', linestyle = '')
plt.legend()
#plt.show()
plt.figure(1)

