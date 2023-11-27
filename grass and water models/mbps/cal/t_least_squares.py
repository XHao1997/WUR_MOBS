# -*- coding: utf-8 -*-
"""
FTE34806 - Modelling of Biobased Production Systems
MSc Biosystems Engineering, WUR
@authors:   Daniel Reyes Lastiri, Stefan Maranus,
            Rachel van Ooteghem, Tim Hoogstad

Tutorial for the use of the least_squares method
Exercises 1
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import least_squares
import numpy.linalg as LA

plt.style.use('ggplot')

# -- EXERCISE 1.1 --
# Simulation time array
# TODO: define an array for a period that matches the period of measurements
# Notice that the analytical solution is based on the running time. Therefore,
# it's better to start the simulation from t0=0.
t_sim = np.linspace(0, 49, 50)  # [d]
print(t_sim)

# Initial condition
# TODO: Based on the data below, propose a sensible value for the initial mass
m0 = 0.156           # [kgDM m-2] initial mass

# Organic matter (assumed equal to DM) measured in Wageningen 1995 [gDM m-2]
# Groot and Lantinga (2004)
t_data = np.array([107, 114, 122, 129, 136, 142, 149, 156])
m_data = np.array([0.156, 0.198, 0.333, 0.414, 0.510, 0.640, 0.663, 0.774])
# TODO: this file uses the analytical solution for logistic growth.
# Adjust t_data so that it matches t_sim with t0=0.
t_data = t_data-107


# Define a function to simulate the model output of interest (m)
# as a function of the estimated parameter array 'p'
def fcn_y(p, t_sim):
    # Model parameters (improved iteratively)
    r, K = p[0], p[1]   # [d-1], [kgDM m-2] model parameters
    # Model output (analytical solution of logistic growth model)
    # TODO: define 'm' based on the analytical solution for logistic growth,
    # using t_sim.
    m = K / (1+((K-m0)/m0)*np.exp(-r*t_sim))  # [kgDM m-2]
    return m

# Define a function to calculate the residuals: e(k|p) = z(k)-y(k|p)
# Notice that m_k must be interpolated for measurement instants t_data
def fcn_residuals(p):
    # TODO: calculate m from the model output function 'fcn_y' defined above
    m = fcn_y(p, t_sim)
    # TODO: create an interpolation function using Scipy interp1d,
    # based on the simulation time and mass arrays
    f_interp = interp1d(t_data, m_data)
    # TODO: call the interpolation function for the measurement instants
    m_k = f_interp(t_sim)
    # TODO: Calculate the residuals (err)
    err = m - m_k
    return err
    
# Model calibration: least_squares
# TODO: Define an array for the initial guess of r and K (mind the order)
p0 = np.array([0.5, 0.5])                   # Initial parameter guess
# TODO: Call the Scipy method least_squares
y_lsq = least_squares(fcn_residuals, p0)              # Minimize sum [ e(k|p) ]^2
print(y_lsq)
# Retrieve the calibration results (parameter estimates)
# TODO: Once the code runs and you obtain y_lsq (a dictionary),
# look into y_lsq and identify its elements. Uncomment the line below
# and retrieve the parameter estimates.
p_hat = y_lsq['x']

# Simulate the model with initial guess (p0) and estimated parameters (p_hat)
# TODO: define variables m_hat0 (mass from initial parameter guess),
# and m_hat (mass from estimated parameters)
ini = fcn_y(p0, t_sim)
estimated = fcn_y(p_hat, t_sim)
# Plot results
# TODO: Make a plot for the growth of m_hat0 (dashed line),
# m_hat (continuous line), and mdata (no line, markers)
plt.figure(figsize=(16, 8))
plt.plot(t_sim, ini, label=r'm_hat0', linestyle='--')
plt.plot(t_sim, estimated, label=r'm_hat', linestyle='-')
plt.plot(t_data, m_data, label=r'mdata', marker='x', linestyle = '')
plt.legend()
plt.show()
# -- EXERCISE 1.2 --
# Calibration accuracy

# Jacobian matrix
print('')
print('-' * 40)
print('Accuracy of estimates')
print('-' * 40)
np.set_printoptions(formatter={'float_kind': '{:.3E}'.format})
print('Parameter estimates \n {} \n'.format(y_lsq['x']))
# Sensitivity matrix (Jacobian, J) (returned as jac)
# TODO: Retrieve the sensitivity matrix (Jacobian) J from y_ls.
J = y_lsq['jac']
    # Residuals (returned as fun)
err = y_lsq['fun']
# Calculated variance of residuals
N , p = J.shape[0], J.shape[1]
varres = 1/(N-p) * np.dot(err.T,err)
# Covariance matrix of parameter estimates
covp = varres * LA.inv(np.dot(J.T, J))
print('Covariance matrix of parameter estimates \n {} \n'.format(covp))
# Standard deviations of parameter estimates
sdp = np.sqrt(np.diag(covp))
print('Standard errors of parameter estimates \n {} \n'.format(sdp))
# Correlation coefficients of parameter estimates
ccp = np.empty_like(covp)
for i,sdi in enumerate(sdp):
    for j,sdj in enumerate(sdp):
        ccp[i,j] = covp[i,j]/(sdi*sdj)
print('Correlation coefficients of parameter estimates \n {} \n'.format(ccp))
# References

# Groot, J. C., & Lantinga, E. A. (2004). An object-oriented model of the
#  morphological development and digestibility of perennial ryegrass.
# Ecological Modelling, 177(3-4), 297-312.

# Bouman, B.A.M., Schapendonk, A.H.C.M., Stol, W., van Kralingen, D.W.G.
#  (1996) Description of the growth model LINGRA as implemented in CGMS.
#  Quantitative Approaches in Systems Analysis no. 7
#  Fig. 3.4, p. 35
