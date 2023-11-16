# -*- coding: utf-8 -*-
"""
FTE34806 - Modelling of Biobased Production Systems
MSc Biosystems Engineering, WUR
@authors:   -- write your team names here --

Tutorial for the sensitivity analysis of the logistic growth model
"""
import numpy as np
import matplotlib.pyplot as plt

from mbps.models.log_growth import LogisticGrowth

plt.style.use('ggplot')

# Simulation time array
tsim = np.linspace(0, 10, 100+1)    # [d]
tspan = (tsim[0], tsim[-1])         # [d]

# Initialize and run reference object
dt = 0.1                        # [d] time-step size
x0 = {'m':1.0}                  # [gDM m-2] initial conditions
p_ref = {'r':1.2,'K':100.0}     # [d-1], [gDM m-2] model parameters (ref. values)
lg = LogisticGrowth(tsim,dt,x0,p_ref)   # Initialize obgect
y = lg.run(tspan)                       # Run object

# One-at-a-time parameter changes, simulation, S, and NS
# dm/dr-
p_rmin = p_ref.copy()                           # define p with r-
p_rmin['r'] = 0.95*p_ref['r']                   # update value of r-
lg_rmin = LogisticGrowth(tsim, dt, x0, p_rmin)  # initialize object
y_rmin = lg_rmin.run(tspan)                     # run object
S_rmin = (y_rmin['m']-y['m'])/(p_rmin['r']-p_ref['r'])  # sensitivity

p_rmax = p_ref.copy()                           # define p with r-
p_rmax['r'] = 1.05*p_ref['r']                   # update value of r-
lg_rmax = LogisticGrowth(tsim, dt, x0, p_rmax)  # initialize object
y_rmax = lg_rmax.run(tspan)                     # run object
S_rmax = (y_rmax['m']-y['m'])/(p_rmax['r']-p_ref['r'])  # sensitivity
NS_rmin = S_rmin*p_ref['r']/y['m']
NS_rmax = S_rmax*p_ref['r']/y['m']
NS_rmin_norm = S_rmin*p_ref['r']/np.mean(y['m'])
NS_rmax_norm = S_rmax*p_ref['r']/np.mean(y['m'])




# TODO: Code the sensitivity S_rpls
# dm/dr+

# TODO: Code sensitivity S_Kmin
# dm/dK-

# TODO: Code sensitivity S_Kpls
# dm/dK+

# Plot results
# m with changes in r
# TODO: Make a plot m vs t changing r+/-5%
plt.subplots()
plt.plot(tsim,y_rmin['m'])
plt.plot(tsim,y_rmax['m'])
# m with changes in K
# TODO: Make a plot m vs t changing K+/-5%
plt.subplots()

p_kmax = p_ref.copy()                           # define p with r-
p_kmax['K'] = 1.05*p_ref['K']                   # update value of r-
lg_kmax = LogisticGrowth(tsim, dt, x0, p_kmax)  # initialize object
y_kmax = lg_kmax.run(tspan)                     # run object
S_kmax = (y_kmax['m']-y['m'])/(p_kmax['K']-p_ref['K'])  # sensitivity


p_kmin = p_ref.copy()                           # define p with r-
p_kmin['K'] = 0.95*p_ref['K']                   # update value of r-
lg_kmin = LogisticGrowth(tsim, dt, x0, p_kmin)  # initialize object
y_kmin = lg_kmin.run(tspan)                     # run object
S_kmin = (y_kmin['m']-y['m'])/(p_kmin['K']-p_ref['K'])  # sensitivity
NS_kmin = S_kmin*p_ref['K']/y['m']
NS_kmax = S_kmax*p_ref['K']/y['m']
NS_kmin_norm = S_kmin*p_ref['K']/np.mean(y['m'])
NS_kmax_norm = S_kmax*p_ref['K']/np.mean(y['m'])
plt.plot(tsim,y_kmin['m'])
plt.plot(tsim,y_kmax['m'])


# S on r
# TODO: Make a plot S vs t changing r+/-5%
plt.subplots()

plt.plot(tsim,S_rmax)
plt.plot(tsim,S_rmin)


# S on K
# TODO: Make a plot S vs t changing K+/-5%
plt.subplots()

plt.plot(tsim,S_kmin)
plt.plot(tsim,S_kmax)

# NS
# TODO: Make a plot NS vs. t, changing r & K +/- 5%
plt.subplots()
plt.plot(tsim,NS_kmin, label = "k-5%")
plt.plot(tsim,NS_kmax, label = "k+5%")
plt.plot(tsim,NS_rmin, label = "r-5%")
plt.plot(tsim,NS_rmax, label = "r+5%")
plt.legend()


plt.subplots()
plt.plot(tsim,NS_kmin_norm, label = "k_norm-5%")
plt.plot(tsim,NS_kmax_norm, label = "k_norm+5%")
plt.plot(tsim,NS_rmin_norm, label = "r_norm-5%")
plt.plot(tsim,NS_rmax_norm, label = "r_norm+5%")
plt.legend()
plt.show()
# plt.plot()

