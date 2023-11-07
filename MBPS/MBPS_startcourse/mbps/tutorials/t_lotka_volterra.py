#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FTE34806 - Modelling of Biobased Production Systems
MSc Biosystems Engineering, WUR
@authors:   Daniel Reyes Lastiri, Stefan Maranus,
            Rachel van Ooteghem, Tim Hoogstad

Tutorial for the Lotka-Volterra model

    dx1/dt = p1*x1 - p2*x1*x2
    dx2/dt = p3*x1*x2 - p4*x2
"""
# Import Python packages
import numpy as np
import matplotlib.pyplot as plt
# IMport our own functions
from mbps.models.lotka_volterra import LotkaVolterra

# Simulation time array
# FIXME: Define an array ranging from 0 to 365 [d],
# with step size 1.0
step_size = 1
time_range = 365.
tsim = np.linspace(0.0, time_range, num= int(time_range/step_size+1))

# Initialize object
# FIXME: Define the following variables
# - model integration step size dt = 7[d],
# - dictionary of initial conditions for prey = 50 and pred = 50 [#],
# - dictionary of model parameters (assume 1 [month] = 30 [d])
#   p1=1 [month-1], p2=0.02 [pred-1 month-1],
#   p3=0.01 [prey-1 month-1], p4=1 [month-1]
dt = 1
x0 = {'prey': 50, 'pred': 50}
p = {'p1': 1/30, 'p2': 0.02/30, 'p3': 0.01/30, 'p4': 1/30}
# FIXME: assign arguments to initialize the object.
# See the documentation of LotkaVolterra for help.
population = LotkaVolterra(tsim, dt, x0, p)

# Run model
tspan = (tsim[0], tsim[-1])
y = population.run(tspan)

# Retrieve results
# FIXME: Retrieve the variables from the dictionary of results 'y'.
# See the file 'lotka_volterra.py'
# to find the names returned by the function 'output'
t = y['t']
prey = y['prey']
pred = y['pred']
t2 = y['t2']
prey2 = y['prey2']
pred2 = y['pred2']


# Plot results
# FIXME: Add labels to the axes.
# For math format, use: r'$ text goes here $'
# To show # in the text, use \#
plt.style.use('ggplot')
plt.figure(1)
plt.plot(t, prey, label='Preys')
plt.plot(t, pred, label='Preds')
plt.legend()
plt.show()
plt.figure(2)
plt.plot(t2, prey2, label='Preys')
plt.plot(t2, pred2, label='Preds')
plt.legend()
plt.show()
# plt.???
