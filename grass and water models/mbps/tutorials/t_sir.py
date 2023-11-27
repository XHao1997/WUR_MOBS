# -*- coding: utf-8 -*-
"""
FTE34806 - Modelling of Biobased Production Systems
MSc Biosystems Engineering, WUR
@authors:   -- add your team names here --

Tutorial for the disease model (SIR)
"""
import numpy as np
import matplotlib.pyplot as plt
# IMport our own functions
from mbps.models.sir import SIR
# with step size 1.0
step_size = 1
time_range = 365.
tsim = np.linspace(0.0, time_range, num= int(time_range/step_size+1))

# TODO: Create the script to simulate the SIR model,
# and analyse parameter beta.
dt = 0.1  # [d] time-step size
x0 = {'s': 0.99,'i':0.01, 'r':0.0}  # [gDM m-2] initial conditions
p = {'beta': 0.5, 'gama': 0.02}  # [d-1], [gDM m-2] model parameters
lg = SIR(tsim, dt, x0, p)

# Run model
tspan = (tsim[0], tsim[-1])
y = lg.run(tspan)

# Plot results
plt.figure(1)
plt.plot(y['t'], y['s'])
plt.plot(y['t'], y['i'])
plt.plot(y['t'], y['r'])

plt.show()


