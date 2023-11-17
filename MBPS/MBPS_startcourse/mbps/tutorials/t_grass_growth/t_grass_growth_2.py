# -*- coding: utf-8 -*-
"""
FTE34806 - Modelling of Biobased Production Systems
MSc Biosystems Engineering, WUR
@authors:   --- Fill in your team names ---
Tutorial: Grass growth model analysis.
2. Irradiance trhoughout day
"""
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.integrate import quad

# TODO: define a function fcn_I0
# with positional arguments tau, h and J
# that returns I0 [W m-2]
# Remember to apply any unit conversion in the function, or its arguments.
def fcn_I0(tau,h=15.0,J=1517.0):
    I0 = 2*J/h*np.sin(math.pi*tau/h)**2
    return I0

# TODO: Define an array for 24 hr, with time step of 1 min.
# Use the function linspace.
tau = np.linspace(0, 24, 24*60+1)

# Reference values
# A nice and fresh liberation day at Wageningen (05/05/2022)
# TODO: Enjoy a Bevrijdingsfestival in Wageningen
h = 15.0    # [hr], (Time and Date AS, 2022)
J = 1517.0  # [J cm-2], (KNMI, 2022)

# TODO: calculate the area below the function curve 'fcn_I0'
# to determine the daily radiation R.
# Use the function 'quad'
# https://docs.scipy.org/doc/scipy/tutorial/integrate.html
# Verify whether J = R.

R = quad(fcn_I0,0,15)
print(R)
# TODO: plot I0 vs tau for h = 8, 15, and 24 [hr]
plt.style.use('ggplot')
plt.figure(1)
plt.plot(tau,fcn_I0(tau))
plt.show()

### References
# [1] Time and Date AS (accesed Sep 2022),
# https://www.timeanddate.com/sun/netherlands/wageningen?month=5&year=2022

# [2] KNMI (accesed Sep 2022), Weather station De Bilt 260,
# https://www.knmi.nl/nederland-nu/klimatologie/daggegevens