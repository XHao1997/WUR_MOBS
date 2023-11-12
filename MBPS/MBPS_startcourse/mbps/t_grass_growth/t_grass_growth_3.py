# -*- coding: utf-8 -*-
"""
FTE34806 - Modelling of Biobased Production Systems
MSc Biosystems Engineering, WUR
@authors:   -- Fill in your team names --

Tutorial: Grass growth model analysis.
3. Temperature index
"""
# TODO: Import the required packages 
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.integrate import quad

T_max = 42
T_min = 0
T_opt = 20


# TODO: Define the values for the TI parameters
def fuc_Tl(T=0, z=1.33, T_max=42, T_min=0, T_opt=20):
    Tl = np.power(
        (T_max - T) / (T_max - T_opt) * np.power((T - T_min) / (T_opt - T_min), ((T_opt - T_min) / (T_max - T_opt))), z)
    return Tl


# TODO: Define a sensible array for values of T
T_range = np.linspace(0, 42, 43)

# TODO: (Optional) Define support variables DTmin, DTmax, DTa, DTb
# Temperature index: TI = ( (DTmax/DTa) * (DTmin/DTb)**(DTb/DTa) )**z
T = T_range
DTmax = T_max - T
DTmin = T - T_min
DTa = T_max - T_opt
DTb = T_opt - T_min
# TODO: Define TI
def fuc_Tl2(DTmax, DTmin, DTa, DTb,z):
    return ((DTmax/DTa) * (DTmin/DTb)**(DTb/DTa) )**z

# TODO: Make a plot for TI vs T
plt.plot(T_range, fuc_Tl(T=T_range))
# plt.show()
plt.plot(T_range, fuc_Tl2(T=T_range),linestyle='dashed')
plt.show()