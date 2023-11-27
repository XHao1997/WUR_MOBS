# -*- coding: utf-8 -*-
"""
FTE34806 - Modelling of Biobased Production Systems
MSc Biosystems Engineering, WUR
@authors:   --- Fill in your team names ---

Tutorial: Grass growth model analysis.
1. Light intensity over leaves.
"""
import numpy as np
import matplotlib.pyplot as plt


### 1. Light intensity over leaves
# TODO: Define a function f_Il
# which takes positional arguments l, k, m, I0
# and returns an array for Il
def f_Il(k, m, I0, l):
    Il = k / (1 - m) * I0 * np.exp(-k * l)
    return Il


# TODO: define an array with sensible values of leaf area index (l)
# Use the function np.linspace
l = np.linspace(0, 4, 1000)
k_0 = 0.5
m_0 = 0.1
I_0 = 100
# TODO: Code the figures for:
# 1) Il vs. l, with three test values for k
k = [0.5, 0.25, 0.75]
y_k = [f_Il(k[0], m_0, I_0, l), f_Il(k[1], m_0, I_0, l), f_Il(k[2], m_0, I_0, l)]

plt.figure(1)
# plt.style.use('ggplot')
for i in range(len(k)):

    plt.plot(l, y_k[i])
plt.show()

# 3) Il vs. l, with three test values for I0
plt.figure(2)
m = [0.5, 0.25, 0.75, 0.8]

for i in range(len(m)):
    plt.plot(l, f_Il(k_0, m[i], I_0, l))
plt.show()
# 2) Il vs. l, with three test values for m
plt.figure(3)
I0 = [50, 100, 150]
y_I0 = [f_Il(k_0, m_0, I0[0], l), f_Il(k_0, m_0, I0[1], l), f_Il(k_0, m_0, I0[2], l)]
for i in range(len(I0)):
    plt.plot(l, y_I0[i])
plt.show()
# plt.figure(2)
#
# plt.figure(3)
