# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 13:22:30 2022

@author: User
"""

import numpy as np
from numpy.linalg import eig
import matplotlib
import matplotlib.pyplot as plt

#parameters
n = 16
t = 3
o = [] #init state
x = [[0, 0.9459343992499862, 0.9515683032664682, 0.956921398052074, 0.9619911870024044, 0.966775301446028, 0.9712715018653035, 0.9754776790520499, 0.9793918551974357, 0.9830121849155021, 0.9863369561997649, 0.9893645913123773, 0.9920936476053729, 0.9945228182735405, 0.9966509330385245, 0.9984769587637724, 1.0], 
     [0, 0.7674494679895795, 0.790565662833076, 0.8127673834379591, 0.8340085039953938, 0.8542447525107093, 0.873433813138727, 0.8915354241908093, 0.9085114715757365, 0.9243260774472821, 0.9389456838426778, 0.9523391311080167, 0.9644777309190192, 0.9753353337184365, 0.9848883904046668, 0.993116008119874, 1.0], 
     [0, 0.3532015826806202, 0.4084307383835277, 0.46335530572310163, 0.5176276589406708, 0.5709010362795464, 0.6228320408364755, 0.6730831302107859, 0.7213250759018542, 0.7672393735719645, 0.8105205856074655, 0.8508785978742284, 0.8880407731704374, 0.9217539846262565, 0.9517865131802361, 0.9779297942697976, 1.0]]
y = []

for i in range(t):
    o.append([])
    x.append([])
    y.append([])
for i in range(t):
    o[i].append(0)
    y[i].append(0)
    for j in range(n):
        o[i].append(0)
        y[i].append(2*(j+1))

#plot
fig= plt.figure()

plt.plot(o[0], y[0], marker='o', linestyle='', color='black')
plt.plot(o[0], y[0], linestyle='-', color='black')
for i in range(t):
    plt.plot(x[i], y[i], marker='o', linestyle='', color=(0.13*(i+3), 0.13*(i+3), 0.13*(i+3)))
    plt.plot(x[i], y[i], linestyle='-', color=(0.13*(i+3), 0.13*(i+3), 0.13*(i+3)), label='ki = {}'.format(10*5**i))
plt.xlim(-3, 3)
plt.title('Change ki of seismic Isolator', fontdict={'family': 'Times New Roman', 'color' : 'black', 'weight': 'bold','size': 28})
plt.legend(loc='upper left', fontsize=14)
