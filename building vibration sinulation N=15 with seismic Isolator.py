# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 14:58:31 2022

@author: User
"""

import numpy as np
from numpy.linalg import eig
import matplotlib
import matplotlib.pyplot as plt

#parameters
t = 100 #precision
n = 16
m = [] #mass
k = [] #coefficient of restitution
for i in range(n-1):
    m.append(1+0.4*i)
    k.append(100*(i+1))
mi = 7 #mass of seismic Isolator
ki = 10 #coefficient of restitution of seismic Isolator
m.append(mi)
k.append(ki)

K = []
M = []
S = [] #symmetric, tridiagonal matrix MKM
S_a = [] #turn S to array
I = [] #identity matrix
P = [] #rotation matrix
R = [] #upper triangular matrix
Q = [] #orthogonal matrix
A = [] #eigen matrix
ws = [] #omega square
w = [] #omega
wsI = [] #ws[i]*I
E = [] #S-eigen value
X = [] #eigen vector
W = 0 #eigen value by using python's caculator
Wl = [] #W's list
V = 0 #eigen vector by using python's caculator
wse = [] # uncertainty of eigen value
Xe = [] # uncertainty of eigen vector

#init K
for i in range(n):
    K.append([])
    for j in range(n):
        if i == 0 and j == 0:
            K[i].append(k[i])
        elif j == i-1:
            K[i].append(-k[j])
        elif j == i+1:
            K[i].append(-k[i])
        elif j == i:
            K[i].append(k[j-1]+k[j])
        else:
            K[i].append(0)
            
#init M
for i in range(n):
    M.append([])
    for j in range(n):
        if j == i:
            M[i].append(m[i])
        else:
            M[i].append(0)
            
#make M^-0.5KM^-0.5
for i in range(n):
    S.append([])
    for j in range(n):
        S[i].append(((1/M[i][i])**0.5)*K[i][j]*((1/M[j][j])**0.5))
        
#matrix add
def Add(a, b, n):
    c = []
    t = 0
    for i in range(n):
        c.append([])
    for i in range(n):
        for j in range(n):
            t = a[i][j]+b[i][j]
            c[i].append(t)
            t = 0
    return c

#matrix minus
def Minus(a, b, n):
    c = []
    t = 0
    for i in range(n):
        c.append([])
    for i in range(n):
        for j in range(n):
            t = a[i][j]-b[i][j]
            c[i].append(t)
            t = 0
    return c

#matrix times
def Times(a, b, n):
    c = []
    t = 0
    for i in range(n):
        c.append([])
    for i in range(n):
        for j in range(n):
            for k in range(n):
                t = t+a[i][k]*b[k][j]
            c[i].append(t)
            t = 0
    return c
             
#matrix trans
def Trans(a, n):
    b = []
    for i in range(n):
        b.append([])
    for i in range(n):
        for j in range(n):
            b[i].append(a[j][i])
    return b

#for test
#S=[[3, 1, 0], [1, 3, 1], [0, 1, 3]]

#identity matrix
for i in range(n):
    I.append([])
    for j in range(n):
        if j == i:
            I[i].append(1)
        else:
            I[i].append(0)

#QR method
#init matrix
A = S
for i in range(t): 
    R = A
    Q = I
    #make R 
    for j in range(1, n):
        list.clear(P)
        for k in range(n): #make P
            P.append([])
            for h in range(n):
                if k < j-1 or k > j:
                    if h == k:
                        P[k].append(1)
                    else:
                        P[k].append(0)
                elif (h == j-1 and k == j-1) or (h == j and k == j):
                    P[k].append(R[j-1][j-1]/(R[j][j-1]**2+R[j-1][j-1]**2)**0.5)
                elif h == j and k == j-1:
                    P[k].append(R[j][j-1]/(R[j][j-1]**2+R[j-1][j-1]**2)**0.5)
                elif h == j-1 and k == j:
                    P[k].append(-R[j][j-1]/(R[j][j-1]**2+R[j-1][j-1]**2)**0.5)
                else:
                    P[k].append(0)
        R = Times(P, R, n)
        Q = Times(Q, Trans(P, n), n)
    A = Times(R, Q, n)
    list.clear(R)
    list.clear(Q)

#obtain omega square and omega
for i in range(n):
    ws.append([])
    for j in range(n):
        if j == i:
            ws[i].append(A[i][j])
        else:
            ws[i].append(0)
    
for i in range(n):
    w.append([])
    for j in range(n):
        if j == i:
            w[i].append(A[i][j]**0.5)
        else:
            w[i].append(0)

#obtain the eigenvector    
#assume that the first element of eigenvector = 1
for i in range(n):
    list.clear(wsI)
    list.clear(E)
    for j in range(n):
        wsI.append([])
        for k in range(n):
            if k == j:
                wsI[j].append(ws[n-i-1][n-i-1])
            else:
                wsI[j].append(0)
    E = Minus(S, wsI, n)
    X.append([])
    X[i].append(1)
    X[i].append(-E[0][0]/E[0][1])
    for j in range(1, n-1):
        X[i].append(-(E[j][j-1]*X[i][j-1]+E[j][j]*X[i][j])/E[j][j+1])
for i in range(n):
    for j in range(n):
        X[i][j] = X[i][j]/M[j][j]**0.5

#Python's eigenvector caculator
S_a = np.array(S)
W, V = eig(S)
for i in range(n):
    Wl.append([])
for i in range(n):
    for j in range(n):
        if j == i:
           Wl[i].append(W[n-j-1])
        else:
            Wl[i].append(0)

for i in range(n):
    for j in range(n):
        V[n-j-1][i] = V[n-j-1][i]/V[0][i]
for i in range(n):
    for j in range(n):
        V[j][i] = V[j][i]/M[j][j]**0.5
V = Trans(V, n)

#uncertainty of self-coding method and python's calculator
for i in range(n):
    wse.append([])
    Xe.append([])
for i in range(n):
    for j in range(n):
        if j == i:
            wse[i].append(100*np.abs(ws[i][j]-Wl[i][j])/np.abs(Wl[i][j]))
        else:
            wse[i].append(0)
        Xe[i].append(100*np.abs(X[i][j]-V[i][j])/np.abs(V[i][j]))

#visulize
#parameters
o = [] #init state
x = []
y = []

for i in range(n):
    o.append([])
    x.append([])
    y.append([])
for i in range(n):
    o[i].append(0)
    x[i].append(0)
    y[i].append(0)
    for j in range(n):
        o[i].append(0)
        x[i].append(X[i][n-j-1])
        y[i].append(2*(j+1))

#plot
fig= plt.figure()

plt.plot(o[0], y[0], marker='o', linestyle='', color='black')
plt.plot(o[0], y[0], linestyle='-', color='black')
plt.plot(x[0], y[0], marker='o', linestyle='', color='blue')
plt.plot(x[0], y[0], linestyle='-', color='blue')
plt.xlim(-3, 3)
plt.title('mi={}, ki={}'.format(mi, ki), fontdict={'family': 'Times New Roman', 'color' : 'black', 'weight': 'bold','size': 28})
