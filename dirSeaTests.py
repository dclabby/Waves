#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 16:49:48 2020

@author: dclabby
"""
import numpy as np
import matplotlib.pyplot as plt
import wavetoolbox as wtb
from mpl_toolkits import mplot3d

plt.close('all')

t = np.arange(0, 60, 0.1)
f = wtb.freqFromTime(t)
#T = 1/f#np.array([6, 8, 12, 8])#np.array([6])#
theta = np.arange(0,360, 5)*(np.pi/180)#np.array([0, 5, 10, 15, 20])*(np.pi/180)#np.array([15])*(np.pi/180)#
x = np.arange(0,500, 5)#np.array([0])#
y = np.arange(0,250, 5)#np.array([0])#

A = (wtb.jonswap(f, 1, 10, 1)*(f[1] - f[0])*2)**0.5 #np.array([0.5, 1, 2, 1])#np.array([1])#
A = A.reshape(-1,1)*(np.ones(np.size(theta))/np.size(theta)) #distribute amplitudes over directions given by theta
phi = np.random.random_sample(np.shape(A))*2*np.pi 

w = 2*np.pi*f#/T
k, L = wtb.waveNumber(f)#1/T)
grid = np.meshgrid(x, y)

## Generate trace using loops through x, y, w
#trace_loop = np.zeros([np.size(x), np.size(y), np.size(t)])
#for ix in range(np.size(x)):
#    for iy in range(np.size(y)):
##        for iw in range(len(A)):
##            trace_loop[ix, iy, :] = trace_loop[ix, iy, :] + A[iw]*np.cos(-w[iw]*t + k[iw]*(x[ix]*np.cos(theta[iw]) + y[iy]*np.sin(theta[iw])) + phi[iw])
#        for iD in range(np.size(theta)):
#            for iw in range(np.size(w)):
#                trace_loop[ix, iy, :] = trace_loop[ix, iy, :] + A[iw, iD]*np.cos(-w[iw]*t + k[iw]*(x[ix]*np.cos(theta[iD]) + y[iy]*np.sin(theta[iD])) + phi[iw, iD])

# Generate trace using matrices
X = np.expand_dims(np.transpose(grid[0]), axis = 2)
Y = np.expand_dims(np.transpose(grid[1]), axis = 2)

## for the case where amplitude and direction are both functions of frequency only, i.e. shape(A, theta) = [Nf]
#waveField = np.zeros([np.size(x), np.size(y), np.size(t)])
#cos_arg = X*(k*np.cos(theta)) + Y*(k*np.sin(theta)) + phi #argument of the cosine function excluding the time term. Shape: [Nx, Ny, Nf]
#cos_arg = np.expand_dims(cos_arg, axis = 2) # add another dimension to facilitate broadcasting of time. Shape [Nx, Ny, 1, Nf]
#cos_arg = cos_arg - np.transpose(w.reshape(-1,1)*t) # add the time term [Nt, Nf] to argument of the cosine function [Nx, Ny, 1, Nf]. Resulting shape: [Nx, Ny, Nt, Nf]
#waveField = np.matmul(np.cos(cos_arg), A) # take cosine and implement matrix multiplication with amplitude. Shape [Nx, Ny, Nt, Nf]*[Nf, 1] = [Nx, Ny, Nt]

# for the case where amplitude specified as a function of frequency and direction, i.e. shape(A) = [Nf, Nd]
X = np.expand_dims(X, axis = 2) # better just to add 2 dimensions at once... need to figure this out
Y = np.expand_dims(Y, axis = 2)
cos_arg = X*(k.reshape(-1,1)*np.cos(theta)) + Y*(k.reshape(-1,1)*np.sin(theta)) + phi#argument of the cosine function excluding the time & phase terms. Shape: [Nx, Ny, Nf, Nd]
cos_arg = np.expand_dims(cos_arg, axis = 2) # add another dimension to facilitate broadcasting of time. Shape [Nx, Ny, 1, Nf, Nd]
wt = np.transpose(w.reshape(-1,1)*t) # time term. Shape: [Nt, Nf]
wt = np.expand_dims(wt, axis = 2) # add dimension for broadcasting of direction
#cos_arg = cos_arg - wt # add the time term [Nt, Nf] to argument of the cosine function [Nx, Ny, 1, Nf]. Resulting shape: [Nx, Ny, Nt, Nf, Nd]
#waveField = np.einsum('...ij,ij', np.cos(cos_arg), A) # take cosine and multiply by amplitude using high dimensional matrix multiplication (Einstein sum). Shape [Nx, Ny, Nt, Nf, Nd]*[Nf, Nd] = [Nx, Ny, Nt]

waveField = np.zeros([np.size(x), np.size(y), np.size(t)])
for ix in range(np.size(x)):
    print(ix)
    for iy in range(np.size(y)):        
        waveField[ix, iy, :] = np.einsum('...ij,ij', np.cos(cos_arg[ix, iy, :, :, :] - wt), A)


ix = np.random.randint(0,len(x))
iy = np.random.randint(0,len(y))
it = np.random.randint(0,len(t))

plt.figure(1)
plt.subplot(3, 1, 1)
plt.plot(t, trace_loop[ix, iy, :], '-b', t, waveField[ix, iy, :], '.r')

plt.subplot(3, 1, 2)
plt.plot(x, trace_loop[:, iy, it], '-b', x, waveField[:, iy, it], '.r')

plt.subplot(3, 1, 3)
plt.plot(y, trace_loop[ix, :, it], '-b', y, waveField[ix, :, it], '.r')

fig = plt.figure(2)
ax = plt.axes(projection='3d')
#ax.contour3D(X, Y, Z, 50, cmap='binary')
ax.plot_surface(np.transpose(grid[0]), np.transpose(grid[1]), waveField[:, :, it], rstride=1, cstride=1, cmap='viridis', edgecolor='k')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')


