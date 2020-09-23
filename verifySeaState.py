#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 15:49:19 2020

@author: dclabby
"""
import numpy as np
import matplotlib.pyplot as plt
import SeaStateClass as sea
import wavetoolbox as wtb

## *******Verify single sea state
#plt.close('all')
#
## parametric definition of sea state
#t = np.arange(0, 1024, 0.5)
#sea01 = sea.SeaState(Hs = 1, Tp = 10, time = t)
#
#plt.figure(1)
#plt.subplot(2, 1, 1)
#plt.plot(sea01.getFreq(), sea01.getVar())
#plt.subplot(2, 1, 2)
#plt.plot(sea01.getTime(), sea01.getTrace())
#
## statistical definition of sea state
#sea02 = sea.SeaState(freq = sea01.getFreq(), varSpec = sea01.getVar(), phase = sea01.getPhase())
#
#plt.figure(2)
#plt.subplot(2, 1, 1)
#plt.plot(sea02.getFreq(), sea02.getVar())
#plt.subplot(2, 1, 2)
#plt.plot(sea02.getTime(), sea02.getTrace())
#
## deterministic definition of sea state
#sea03 = sea.SeaState(time = sea02.getTime(), trace = sea02.getTrace())
#sea03.setTrace(wtb.traceFromSpec(sea03.getFreq(), sea03.getVar(), sea03.getPhase()))
#
#plt.figure(3)
#plt.subplot(2, 1, 1)
#plt.plot(sea03.getFreq(), sea03.getVar())
#plt.subplot(2, 1, 2)
#plt.plot(sea03.getTime(), sea03.getTrace())
#
## compare all
#plt.figure(4)
#plt.subplot(2, 1, 1)
#plt.plot(sea01.getFreq(), sea01.getVar(), '-k', sea02.getFreq(), sea02.getVar(), '+b', sea03.getFreq(), sea03.getVar(), '.r')
#
#plt.subplot(2, 1, 2)
#plt.plot(sea01.getTime(), sea01.getTrace(), '-k', sea02.getTime(), sea02.getTrace(), '+b', sea03.getTime(), sea03.getTrace(), '.r')

## *******Verify summation of sea states
#plt.close('all')
#sea01 = SeaState(Hs = 0.5, Tp = 6, time = t)
#sea02 = SeaState(Hs = 1, Tp = 12, time = t)
#sea03 = sea01 + sea02
#sea03_tracefromspec = wtb.traceFromSpec(sea03.getFreq(), sea03.getVar(), sea03.getPhase())
#
#plt.figure(1)
#plt.subplot(2, 1, 1)
#plt.plot(sea01.getFreq(), sea01.getVar() + sea02.getVar(), '-b', sea03.getFreq(), sea03.getVar(), 'or')
#
#plt.subplot(2, 1, 2)
#plt.plot(sea01.getTime(), sea01.getTrace() + sea02.getTrace(), '-b', sea03.getTime(), sea03.getTrace(), 'or', sea03.getTime(), sea03_tracefromspec, '.c')



## *******Verify fft
#plt.close('all')
time = np.arange(0, 600, 0.5)
A = np.array([1, 2, 3, 4])
T = np.array([30, 15, 30, 50])
phi = np.random.rand(len(A))*2*np.pi #np.array([73*np.pi/180])#
trace = np.zeros(np.size(time))
for i in range(len(A)):
    trace = trace + A[i]*np.cos((2*np.pi/T[i])*time + phi[i])

# generate trace using matrices rather than loop
w = 2*np.pi/T.reshape(-1, 1) # column vector of angular frequency
trace2 = np.matmul(A, np.cos(np.add(w*time, phi.reshape(-1,1))))
plt.plot(time, trace, '-b', time, trace2, '.c')

#
#dt = time[1] - time[0]
#Lt = np.size(time)
#Lf = int(Lt/2)
#Y = np.fft.fft(trace)
#freq = np.arange(0, Lf)/(Lt*dt)
#amp = abs(Y[0:Lf])/(Lf)
#phase = np.angle(Y[0:Lf])
#varSpec = (amp**2)/(2*(freq[1] - freq[0]))
#
##Z = (Lt/2)*amp*(np.cos(phase) + 1j*np.sin(phase))
##Z = (Lt/2)*amp*(np.sin(phase) + 1j*np.cos(phase))
##Z = np.concatenate((Z, np.flip(Z)))
#Z = (Lt)*amp*(np.cos(phase) + 1j*np.sin(phase))
#Z = np.concatenate((Z, np.zeros(np.size(Z))))
##Z = np.concatenate((Z, Z))
#trace2 = np.real(np.fft.ifft(Z))
##trace2 = trace2[:Lt]
#
#plt.figure(1)
#plt.subplot(1, 2, 1)
#plt.plot(time, trace, '-k', time, trace2, '-r')
#plt.subplot(1, 2, 2)
#plt.plot(1/T, A, 'ob', freq, amp, '-k')