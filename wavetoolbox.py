#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 14:01:55 2020

@author: dclabby
"""

import numpy as np

# Functions to implement conversion between time & frequency domain vectors
def freqFromTime(time):
    L = np.size(time)
    dt = time[1] - time[0]
    freq = np.arange(0, L/2)/(L*dt)
    return freq

def timeFromFreq(freq):
    L = np.size(freq)*2
    df = freq[1] - freq[0]
    time = np.arange(0, L)/(L*df)
    return time

# Functions to implement Fourier analysis for conversion between time & frequency domain representations
def traceFromSpec(freq, varSpec, phase): # implements inverse Fourier transform to obtain a time trace from a spectrum
    L = 2*np.size(freq)
    ampSpec = (2*varSpec*(freq[1] - freq[0]))**0.5
    Z = L*ampSpec*(np.cos(phase) + 1j*np.sin(phase))
    Z = np.concatenate((Z, np.zeros(np.size(Z)))) # Z is one sided, not symmetric
    trace = np.real(np.fft.ifft(Z))
    return trace

def specFromTrace(time, trace): # implements Fourier transform to obtain a spectrum from a time trace
    #fft(trace)
    dt = time[1] - time[0]
    Lt = np.size(time)
    Lf = int(Lt/2)
    Y = np.fft.fft(trace)
    freq = np.arange(0, Lf)/(Lt*dt)
    ampSpec = abs(Y[0:Lf])/(Lf)
    phase = np.angle(Y[0:Lf])
    varSpec = (ampSpec**2)/(2*(freq[1] - freq[0]))
    return varSpec, phase

# Parametric wave spectra
def jonswap(freq, Hs, Tp, gamma):
    varSpec = np.zeros(np.size(freq))
    fp = 1/Tp
    f_tmp = freq[freq > 0]
    sigma = np.zeros(np.size(f_tmp))
    sigma[f_tmp <= fp] = 0.07
    sigma[f_tmp > fp] = 0.09
    beta = (1.094-0.01915*np.log(gamma))*(0.06238)/(0.23+(0.0336*gamma)-0.185/(1.9+gamma))
    A = beta*(Hs**2)*(Tp**(-4))*(f_tmp**(-5))
    E = np.exp(-1.25*(Tp*f_tmp)**(-4))
    P = gamma**(np.exp(-(Tp*f_tmp - 1)**2/(2*sigma**2)))
    varSpec[freq > 0] = A*E*P
    Hm0 = 4*((np.sum(varSpec)*(freq[1] - freq[0]))**0.5)
    varSpec = varSpec*((Hs/Hm0)**2)
    return varSpec

def monochromatic(freq, A, T):
    df = freq[1] - freq[0]
    varSpec = np.zeros(np.size(freq))
    varSpec[[int(v) for v in (np.round((1/T)/df))]] = (A**2)/(2*df)
    
# Calculaton of wave parameters
def spectralMoment(freq, varSpec, j): # calculates the jth spectral moment of the spectrum
    mj = np.dot(freq[freq > 0]**j, varSpec[freq > 0])*(freq[1] - freq[0])
    return mj

def waveNumber(freq): # note: freq is normal frequency, not angular frequency
    k = np.zeros(np.shape(freq))
    L = np.zeros(np.shape(freq))
    w = 2*np.pi*freq[freq != 0]# convert to angular frequency
    k[freq != 0] = (w**2)/9.81 # deep water assumption
    L[freq != 0] = 2*np.pi/k[freq != 0] # wavelength
    return k, L

def calcHs(freq, varSpec): # calculate the significant wave height from the variance spectrum
    Hs = 4*((spectralMoment(freq, varSpec, 0))**0.5)
    return Hs

def calcTp(freq, varSpec): # calculate the peak wave period from the variance spectrum
    Tp = 1/freq[np.argmax(varSpec)]
    return Tp

def calcTm(freq, varSpec): # calculate the mean wave period from the variance spectrum
    Tm = spectralMoment(freq, varSpec, 0)/spectralMoment(freq, varSpec, 1)
    return Tm

def calcTe(freq, varSpec): # calculate the wave energy period from the variance spectrum
    Te = spectralMoment(freq, varSpec, -1)/spectralMoment(freq, varSpec, 0)
    return Te