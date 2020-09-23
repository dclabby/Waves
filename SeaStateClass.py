#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 14:27:13 2020

@author: dclabby
"""

import numpy as np
import wavetoolbox as wtb
import scipy.stats

class SeaState(object):
    def __init__(self, Hs = None, Tp = None, time = None, freq = None, trace = None, varSpec = None, phase = None, gamma = 1, depth = 1E6):
        self.Hs = Hs
        self.Tp = Tp
        self.time = time
        self.freq = freq
        self.trace = trace
        self.varSpec = varSpec
        self.phase = phase
        self.gamma = gamma
        self.depth = depth
                
        # initialize time or frequency (depending on which is undefined)
        if self.freq is None and self.time is not None:
            self.freq = wtb.freqFromTime(self.time)
        elif self.time is None and self.freq is not None:
            self.time = wtb.timeFromFreq(self.freq)
        else:
            print('either time or frequency vector must be defined')
        
        # initialize phase if undefined (will be overwritten if sea is deterministically defined)
        if self.phase is None:
            self.phase = np.random.rand(np.size(self.freq))*(2*np.pi)
        
        # initialize remaining undefined parameters
        if all([v is not None for v in [Hs, Tp]]) and  all([v is None for v in [trace, varSpec]]): # parametric definition of sea state
            self.varSpec = wtb.jonswap(self.freq, self.Hs, self.Tp, self.gamma)
            self.trace = wtb.traceFromSpec(self.freq, self.varSpec, self.phase)
        elif varSpec is not None and all([v is None for v in [Hs, Tp, trace]]): # spectral definition of sea state
            print('Spectral definition of sea')
            self.trace = wtb.traceFromSpec(self.freq, self.varSpec, self.phase)
            self.Hs = wtb.calcHs(self.freq, self.varSpec)
            self.Tp = wtb.calcTp(self.freq, self.varSpec)
        elif trace is not None and all([v is None for v in [Hs, Tp, varSpec]]): # deterministic definition of sea state
            print('Deterministic definition of sea')
            self.varSpec, self.phase = wtb.specFromTrace(self.time, self.trace)
            self.Hs = wtb.calcHs(self.freq, self.varSpec)
            self.Tp = wtb.calcTp(self.freq, self.varSpec)
        else:
            print('sea state may not be fully defined')
    
    def __add__(self, other):
        trace_sum = self.getTrace() + other.getTrace()
        sea_sum = SeaState(time = self.getTime(), trace = trace_sum)
        return sea_sum
    
    # Getters
    def getHs(self):
        return self.Hs
    def getTp(self):
        return self.Tp
    def getTime(self):
        return self.time
    def getFreq(self):
        return self.freq
    def getTrace(self):
        return self.trace
    def getVar(self):
        return self.varSpec
    def getPhase(self):
        return self.phase
    def getGamma(self):
        return self.gamma
    def getDepth(self):
        return self.depth
    
    # Setters
    def setHs(self, Hs):
        self.Hs = Hs
    def setTp(self, Tp):
        self.Tp = Tp
    def setTime(self, time):
        self.time = time
    def setFreq(self, freq):
        self.freq = freq
    def setTrace(self, trace):
        self.trace = trace
    def setVar(self, varSpec):
        self.varSpec = varSpec
    def setPhase(self, phase):
        self.phase = phase
    def setGamma(self, gamma):
        self.gamma = gamma
    def setDepth(self, depth):
        self.depth = depth
    
class DirectionalSea(SeaState):
    def __init__(self, Hs = None, Tp = None, time = None, freq = None, trace = None, varSpec = None, phase = None, gamma = 1, depth = 1E6, dTheta = 1, waveDir = 0, x = 0, y = 0):
        
        if np.ndim(varSpec) <= 1: # parametric definition of directional sea state (np.ndim(varSpec) = 0) or spectral definition of non-directional sea state (np.ndim(varSpec) = 1)
            super().__init__(Hs, Tp, time, freq, trace, varSpec, phase, gamma, depth)
            
            self.dTheta = dTheta # angular resolution [degrees]
            self.theta = np.arange(-180, 180, self.dTheta) # direction vector from -180 degrees to 180 degrees at increment given by dTHeta [degrees]
            self.waveDir = waveDir #[mean wave direction [degrees]; standard deviation of wave direction [degrees]]
            self.x = x
            self.y = y
            
            if np.size(self.waveDir) == 1: # only average direction specified, no spread
                if type(waveDir) is not np.ndarray: # if waveDir is not a numpy array
                    self.waveDir = np.array([waveDir, 0]) # convert it to an array
                spreadVector = np.zeros(np.size(self.theta)) # initialize the spread vector
                spreadVector[(np.abs(self.theta - self.waveDir[0])).argmin()] = 1 # no spread - spectrum will be concentrated entirely in the directional band corresponding to the mean wave direction, specified by waveDir[0]
            else:
                spreadVector = scipy.stats.norm(0, self.waveDir[1]).pdf(self.theta)*dTheta # define spread vector as a normal distribution with zero mean and standard deviation defined by waveDir[1]
                spreadVector = np.roll(spreadVector, int(self.waveDir[0]/dTheta)) # roll (or shift) spread vector to the target mean defined by waveDir[0], this ensures that the spread vector is "circularly" distributed about the 360 degree directional range
            self.varSpec = self.varSpec.reshape(-1,1)*spreadVector # apply the spread vector to the variance specturum
            
#        else: # spectral definition of directional sea state (np.ndim(varSpec) = 2)
#            if self.freq is None and self.time is not None:
#                self.freq = wtb.freqFromTime(self.time)
#            elif self.time is None and self.freq is not None:
#                self.time = wtb.timeFromFreq(self.freq)
#            else:
#                print('either time or frequency vector must be defined')
#            
#            self.Hs = wtb.calcHs(self.freq, np.sum(self.varSpec, axis = 1))
#            self.Tp = wtb.calcTp(self.freq, np.sum(self.varSpec, axis = 1))
        
        ampSpec = (2*(self.freq[1] - self.freq[0])*self.varSpec)**0.5 # convert variance spectrum to amplitude spectrum
        
        if np.shape(self.phase) != np.shape(ampSpec): # if the phase & amplitude arrays are not of the same size, then the phase array must be redefined 
            phase_tmp = np.random.random_sample(np.shape(ampSpec))*2*np.pi # temporary 
            if self.phase is None: # if phase is not defined then assign phase_tmp as phase
                self.phase = phase_tmp
            elif np.ndim(self.phase) == 1: # if the phase is 1D then retain this in the appropriate directional band (given by waveDir[0]), with random values for other directional bands taken from phase_tmp
                phase_tmp[:, (np.abs(self.theta - self.waveDir[0])).argmin()] = self.phase # overwrite the temporary phase in the appropriate directional band with the original 1D phase vector
                self.phase = phase_tmp # assign phase_tmp as phase
#            self.trace = None # trace will need to be overwritten
        
        grid = np.meshgrid(self.x, self.y)
        w = 2*np.pi*self.freq
        k, L = wtb.waveNumber(self.freq)  
        
        X = np.expand_dims(np.transpose(grid[0]), axis = 2)
        Y = np.expand_dims(np.transpose(grid[1]), axis = 2)
        X = np.expand_dims(X, axis = 2) # better just to add 2 dimensions at once... need to figure this out
        Y = np.expand_dims(Y, axis = 2)
        
#        X = np.moveaxis(np.array([[np.transpose(grid[0])]]), [0, 1], [-2, -1])
#        Y = np.moveaxis(np.array([[np.transpose(grid[1])]]), [0, 1], [-2, -1])
        
        cos_arg = X*(k.reshape(-1,1)*np.cos(self.theta*np.pi/180)) + Y*(k.reshape(-1,1)*np.sin(self.theta*np.pi/180)) + self.phase # argument of the cosine function excluding the time term. Shape: [Nx, Ny, Nf, Nd]
        cos_arg = np.expand_dims(cos_arg, axis = 2) # add another dimension to facilitate broadcasting of time. Shape [Nx, Ny, 1, Nf, Nd]
        wt = np.transpose(w.reshape(-1,1)*self.time) # time term. Shape: [Nt, Nf]
        wt = np.expand_dims(wt, axis = 2) # add dimension for broadcasting of direction [Nt, Nf, 1]
        cos_arg = cos_arg + wt # add the time term [Nt, Nf, 1] to argument of the cosine function [Nx, Ny, 1, Nf, Nd]. Resulting shape: [Nx, Ny, Nt, Nf, Nd]
        waveField = np.einsum('...ij,ij', np.cos(cos_arg), ampSpec) # take cosine and multiply by amplitude specturm using high dimensional matrix multiplication (Einstein sum). Shape [Nx, Ny, Nt, Nf, Nd]*[Nf, Nd] = [Nx, Ny, Nt]
        
        self.grid = grid
        self.waveField = waveField
#        if self.trace is None:
#            self.trace = self.waveField[0, 0, :]
            
#        print('size(cos_arg) = ', str(np.shape(cos_arg)), '; size(ampSpec) = ', str(np.shape(ampSpec)))
        
        
                
                