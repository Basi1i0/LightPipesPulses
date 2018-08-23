# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 17:07:30 2018

@author: Basil
"""
import numpy
from copy import deepcopy

#import LP

class lpPulse:
    
    def __init__(self, lambdas, amplitudes, GridSize, N, lps = None):
        self._lambdas = lambdas
        self._amplitudes = amplitudes
        self._GridSize = GridSize
        self._N = N
        
        self._lps = lps
        if (self._lps == None):
            self._initLPs()


    def _initLPs(self):
        if (not (len(self._lambdas) == len(self._amplitudes) and 
                 (self._lps == None) )):
            raise ValueError('There is a mismach between the length ' +
                             'of lambdas, amplitudes and fields arrays')
        self._lps = []
        for i in range(0, len(self._lambdas)):
            self._lps.append( LP(self._GridSize, self._lambdas[i], self._N) )        
        
    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        methods = []
        for f in self._lps:
            attr = getattr(f, name)
            if callable(attr):
                methods.append(attr)
        if methods:
            def _map(*args, **kw):
                return [method(*args, **kw) for method in methods]
            return _map
        raise AttributeError(name)
        
    def __reduce__(self):
        return (self.__class__, (self._lambdas, self._amplitudes, self._GridSize, self._N, deepcopy(self._lps)) )
    
    def AddDispersion(self, value, n = 2): #value has dimention [fs**n]
        central = numpy.mean(self._lambdas)
        dlambdas = (self._lambdas - central )
        phases = numpy.exp(-1j*value/2*(2*numpy.pi*2.998 * (dlambdas*1e7) / (central*1e7)**2 )**n )
        self._amplitudes = self._amplitudes*phases
    
    def Field (self, time):
        amp = numpy.multiply(self._lps[0].Field(), 
                             self._amplitudes[0]*numpy.exp(-1j*2*numpy.pi/self._lambdas[0]*time))
        for i in range(1,len(self._lps)):
            amp = numpy.add(amp, numpy.multiply(self._lps[i].Field(), 
                                                self._amplitudes[i]*numpy.exp(-1j*2*numpy.pi/self._lambdas[i]*time)))
        amp = numpy.divide(amp, len(self._lps))
        return amp
    
    def Intensity (self, flag, time):
        field = self.Field(time)
        return self._lps[0].IntensityExt(flag, field)
    
    def getGridSize(self):
        return self._GridSize
    
    def getGridDimension(self):
        return self._N
    
        
    def get_lambdas(self):
        return self._lambdas
    
    
    def get_amplitudes(self):
        return self._amplitudes
    
    
    def get_fields(self):
        return self._lps
    

#x = lpPulse([5e-7, 5e-7], [1, 0.6], 0.01, 50)


        