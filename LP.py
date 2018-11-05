# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 16:40:08 2018

@author: Basil
"""
import LightPipes
import numpy
from warnings import warn
from copy import deepcopy

class LP:
    def __init__(self, size, wavelength, N, field = None ):
        self._N = N
        self._wavelength = wavelength
        self._size = size
   
        self._lp = LightPipes.Init()
        
        self._field = field
        if (numpy.all( self._field == None ) ):
            self._field = numpy.array(self._lp.Begin(size, wavelength, N))
        else:
            self._lp.Begin(size, wavelength, N)
        
#    def __add__(self, LP2):
#        if (self._N != LP2.getGridDimension()):
#            IndexError("Grid dimensions do not match")
#        if (self._size != LP2.getGridSize()):
#            IndexError("Grid sizes do not match")
#        
#        return self._lp.BeamMix(self._field, LP2.getField() )      
        
    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        attr = getattr(self._lp , name)
        if callable(attr):
            def _map(*args):
                self._field = numpy.array(attr(*args, self._field))
                return self._field
            return _map
        raise AttributeError(name)
        
#    def __deepcopy__(self, memo):
#        new_instance = LP(self._size, self._wavelength, self._N. self._field)
#        warn('LightPipes obj is copied, doub1 and int1 are set to 0 in the copy')
#        return new_instance
        
    def __reduce__(self):
        warn('LightPipes obj is copied, doub1 and int1 are set to 0 in the copy')
        return (self.__class__, (self._size, self._wavelength, self._N, self._field) )
    
    def GratingX(self, algular_dispersion, lambda0 = 500e-9):
        if algular_dispersion == 0:
            return
        
        tilt = algular_dispersion*(self._wavelength - lambda0)
        pointsperperiod = 2*numpy.pi*self._wavelength/numpy.abs(tilt)*self._N/self._size
        if pointsperperiod < 10:
            warn(str(pointsperperiod) + ' points per phase period, consider decreasing lateral step')
        self._field = numpy.array(self._lp.Tilt( tilt, 0, self._field))
    
    def Field(self):
        return self._field
                
    def IntensityExt(self, flag, field):
        return numpy.array(self._lp.Intensity(flag, field))
    
    def Intensity(self, flag):
        return numpy.array(self._lp.Intensity(flag, self._field))
    
    def Phase(self):
        return numpy.array(self._lp.Phase(self._field))
    
    def PhaseUnwrap(self, Phi):
        return numpy.array(self._lp.PhaseUnwrap(Phi))
                
    def Power(self):
        return self._lp.Power(self._field)
        
    def Strehl(self):
        return self._lp.Strehl(self._field)
    
    def version(self):
        return self._lp.version();
    
    def getGridSize(self):
        return self._lp.getGridSize();
    
    def setGridSize(self, newSize):
        return self._lp.setGridSize(newSize);
        
    def getWavelength(self):
        return self._lp.getWavelength();
       
    def setWavelength(self, newWavelength):
        return self._lp.setWavelength(newWavelength)
        
    def getGridDimension(self):
        return self._lp.getGridDimension()
    