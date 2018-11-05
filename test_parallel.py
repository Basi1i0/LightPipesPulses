# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 14:21:09 2018

@author: Basil
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 19:33:08 2018

@author: Basil
"""
#from numpy import *
import gc
import numpy
import time

from joblib import Parallel, delayed

    
def GetIntensityDependence(zs, useRunningTime = False):
    intensities_z = numpy.full( (len(zs), Nall), numpy.NaN);
    time.sleep(10)
    return intensities_z;


Izt0yx = None;
gc.collect()

n_jobs = 2
Nall = 8000000
Nz=60;

zs_intervals = numpy.split(numpy.linspace(1, Nz, Nz), n_jobs)
Izt0yx = numpy.array(Parallel(n_jobs=n_jobs)(delayed(GetIntensityDependence)(zs, True) for zs in zs_intervals))#GetIntensityDependence(x, ts+t0, zs, True)
  
    
gc.collect()
    
