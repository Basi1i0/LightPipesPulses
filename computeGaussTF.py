# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 19:33:08 2018

@author: Basil
"""
#from numpy import *
import gc
import os
import numpy
import time
import timeit

from joblib import Parallel, delayed

from LightPipes import cm, mm, nm

import SpecGenerator
import StepsGenerator
import SaveLoad
import namedtuplesGaussTF 

    
def GetIntensityDependence(x, ts, zs, useRunningTime = False):
    intensities_z = numpy.full( (len(zs), len(ts), x.getGridDimension(), x.getGridDimension()), numpy.NaN);
    
    x.Forvard(zs[0])#x.Forvard(zs[0])
    for i in range(0, len(ts)):
        print('\rz =', numpy.round(zs[0], 4),', t =', numpy.round(ts[i], 5), end='\r')
        Isum = x.Intensity(0, ts[i] + zs[0]*useRunningTime) #t + z - z0
        intensities_z[0][i] = numpy.array(Isum)#intensities_t.append(Isum)    
    
#    refr_n = (0.913 + 10.j)*numpy.ones((x.getGridDimension(), x.getGridDimension()))
    for iz in range(1, len(zs)):
        z_prev = zs[iz-1] if iz != 0 else 0
        dz = zs[iz] - z_prev;
        x.Forvard(dz)#x.Steps(dz, 1, refr_n) #x.Fresnel(z) #
#        x.AddDispersion(z/delta_z*10000000, 2)
        for i in range(0, len(ts)):
            print('\rz =', numpy.round(zs[iz], 4),', t =', numpy.round(ts[i], 5), end='\r')
            Isum = x.Intensity(0, ts[i] + zs[iz]*useRunningTime) #t + z - z0
            intensities_z[iz][i] = numpy.array(Isum)#intensities_t.append(Isum)
    return intensities_z;

def Propogate(x, zs, ts, t0, n_jobs = 1, useRunningTime = False):
    zs_intervals = numpy.split(zs, n_jobs)
    # for lp.n_jobs = 1 use copy.deepcopy(x) as an argument
    Izt0yx = numpy.array(Parallel(n_jobs=n_jobs)(delayed(GetIntensityDependence)(x, ts+t0, zs, True) for zs in zs_intervals))#GetIntensityDependence(x, ts+t0, zs, True)
    return numpy.array([item for sublist in Izt0yx for item in sublist])
    
def PrePropogation(x, f): 
#    x.Axicon(175/180*3.1415, 2, 0, 0)
#    x.Forvard(0.005)
    x.Forvard(f)
    x.Lens(f, 0.00, 0)
    x.Forvard(2*f)
    x.Lens(f, 0.00, 0)
    return 3.*f

#if __name__ == '__main__':
def Run(s, lp):    
#    s = SpecGenerator.SpecGenerate(800*nm, 3*nm, 120, False)
#    lp = namedtuplesGaussTF.LaunchParams(n_jobs, Nx, Nz, Nt, size, f, w1, alpha,  b)
    cp = namedtuplesGaussTF.ComputParmas((2*numpy.pi/s.lambda0 * lp.w1**2), lp.alpha/lp.w1/(s.tau/2.998e-4), 
                                         -lp.b/((s.tau/2.998e-7)**2), (2*numpy.pi/s.lambda0 * lp.w1**2) / lp.f)
    g  = StepsGenerator.GridsAll(StepsGenerator.StepsGenerate(-lp.size/2, lp.size/2, lp.Nx), 
                                 StepsGenerator.StepsGenerate(0.8*lp.f, 1.2*lp.f, lp.Nz), 
                                 StepsGenerator.StepsGenerate(-3*s.tau*numpy.sqrt(1 + (lp.alpha/lp.w1/(s.tau/2.998e-4))**2), 
                                                               3*s.tau*numpy.sqrt(1 + (lp.alpha/lp.w1/(s.tau/2.998e-4))**2), lp.Nt))#2e-6*lp.alpha*lambda0/w1**3
    
    x = lpPulse(s.lambdas, s.amps, lp.size, lp.Nx)#LP(lp.size, 800e-9, lp.Nx)#
    x.GaussAperture(lp.f/(2*numpy.pi/s.lambda0)/lp.w1, 0*mm, 0, 1)
    x.AddDispersion(lp.b, 2)   
    x.GratingX( lp.alpha*2*numpy.pi*(2.998*1e-4) / (lp.f*s.lambda0**2) , 800e-9)
   
    t0 = PrePropogation(x, lp.f)
       
    start_time = timeit.default_timer()
    Izt0yx = Propogate(x, g.z.points, g.t.points, t0, lp.n_jobs, True)
    elapsed = timeit.default_timer() - start_time
    print('\nIt took', elapsed)
    
    return Izt0yx,s,lp,cp,g

    
#plt.rcParams["figure.figsize"] = [12, 8]
Izt0yx = None; Iztyx = None; Ixtyz = None; Iytxz = None;
gc.collect()

s = SpecGenerator.SpecGenerate(800*nm, 3*nm, 120, False)
n_jobs = 2
Nx=270; Nz=26; Nt=100
size = 5.*mm;
f = 100*mm
w1 = f/(2*numpy.pi/s.lambda0)/(0.1*mm)
alpha = 0.1*mm # mm*ps
b = 0.8e5 #fs**2

for bscale in numpy.linspace(-0.5, 1.5, 9):
    print(bscale)

    Izt0yx,s,lp,cp,g = Run(s, namedtuplesGaussTF.LaunchParams(n_jobs, Nx, Nz, Nt, size, f, w1, alpha,  b*bscale ))
    
    dirname = 'C://Users//Basil//ResearchData//FourierOptics//GaussianTF//' + \
                               time.strftime("%Y%m%d%H%M%S", time.gmtime())  + '_' + \
                               'f=' + str(lp.f) + '_w1=' + str('%.2E' % lp.w1) + \
                               '_alpha=' + str('%.2E' % lp.alpha) + '_b=' + str('%.2E' % lp.b)            
    os.mkdir(dirname)
    SaveLoad.DumpParamsToDisk(Izt0yx, s, lp, cp, g, dirname)


