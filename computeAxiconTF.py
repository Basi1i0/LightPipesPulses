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
import copy

from joblib import Parallel, delayed

from LightPipes import cm, mm, nm

import SpecGenerator
import StepsGenerator
import SaveLoad
import namedtuplesAxiconTF
from r1r2 import *
    

    
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
    Izt0yx = numpy.array(Parallel(n_jobs=n_jobs, prefer="threads")(delayed(GetIntensityDependence)(copy.deepcopy(x), ts+t0, zs, True) for zs in zs_intervals))#GetIntensityDependence(x, ts+t0, zs, True)
    return numpy.array([item for sublist in Izt0yx for item in sublist])
    
def PrePropogation(x, f, z =0): 
#    x.Axicon(175/180*3.1415, 2, 0, 0)
#    x.Forvard(0.005)
    x.Forvard(f)
    x.Lens(f, 0.00, 0)
    x.Forvard(2*f)
    x.Lens(f, 0.00, 0)
    x.Forvard(z)
    return 3.*f + z

#if __name__ == '__main__':
def Run(s, lp):    
#    s = SpecGenerator.SpecGenerate(800*nm, 3*nm, 120, False)
#    lp = namedtuplesGaussTF.LaunchParams(n_jobs, Nx, Nz, Nt, size, f, w1, alpha,  b)
    cp = namedtuplesAxiconTF.ComputParmas((2*numpy.pi/s.lambda0 * lp.w1**2), lp.alpha/lp.w1/(s.tau/2.998e-4), 
                                          -lp.b/((s.tau/2.998e-7)**2), (2*numpy.pi/s.lambda0 * lp.w1**2) / lp.f)
    ztmax = 0
    if(lp.b != 0): ztmax = lp.f*(1 - 1/(cp.D2*cp.BdS2))
    
    tmaxr = numpy.sqrt(r2(ztmax, cp)/r1(ztmax, cp))
    zdr = 1/(cp.D2*numpy.sqrt(1 + cp.AdS**2))
    g  = StepsGenerator.GridsAll(StepsGenerator.StepsGenerate(-lp.size/2, lp.size/2, lp.Nx), 
                                 StepsGenerator.StepsGenerate(max((1-2*zdr)*lp.f, 0), (1+2*zdr)*lp.f, lp.Nz), 
                                 StepsGenerator.StepsGenerate(-5*tmaxr*s.tau, 5*tmaxr*s.tau, lp.Nt))#2e-6*lp.alpha*lambda0/w1**3
     
    x = lpPulse(s.lambdas, s.amps, lp.size, lp.Nx)#LP(lp.size, 800e-9, lp.Nx)#
    x.GaussAperture( lp.sz/(lp.a*(2*numpy.pi/s.lambda0) ) , 0*mm, 0, 1)
    x.AddDispersion(lp.b, 2)   
       
    x.Axicon(numpy.pi - theta, n, 0, 0)
    x.Forvard( sz/numpy.sqrt(2) )
    x.GratingX( lp.alpha*2*numpy.pi*(2.998*1e-4) / (lp.f*s.lambda0**2) , 800e-9)

    t0 = PrePropogation(x, lp.f, g.z.points[0] ) + sz/numpy.sqrt(2)
#    
    xmax = lp.a*numpy.sqrt(r1(g.z.points[0], cp))
    Nx2 = int(15*xmax*lp.Nx/lp.size); Nx2 = Nx2 + (Nx2%2)
    if(Nx2 > lp.Nx): 
        Nx2 = lp.Nx
        y = x
        lp2 = lp
    else:
        sub_lps = []   
        for i in range(0, len(x.get_lambdas())):
            sub_lps.append(LP(lp.size*Nx2/lp.Nx, x.get_lambdas()[i], Nx2, 
                              x._lps[i]._field[((lp.Nx - Nx2)//2) : ((lp.Nx + Nx2)//2), 
                                               ((lp.Nx - Nx2)//2) : ((lp.Nx + Nx2)//2)]))      
        y = lpPulse(s.lambdas, x.get_amplitudes(), lp.size*Nx2/lp.Nx,  Nx2, sub_lps)    
        lp2 = namedtuplesAxiconTF.LaunchParams(lp.n_jobs, Nx2, lp.Nz, lp.Nt, lp.size*Nx2/lp.Nx, lp.f, lp.w1, lp.sz, lp.a, lp.theta, lp.alpha, lp.b )
        x = None
    
    start_time = timeit.default_timer()
    Izt0yx = Propogate(y, g.z.points - g.z.points[0], g.t.points, t0, lp.n_jobs, True)
    elapsed = timeit.default_timer() - start_time
    print('\nIt took', elapsed)
    
    return Izt0yx,s,lp2,cp,g

    
#plt.rcParams["figure.figsize"] = [12, 8]
Izt0yx = None; Iztyx = None; Ixtyz = None; Iytxz = None;
Iyz2p = None; Ixz2p = None; Ixy2p = None
gc.collect()

s = SpecGenerator.SpecGenerate(800*nm, 3*nm, 120, False)
n_jobs = 2
Nx= 1200; Nz=100; Nt=100
size = 10*mm;

f = 50*mm


w1 = 0.1*mm #0.06 * 2/(a*2*numpy.pi/s.lambda0) #sigma_zw #0.0005
sz = 500*mm 
alpha = 0.15 *mm # # mm*ps
r
b = 5*10**4   #fs**2


k0 = (2*numpy.pi/s.lambda0)
a = w1
n = 1.5
theta = numpy.arctan( 2.*1/(k0*(n - 1)*a ) ) #1/180*numpy.pi

win = sz / (a*k0)

#
#for bscale in numpy.linspace(-0.5, 3, 8):

bscale = 5
ascale = 1

print(bscale)

lp = namedtuplesAxiconTF.LaunchParams(n_jobs, Nx, Nz, Nt, size, f, f/(2*numpy.pi/s.lambda0)/w1 , sz, a, theta, alpha*ascale, b*bscale)
Izt0yx,s,lp,cp,g = Run(s, lp)

dirname = 'C:\\Users\\Basil\\ResearchData\\FourierOptics\\AxiconTF\\' + \
                           time.strftime("%Y%m%d%H%M%S", time.gmtime())  + '_' + \
                           'f=' + str(lp.f) + '_w1=' + str('%.2E' % lp.w1) + \
                           '_alpha=' + str('%.2E' % lp.alpha) + '_b=' + str('%.2E' % lp.b)            
os.mkdir(dirname)
SaveLoad.DumpParamsToDisk(Izt0yx, s, lp, cp, g, dirname)
#
#wt_z,wx_z,wy_z = PropogationFunctions.Ws_z(lp, g, Izt0yx)
#plt.plot(g.z.points, wt_z/3e-7, 'o-')
#plt.show()


