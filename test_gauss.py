# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 17:41:21 2018

@author: Basil
"""

import matplotlib.pyplot as plt
import gc
import numpy
import random

from joblib import Parallel, delayed

from LightPipes import cm, mm, nm

import scipy
import StepsGenerator

import copy

def AxiconZ(z):
   return 2*numpy.pi* k0**2 * w1**6 *z/a**2/(k0**2 * w1**4 + z**2)**(3/2) *numpy.exp( - z**2 * w1**2 /a**2 /(k0**2 * w1**4 + z**2) ) # 2*numpy.pi*z/a**2/k0*numpy.exp( - z**2 /a**2 /k0**2 / w1**2 )

ms = list([])
#for scale in range(0,20):
#print(scale)

lambda0 = 802*nm
N = 500
size = 1*mm
xs =  numpy.linspace(-size/2, size/2, N)
f = 10*mm
k0 = (2*numpy.pi/lambda0)
w1 =  0.01*mm#f/k0/(1*mm) #sigma_zw #0.0005
w0 = f/(k0*w1)
sz = k0*w1**2
alpha = (0.02*mm) 

x = LP(size, lambda0, N)

x.GaussAperture(w1, 0*mm, 0, 1)

x.GratingX( alpha*2*numpy.pi*(2.998*1e-4) / (f*lambda0**2) , 800e-9)
   
x.Forvard( f )
x.Lens(f, 0, 0)
x.Forvard( 2*f )
x.Lens(f, 0, 0)

zs,start_z,end_z,steps_z,delta_z = StepsGenerator.StepsGenerate(0, 2*f, 200)
 
intensities_z = numpy.full( (len(zs), x.getGridDimension(), x.getGridDimension()), numpy.NaN);

for iz in range(0, len(zs)):
#    xc = copy.deepcopy(x)
    z_prev = zs[iz-1] if iz != 0 else 0
    dz = zs[iz] - z_prev;
    if( dz > 0):
        x.Forvard(dz)
    #t + z - z0
    intensities_z[iz] =  x.Intensity(0)
       
plt.imshow(intensities_z[:,N//2])
plt.show()
plt.plot(zs / mm, intensities_z[:,N//2, N//2] / max(intensities_z[:,N//2, N//2]), 'o' )
plt.plot(zs / mm, 1/( 1 + ( (1 - zs/f)/(k0*w1**2/f) )**2  ) )

plt.axvline(f / mm, )
plt.xlabel('Z, mm'); plt.ylabel('I') 
plt.show()

plt.plot(xs / mm, intensities_z[ numpy.argmax(intensities_z[:,N//2, N//2]),  N//2, :] / max(intensities_z[ numpy.argmax(intensities_z[:,N//2, N//2]), :, N//2] ), 'o' )
plt.plot(xs / mm, numpy.exp( -xs**2 / w1**2 ) )
plt.xlabel('X, mm'); plt.ylabel('I') 
plt.show()
#
#plt.plot(xs / mm, intensities_z[200-1,N//2,] , 'o' )
#plt.xlabel('X, mm'); plt.ylabel('I') 
#plt.show()

