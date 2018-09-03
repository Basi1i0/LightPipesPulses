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

def AxiconZ(z):
   return 2*numpy.pi* k0**2 * w1**6 *z/a**2/(k0**2 * w1**4 + z**2)**(3/2) *numpy.exp( - z**2 * w1**2 /a**2 /(k0**2 * w1**4 + z**2) ) # 2*numpy.pi*z/a**2/k0*numpy.exp( - z**2 /a**2 /k0**2 / w1**2 )

ms = list([])
#for scale in range(0,20):
#print(scale)

lambda0 = 800*nm
N = 400
size = 0.0015
xs =  numpy.linspace(-size/2, size/2, N)
f = 100*mm
k0 = (2*numpy.pi/lambda0)
n = 1.5
a = 0.1*mm /2*(random.random() + 0.5) #sigma_x # 2.*1/(k0*(n - 1)*numpy.tan(alpha) ) # why 2.0 ???
alpha = numpy.arctan( 2.*1/(k0*(n - 1)*a ) ) #1/180*numpy.pi
w1 = 0.06 * 2/(a*k0)*(random.random() + 0.5) #sigma_zw #0.0005


x = LP(size, lambda0, N)

x.GaussAperture(w1, 0*mm, 0, 1)
x.Axicon(numpy.pi - alpha, n, 0, 0)

zs,start_z,end_z,steps_z,delta_z = StepsGenerator.StepsGenerate(0, 8*f, 200)
 
intensities_z = numpy.full( (len(zs), x.getGridDimension(), x.getGridDimension()), numpy.NaN);

for iz in range(0, len(zs)):
    z_prev = zs[iz-1] if iz != 0 else 0
    dz = zs[iz] - z_prev;
    x.Forvard(dz)
    #t + z - z0
    intensities_z[iz] =  x.Intensity(0)
       
plt.imshow(intensities_z[:,N//2])
plt.show()
plt.plot(zs, intensities_z[:,N//2, N//2] / max(intensities_z[:,N//2, N//2]), 'o' )
plt.plot(zs, AxiconZ(zs) / max( AxiconZ(zs)) )
plt.show()

plt.plot(xs, intensities_z[ numpy.argmax(intensities_z[:,N//2, N//2]), :, N//2] / max(intensities_z[ numpy.argmax(intensities_z[:,N//2, N//2]), :, N//2] ), 'o' )
plt.plot(xs, scipy.special.jv(0, xs/a)**2 )
plt.show()

ms.append([numpy.dot(zs, intensities_z[:,N//2, N//2]) / sum(intensities_z[:,N//2, N//2]) , 
           numpy.dot(zs, AxiconZ(zs)) / sum(AxiconZ(zs)) ])

#ms = numpy.array(ms)
#
#
#plt.plot(ms[:,0], ms[:, 1], 'o')
#plt.plot(ms[:,0], 0.48*ms[:, 0])

