# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 17:58:37 2018

@author: Basil
"""


import matplotlib.pyplot as plt
import gc
import numpy
import scipy.special
from LightPipes import cm, mm, nm


import PropogationFunctions 
import PlotFunctions 
import namedtuplesAxiconTF
import SaveLoad

def Normalized(v, n = 0):
    if(n == 0):
        return v / max(numpy.abs(v))
    elif(n > 0):
        return v / sum(numpy.power(numpy.abs(v), n))
   
    print("n < 0, no normalization can be performed")
    return v
        
    
    

def AxiconZ(z):
   return 2*numpy.pi* k0**2 * w0**6 *z/a**2/(k0**2 * w0**4 + z**2)**(3/2) *numpy.exp( - z**2 * w0**2 /a**2 /(k0**2 * w0**4 + z**2) ) * numpy.heaviside(z, 0) # 2*numpy.pi*z/a**2/k0*numpy.exp( - z**2 /a**2 /k0**2 / w0**2 )

Izt0yx = None; Iztyx = None; Ixtyz = None; Iytxz = None;
Iyz2p = None; Ixz2p = None; Ixy2p = None
gc.collect()

#dirname = 'C:\\Users\\Basil\\ResearchData\\FourierOptics\\AxiconTF\\' + \
#          '20181031194535_f=0.05_w1=6.37E-05_alpha=1.50E-04_b=1.50E+05'
Iztyx,s,lp,cp,g = SaveLoad.ReadDumpFromDisk(dirname, namedtuplesAxiconTF.LaunchParams, namedtuplesAxiconTF.ComputParmas)


#    for different Z
#PlotFunctions.PlotMovie(numpy.sum(numpy.swapaxes(Iztyx, 1, 3), axis =2 ),
#                        (g.t.start/mm, g.t.end/mm), (-lp.size/2/mm, lp.size/2/mm),
#                        g.z.points, 'T, mm', 'X, mm', aspect = lp.Nx/g.t.steps*0.2)

wt_z,wx_z,wy_z = PropogationFunctions.Ws_z(lp, g, Iztyx)

plt.plot(g.z.points, wt_z/3e-7, 'o')
plt.plot(g.z.points, numpy.repeat(s.tau/3e-7/numpy.sqrt(2), g.z.steps ))
plt.plot(g.z.points, s.tau/3e-7/numpy.sqrt(2) * numpy.sqrt(r2(g.z.points, cp)/r1(g.z.points, cp)))
plt.xlabel('Z, m'); plt.ylabel('\Delta T, fs') 
plt.ylim( 0, max(wt_z)*1.1/3e-7 ) 
plt.show()

plt.plot(g.z.points, wx_z/mm, 'o')
plt.plot(g.z.points, s.lambda0/(2*numpy.pi)*lp.f/lp.w1 * numpy.sqrt(r1(g.z.points, cp)) / mm )
plt.plot(g.z.points, wy_z/mm, 'o')
plt.xlabel('Z, m'); plt.ylabel('\Delta X & Y, mm') 
plt.show()  
    
#Iztyx = ConvertToRealTime(Izt0yx, start_t, delta_t, steps_t2, start_t2)
#Iztyx = Izt0yx

Izyx2p = numpy.sum(numpy.power(Iztyx, 2), axis = 1)


max_intensity2p = numpy.max(Izyx2p[:,lp.Nx//2,:])
plt.imshow(numpy.log(numpy.add(numpy.transpose(Izyx2p[:,lp.Nx//2,:]), max_intensity2p/200 )), cmap='hot', 
           vmin=numpy.log( max_intensity2p/200 ), vmax= numpy.log(max_intensity2p + max_intensity2p/200 ),
           extent = [g.z.start, g.z.end, -lp.size/2/mm, lp.size/2/mm],  aspect=0.00005*lp.Nx/g.z.steps);
plt.show()


max_intensity2p = numpy.max(Izyx2p[:,:,lp.Nx//2])
plt.imshow(numpy.log(numpy.add(numpy.transpose(Izyx2p[:,:,lp.Nx//2]), max_intensity2p/200 )), cmap='hot', 
           vmin=numpy.log( max_intensity2p/200 ), vmax= numpy.log(max_intensity2p + max_intensity2p/200 ),
           extent = [g.z.start, g.z.end, -lp.size/2/mm, lp.size/2/mm],  aspect=0.00005*lp.Nx/g.z.steps);
plt.show()


         
plt.plot(g.z.points/mm, Normalized( Izyx2p[:,lp.Nx//2,lp.Nx//2] ),  'o-')
plt.plot(g.z.points/mm, Normalized( 1/numpy.sqrt(r1(g.z.points, cp)*r2(g.z.points, cp)) ))

k0 = (2*numpy.pi/s.lambda0)
w1 = lp.w1
a = lp.a

plt.plot(g.z.points/mm, Normalized( numpy.sum(Iztyx, axis = 1)[:, lp.Nx//2, lp.Nx//2] ),  'o-')
plt.plot(g.z.points/mm, Normalized( AxiconZ(g.z.points -lp.f + lp.sz/numpy.sqrt(2)) ))
plt.plot(g.z.points/mm, Normalized( 1/numpy.sqrt(r1(g.z.points, cp)) ))
plt.xlabel('Z, mm'); plt.ylabel('I(axis)') 
plt.show()


ifoc = numpy.argmax( Izyx2p[:,lp.Nx//2,lp.Nx//2] )
r = range( int(lp.Nx//(7/3)),  int(lp.Nx//(7/4)) )

xp = g.x.points[((g.x.steps - lp.Nx)//2) : ((g.x.steps + lp.Nx)//2), ]

plt.plot(xp[r]/mm, Normalized( numpy.sum(Iztyx, axis = 1)[ifoc,lp.Nx//2,r] ),  'o-')
plt.plot(xp[r]/mm, scipy.special.jv(0, xp[r]/lp.a)**2 )
plt.plot(xp[r]/mm, Normalized( Izyx2p[ifoc,lp.Nx//2,r] / max(Izyx2p[ifoc,lp.Nx//2,r]) ), 'o-')
plt.xlabel('X, mm'); plt.ylabel('I(foc)') 
plt.show()


#    print('x scale FWHM is ', 
#          1e6*(lp.size)*sum(Ixy2p[ifoc][lp.Nx//2] > numpy.max(Ixy2p[ifoc][lp.Nx//2])*0.5)/len(Ixy2p[ifoc][lp.Nx//2]),
#          'mkm (',   sum(Ixy2p[ifoc][lp.Nx//2] > numpy.max(Ixy2p[ifoc][lp.Nx//2])*0.5), ' points)')