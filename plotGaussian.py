# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 17:58:37 2018

@author: Basil
"""


import matplotlib.pyplot as plt
import gc
import numpy
from LightPipes import cm, mm, nm


import PropogationFunctions 
import PlotFunctions 
import namedtuplesGaussTF
import SaveLoad
from r1r2 import*

#plt.rcParams["figure.figsize"] = [6, 8]
Izt0yx = None; Iztyx = None; Ixtyz = None; Iytxz = None;
gc.collect()
#
dirname = 'C:\\Users\\Basil\\ResearchData\\FourierOptics\\GaussianTF\\' +\
          '20181030091304_f=0.01_w1=2.55E-04_alpha=2.50E-05_b=0.00E+00'
#          '\\_\\20181026080443_f=0.01_w1=5.09E-04_alpha=4.00E-05_b=6.00E+04\\'

Izt0yx,s,lp,cp,g = SaveLoad.ReadDumpFromDisk(dirname, namedtuplesGaussTF.LaunchParams, 
                                                      namedtuplesGaussTF.ComputParmas)

plt.plot(numpy.linspace(1,100), numpy.linspace(1,100)**2)

#    for different Z
PlotFunctions.PlotMovie(numpy.sum(numpy.swapaxes(Izt0yx, 1, 3), axis =2 ),
                        (g.t.start/mm, g.t.end/mm), (-lp.size/2/mm, lp.size/2/mm),
                        g.z.points, 'T, mm', 'X, mm', aspect = lp.Nx/g.t.steps*0.2)

wt_z,wx_z,wy_z = PropogationFunctions.Ws_z(lp, g, Izt0yx)

plt.plot(g.z.points, wt_z/3e-7, 'o')
plt.plot(g.z.points, numpy.repeat(s.tau/3e-7/numpy.sqrt(2), g.z.steps ))
plt.plot(g.z.points, numpy.repeat(s.tau/3e-7/numpy.sqrt(2)*numpy.sqrt(r2(0, cp)/r1(0, cp)), g.z.steps ) )
plt.plot(g.z.points, 
         s.tau/3e-7/numpy.sqrt(2) * numpy.sqrt(r2(g.z.points, cp)/r1(g.z.points, cp)))
plt.xlabel('Z, m'); plt.ylabel('\Delta T, fs') 
plt.ylim( 0, max(wt_z)*1.1/3e-7 ) 
plt.show()

plt.plot(g.z.points, wx_z/mm, 'o')
plt.plot(g.z.points, 
         s.lambda0/(2*numpy.pi)*lp.f/lp.w1 * numpy.sqrt(r1(g.z.points, cp)) / mm )
#    plt.xlabel('Z, m'); plt.ylabel('\Delta X, mm') 
#    plt.show()  
plt.plot(g.z.points, wy_z/mm, 'o')
plt.xlabel('Z, m'); plt.ylabel('\Delta X & Y, mm') 
plt.show()  
    
#    Iyxzt0 = swapaxes(swapaxes(Izt0yx, 0, 2), 1, 3);
#    wt_xz =[list(map(density_var, Iyxzt0[lp.Nx//2][i]  )) for i in range(0, lp.Nx)];
#    for i in range(0, lp.Nx):
#        plt.plot(wt_xz[i])
#           
#    steps_t2 = int(steps_t)
#    start_t2 = -0.5*end_z; end_t2 = end_z*1.5;
#    dt2 = (end_t2-start_t2)/steps_t2
#    Iztyx = ConvertToRealTime(Izt0yx, start_t, delta_t, steps_t2, start_t2
Iztyx = Izt0yx
  

Ixtyz = numpy.swapaxes(Iztyx, 0, 3) 
#    for different T
#    PlotMovie(Ixtyz[lp.Nx//2], (start_z, end_z), (-lp.size/2/mm, lp.size/2/mm), ts, 'Z, m', 'X, mm', aspect = 0.05)
    
Iytxz = numpy.swapaxes(Ixtyz, 0, 2) 
#    #for different T
#    PlotMovie(Iytxz[lp.Nx//2], (start_z, end_z), (-lp.size/2/mm, lp.size/2/mm), ts, 'Z, m', 'Y, mm', aspect = 0.001)



max_intensity = numpy.max(Ixtyz[lp.Nx//2])
min_intensity = numpy.min(Ixtyz[lp.Nx//2])

Iyz2p = numpy.sum(numpy.power(Iytxz, 1)**2, axis = 1)  # x y z
Ixz2p = numpy.sum(numpy.power(Ixtyz, 1)**2, axis = 1)  # x y z
Ixy2p = numpy.swapaxes(numpy.swapaxes(Iyz2p, 2, 1), 1, 0)  # z x y

max_intensity2p = numpy.max(Iyz2p[lp.Nx//2])
plt.imshow(numpy.log(numpy.add(Iyz2p[lp.Nx//2], max_intensity2p/200 )), cmap='hot', 
           vmin = numpy.log( max_intensity2p/200 ), 
           vmax = numpy.log(max_intensity2p + max_intensity2p/200 ),
           extent = [g.z.start, g.z.end, -lp.size/2/mm, lp.size/2/mm],  
           aspect = 0.005*g.z.steps/lp.Nx);
plt.show()

max_intensity2p = numpy.max(Iyz2p[lp.Nx//2])
plt.imshow(numpy.log(numpy.add(Ixz2p[lp.Nx//2], max_intensity2p/200 )), cmap='hot', 
           vmin = numpy.log( max_intensity2p/200 ), 
           vmax = numpy.log(max_intensity2p + max_intensity2p/200 ),
           extent = [g.z.start, g.z.end, -lp.size/2/mm, lp.size/2/mm],  
           aspect = 0.005*g.z.steps/lp.Nx);
plt.show()



         
plt.plot(g.z.points/mm, Iyz2p[lp.Nx//2][lp.Nx//2] / max(Iyz2p[lp.Nx//2][lp.Nx//2]), 'o' )
plt.plot(g.z.points/mm, (1/numpy.sqrt(r1(g.z.points, cp)*r2(g.z.points, cp)) / \
                         max(1/numpy.sqrt(r1(g.z.points, cp)*r2(g.z.points, cp))) )**1 )
plt.plot(g.z.points/mm, numpy.sum(Iytxz, axis = 1)[lp.Nx//2][lp.Nx//2] / \
                        max(numpy.sum(Iytxz, axis = 1)[lp.Nx//2][lp.Nx//2]), 'o' )
plt.plot(g.z.points/mm, (1/numpy.sqrt(r1(g.z.points, cp)) / \
                         max(1/numpy.sqrt(r1(g.z.points, cp))))**1 )

plt.show()
print('z scale FWHM is ', 
      1e3*(g.z.end - g.z.start)*sum(Iyz2p[lp.Nx//2][lp.Nx//2] > numpy.max(Iyz2p[lp.Nx//2][lp.Nx//2])*0.5)/len(Iyz2p[lp.Nx//2][lp.Nx//2]),
      'mm (',               sum(Iyz2p[lp.Nx//2][lp.Nx//2] > numpy.max(Iyz2p[lp.Nx//2][lp.Nx//2])*0.5), ' points)')

#    ifoc = numpy.argmax(Iyz2p[lp.Nx//2][lp.Nx//2])
#    plt.plot(linspace(-lp.size/2, lp.size/2, lp.Nx)*1e6, Ixy2p[ifoc][lp.Nx//2]) # z t x y
#    plt.show()
#    print('x scale FWHM is ', 
#          1e6*(lp.size)*sum(Ixy2p[ifoc][lp.Nx//2] > numpy.max(Ixy2p[ifoc][lp.Nx//2])*0.5)/len(Ixy2p[ifoc][lp.Nx//2]),
#          'mkm (',   sum(Ixy2p[ifoc][lp.Nx//2] > numpy.max(Ixy2p[ifoc][lp.Nx//2])*0.5), ' points)')