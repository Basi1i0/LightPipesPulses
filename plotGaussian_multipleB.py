# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 20:01:07 2018

@author: Basil
"""
import os



basedir = 'C:\\Users\\Basil\\ResearchData\\FourierOptics\\GaussianTF\\_\\'

dirs = os.listdir(basedir)
ms = list()
for d in dirs[0:6] :
    Izt0yx,s,lp,cp,g = \
        SaveLoad.ReadDumpFromDisk(basedir + d, namedtuplesGaussTF.LaunchParams, 
                                               namedtuplesGaussTF.ComputParmas)
    wt_z,wx_z,wy_z = PropogationFunctions.Ws_z(lp, g, Izt0yx)
    
#    plt.plot(g.z.points, wt_z/3e-7, 'o')
#    plt.plot(g.z.points, numpy.repeat(s.tau/3e-7/numpy.sqrt(2), g.z.steps ))
#    plt.plot(g.z.points, s.tau/3e-7/numpy.sqrt(2) * numpy.sqrt(r2(g.z.points, cp)/r1(g.z.points, cp)))
#    plt.xlabel('Z, m'); plt.ylabel('\Delta T, fs') 
#    plt.ylim( 0, max(wt_z)*1.1/3e-7 ) 
    
    Iztyx = Izt0yx; Ixtyz = numpy.swapaxes(Iztyx, 0, 3); Iytxz = numpy.swapaxes(Ixtyz, 0, 2) 
    
    Iyz2p = numpy.sum(numpy.power(Iytxz, 1)**2, axis = 1)  # x y z
             
    p = plt.plot(g.z.points/mm, Iyz2p[lp.Nx//2][lp.Nx//2] / max(Iyz2p[lp.Nx//2][lp.Nx//2]), 'o-' )
#    plt.plot(g.z.points/mm, 1/numpy.sqrt(r1(g.z.points, cp)*r2(g.z.points, cp)) / \
#                            max(1/numpy.sqrt(r1(g.z.points, cp)*r2(g.z.points, cp))) * max(Iyz2p[lp.Nx//2][lp.Nx//2] ),
#             '--' , color = p[0].get_color())
   
    ms.append(numpy.array([lp.b, numpy.argmax(Iyz2p[lp.Nx//2][lp.Nx//2]), numpy.max(Iyz2p[lp.Nx//2][lp.Nx//2]), ( 1 + cp.BdS2 / cp.D2 / (1 + cp.AdS**2) )]))
   
plt.xlabel('Z, mm'); plt.ylabel('I2p') 
plt.show()
ms = numpy.array(ms)



plt.plot(ms[:,0], g.z.points[list(map(int, ms[:,1]))] / mm, 'o')
plt.plot(ms[:,0], lp.f * ms[:,3]  / mm)
plt.xlabel('b, fs**2'); plt.ylabel('Z, mm') 
plt.show()

plt.plot(g.z.points[list(map(int, ms[:,1]))] / mm,  ms[:,2] / max(ms[:,2]), 'o-')

