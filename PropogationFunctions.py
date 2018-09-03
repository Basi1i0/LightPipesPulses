# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 17:46:15 2018

@author: Basil
"""
import numpy


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


#def ConvertToRealTime(Izt0, start_t, delta_t, steps_t2, start_t2):
#    image = numpy.zeros(Izt0.shape[0:1]+(steps_t2,)+Izt0.shape[2::])
#    for it in range(0, steps_t2):
#        for iz in range(0, Izt0.shape[0]):
#            it_shifted = int(round((it*dt2 - iz*delta_z - start_t + start_t2)/delta_t, 0))
##            print('iz=', iz, ', it = ', it, ', it_shifted = ', it_shifted)
#            if(it_shifted >= 0 and it_shifted <  Izt0.shape[1]):
#                image[iz][it]= Izt0[iz][it_shifted]
#    return image



def density_mean(F):
    return numpy.dot(F, numpy.array(range(0, len(F)))) / sum(F)

def density_var(F):
    return numpy.sqrt(numpy.dot(F, (numpy.array(range(0, len(F)))  - density_mean(F) )**2 ) / sum(F) )

def Ws_z(lp, g, Izt0yx):
    wt_z = numpy.array(list(map(density_var, numpy.swapaxes(numpy.swapaxes(Izt0yx, 0, 2), 1, 3)[lp.Nx//2][lp.Nx//2]  )))*g.t.delta
#wt_zyx = numpy.apply_along_axis(density_var, 1, Izt0yx)*delta_z/3e-4b
    wx_z = numpy.array(list(map(density_var, numpy.sum(Izt0yx, axis = (1,2)) )))*numpy.sqrt(2)*lp.size/lp.Nx
    wy_z = numpy.array(list(map(density_var, numpy.sum(Izt0yx, axis = (1,3)) )))*numpy.sqrt(2)*lp.size/lp.Nx
    
    return wt_z, wx_z, wy_z