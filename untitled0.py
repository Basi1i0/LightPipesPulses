# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 19:33:08 2018

@author: Basil
"""
#from numpy import *
import matplotlib.pyplot as plt
import gc
import numpy
from joblib import Parallel, delayed

from LightPipes import cm, mm, nm

import timeit


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

def GetIntensityDependenceIntegrated(i, xin, z, z0 = 0):
    return numpy.sum(GetIntensityDependence(i, xin, z, z0))

def ConvertToRealTime(Izt0, start_t, delta_t, steps_t2, start_t2):
    image = numpy.zeros(Izt0.shape[0:1]+(steps_t2,)+Izt0.shape[2::])
    for it in range(0, steps_t2):
        for iz in range(0, Izt0.shape[0]):
            it_shifted = int(round((it*dt2 - iz*delta_z - start_t + start_t2)/delta_t, 0))
#            print('iz=', iz, ', it = ', it, ', it_shifted = ', it_shifted)
            if(it_shifted >= 0 and it_shifted <  Izt0.shape[1]):
                image[iz][it]= Izt0[iz][it_shifted]
    return image

def PlotMovie(Ixz, limits1, limits2, steps, label1 = 'Z', label2 = 'X', aspect = 1):
    max_intensity = numpy.max(Ixz)
    for i in range(0, len(Ixz) ):
        print(steps[i])
        plt.imshow(numpy.log(numpy.add(Ixz[i], max_intensity/100 )), cmap='hot', 
                   vmin=numpy.log( max_intensity/100 ), vmax= numpy.log(max_intensity + max_intensity/100 ), 
                   aspect=aspect, extent=[limits1[0], limits1[1], limits2[0], limits2[1]]);
        plt.xlabel(label1)
        plt.ylabel(label2) 
#        plt.xlim(( start_z, end_z) )
        plt.show()
        
def density_mean(F):
    return numpy.dot(F, numpy.array(range(0, len(F)))) / sum(F)

def density_var(F):
    return numpy.sqrt(numpy.dot(F, (numpy.array(range(0, len(F)))  - density_mean(F) )**2 ) / sum(F) )
    
def SpecGenerate(lambda0, sigma, specN, doplot = False):
    lambdas = numpy.linspace(lambda0 - 3*sigma, lambda0 + 3*sigma, specN) #linspace(800e-9 - 3*sigma, 800e-9 + 3*sigma, 50) #sort(random.uniform(800e-9 - 3*sigma, 800e-9 + 3*sigma, 50)) #random.normal(mean(lambdas), sigma/24, 500) #[800e-9]#
    amps = numpy.exp(-(lambdas - lambda0 )**2 / sigma**2/2 )#[1]
    tau = numpy.mean(lambdas)**2/sigma/(2*numpy.pi) 
    if(doplot):
        plt.plot(lambdas, amps, '.') 
        plt.show()
    return lambdas,amps,sigma,lambda0,specN,tau

def StepsGenerator(start, end, steps):
    delta=(end-start)/steps
    zs = numpy.linspace(start, end, steps, endpoint=False)
    return start,end,steps,delta,zs

def Propogate(x, zs, ts, t0, n_jobs = 1, useRunningTime = False):
    zs_intervals = numpy.split(zs, n_jobs)
    # for n_jobs = 1 use copy.deepcopy(x) as an argument
    Izt0yx = numpy.array(Parallel(n_jobs=n_jobs)(delayed(GetIntensityDependence)(x, ts+t0, zs, True) for zs in zs_intervals))#GetIntensityDependence(x, ts+t0, zs, True)
    return numpy.array([item for sublist in Izt0yx for item in sublist])

def r1(z):
    AdS =alpha/w1/(tau/2.998e-4)
    kw2 = (2*numpy.pi/lambda0 * w1**2)
    D2 = kw2 / f
    Z = z / kw2
    return 1 + D2**2 *(-1 + D2*Z)**2 *(1 + AdS**2)


def r2(z):
    AdS  = alpha/w1/(tau/2.998e-4)
    BdS2 = -b/((tau/2.998e-7)**2)
    kw2 = (2*numpy.pi/lambda0 * w1**2)
    D2 = kw2 / f
    Z = z / kw2
    return (1 + BdS2*D2*(-1 + D2*Z))**2 + (BdS2 - D2*(-1 + D2*Z)*(1 + AdS**2))**2

def PrePropogation(x, f): 
#    x.Axicon(175/180*3.1415, 2, 0, 0)
#    x.Forvard(0.005)
    x.Forvard(f)
    x.Lens(f, 0.00, 0)
    x.Forvard(2*f)
    x.Lens(f, 0.00, 0)
    return 3.*f

if __name__ == '__main__':
    #plt.rcParams["figure.figsize"] = [24, 16]
    x = None; Izt0yx = None; Iztyx = None; Ixtyz = None; Iytxz = None;
    gc.collect()

    lambdas,amps,sigma_s,lambda0,steps_s,tau = SpecGenerate(800*nm, 3*nm, 120, False)
    
    n_jobs = 2
    N = 170
    size = 5.*mm
    f = 100*mm
    alpha = 0.1*mm # mm*ps
    b = 0.8e5 #fs**2
    w1 = f/(2*numpy.pi/lambda0)/(0.1*mm)
    
    x = lpPulse(lambdas, amps, size, N)#LP(size, 800e-9, N)#
    x.GaussAperture(f/(2*numpy.pi/lambda0)/w1, 0*mm, 0, 1)
    x.AddDispersion(b, 2)   
    x.GratingX( alpha*2*numpy.pi*(2.998*1e-4) / (f*lambda0**2) , 800e-9)
   
    t0 = PrePropogation(x, f)
       
    start_z,end_z,steps_z,delta_z,zs = StepsGenerator(0.8*f, 1.2*f, 40)
    start_t,end_t,steps_t,delta_t,ts = StepsGenerator(-3*tau*numpy.sqrt(1 + (alpha/w1/(tau/2.998e-4))**2), 
                                                       3*tau*numpy.sqrt(1 + (alpha/w1/(tau/2.998e-4))**2), 100)#2e-6*alpha*lambda0/w1**3 
        
    start_time = timeit.default_timer()
    Izt0yx = Propogate(x, zs, ts, t0, n_jobs, True)
    elapsed = timeit.default_timer() - start_time
    print('\nIt took', elapsed)
    
#    numpy.save('C://Users//Basil//ResearchData//FourierOptics//GaussianTF//' + 
#               time.strftime("%Y%m%d%H%M%S", time.gmtime()) + '.npy', Izt0yx)
              
#    for different Z
    PlotMovie(numpy.sum(numpy.swapaxes(Izt0yx, 1, 3), axis =2 ), 
              (start_t/mm, end_t/mm), (-size/2/mm, size/2/mm), zs, 'T, mm', 'X, mm', aspect = 1.5*N/steps_t/delta_t*1e-6)
#    PlotMovie(bsum(swapaxes(Izt0yx, 1, 2), axis =3 ), 
#          (start_t/mm, end_t/mm), (-size/2/mm, size/2/mm), zs, 'T, mm', 'Y, mm', aspect =  0.5*N/steps_t/delta_t*1e-6)
        
    wt_z = numpy.array(list(map(density_var, numpy.swapaxes(numpy.swapaxes(Izt0yx, 0, 2), 1, 3)[N//2][N//2]  )))*delta_t
    #wt_zyx = numpy.apply_along_axis(density_var, 1, Izt0yx)*delta_z/3e-4
    wx_z = numpy.array(list(map(density_var, sum(Izt0yx, axis = (1,2)) )))*numpy.sqrt(2)*size/N
    wy_z = numpy.array(list(map(density_var, sum(Izt0yx, axis = (1,3)) )))*numpy.sqrt(2)*size/N

    plt.plot(zs, wt_z/3e-7, 'o')
    plt.plot(zs, numpy.repeat(tau/3e-7/numpy.sqrt(2), len(zs)))
    plt.plot(zs, tau/3e-7/numpy.sqrt(2) * numpy.sqrt(r2(zs)/r1(zs)))
    plt.xlabel('Z, m'); plt.ylabel('\Delta T, fs') 
    plt.ylim( 0, max(wt_z)*1.1/3e-7 ) 
    plt.show()

    plt.plot(zs, wx_z/mm, 'o')
    plt.plot(zs, lambda0/(2*numpy.pi)*f/w1 * numpy.sqrt(r1(zs)) / mm )
#    plt.xlabel('Z, m'); plt.ylabel('\Delta X, mm') 
#    plt.show()  
    plt.plot(zs, wy_z/mm, 'o')
    plt.xlabel('Z, m'); plt.ylabel('\Delta X & Y, mm') 
    plt.show()  
        
#    Iyxzt0 = swapaxes(swapaxes(Izt0yx, 0, 2), 1, 3);
#    wt_xz =[list(map(density_var, Iyxzt0[N//2][i]  )) for i in range(0, N)];
#    for i in range(0, N):
#        plt.plot(wt_xz[i])
#           
#    steps_t2 = int(steps_t)
#    start_t2 = -0.5*end_z; end_t2 = end_z*1.5;
#    dt2 = (end_t2-start_t2)/steps_t2
#    Iztyx = ConvertToRealTime(Izt0yx, start_t, delta_t, steps_t2, start_t2)
    Iztyx = Izt0yx
  

    Ixtyz = numpy.swapaxes(Iztyx, 0, 3) 
#    for different T
#    PlotMovie(Ixtyz[N//2], (start_z, end_z), (-size/2/mm, size/2/mm), ts, 'Z, m', 'X, mm', aspect = 0.05)
        
    Iytxz = numpy.swapaxes(Ixtyz, 0, 2) 
#    #for different T
#    PlotMovie(Iytxz[N//2], (start_z, end_z), (-size/2/mm, size/2/mm), ts, 'Z, m', 'Y, mm', aspect = 0.001)

    

    max_intensity = numpy.max(Ixtyz[N//2])
    min_intensity = numpy.min(Ixtyz[N//2])
    
    Iyz2p = numpy.sum(numpy.power(Iytxz, 1)**2, axis = 1)  # x y z
    Ixz2p = numpy.sum(numpy.power(Ixtyz, 1)**2, axis = 1)  # x y z
    Ixy2p = numpy.swapaxes(numpy.swapaxes(Iyz2p, 2, 1), 1, 0)  # z x y
    
    max_intensity2p = numpy.max(Iyz2p[N//2])
    plt.imshow(numpy.log(numpy.add(Iyz2p[N//2], max_intensity2p/200 )), cmap='hot', 
               vmin=numpy.log( max_intensity2p/200 ), vmax= numpy.log(max_intensity2p + max_intensity2p/200 ),
               extent = [start_z, end_z, -size/2/mm, size/2/mm],  aspect=0.0005*N/steps_z);
    plt.show()
    
    max_intensity2p = numpy.max(Iyz2p[N//2])
    plt.imshow(numpy.log(numpy.add(Ixz2p[N//2], max_intensity2p/200 )), cmap='hot', 
               vmin=numpy.log( max_intensity2p/200 ), vmax= numpy.log(max_intensity2p + max_intensity2p/200 ),
               extent = [start_z, end_z, -size/2/mm, size/2/mm],  aspect=0.0005*N/steps_z);
    plt.show()
    
    
    
             
    plt.plot(zs/mm, Iyz2p[N//2][N//2] / max(Iyz2p[N//2][N//2]), 'o' )
    plt.plot(zs/mm, 1/numpy.sqrt(r1(zs)*r2(zs)) /max(1/numpy.sqrt(r1(zs)*r2(zs))) )
    
    plt.plot(zs/mm, numpy.sum(Iytxz, axis = 1)[N//2][N//2] / max(numpy.sum(Iytxz, axis = 1)[N//2][N//2]), 'o' )
    plt.plot(zs/mm, 1/numpy.sqrt(r1(zs)) / max(1/numpy.sqrt(r1(zs))) )
    
    plt.show()
    print('z scale FWHM is ', 
          1e3*(end_z - start_z)*sum(Iyz2p[N//2][N//2] > numpy.max(Iyz2p[N//2][N//2])*0.5)/len(Iyz2p[N//2][N//2]),
          'mm (',               sum(Iyz2p[N//2][N//2] > numpy.max(Iyz2p[N//2][N//2])*0.5), ' points)')
    
#    ifoc = numpy.argmax(Iyz2p[N//2][N//2])
#    plt.plot(linspace(-size/2, size/2, N)*1e6, Ixy2p[ifoc][N//2]) # z t x y
#    plt.show()
#    print('x scale FWHM is ', 
#          1e6*(size)*sum(Ixy2p[ifoc][N//2] > numpy.max(Ixy2p[ifoc][N//2])*0.5)/len(Ixy2p[ifoc][N//2]),
#          'mkm (',   sum(Ixy2p[ifoc][N//2] > numpy.max(Ixy2p[ifoc][N//2])*0.5), ' points)')