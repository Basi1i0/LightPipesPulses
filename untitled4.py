# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 15:56:31 2018

@author: Basil
"""
from SpecGenerator import SpecGenerate
from numpy import sqrt, linspace, swapaxes, mean, diff
from PropogationFunctions import density_var

lambdas,amps,sigma_s,lambda0,steps_s,tau  = SpecGenerate(800*nm, 3*nm, 100, False)
b = 5e4 # [fs**2]
tau2 = tau*sqrt(1 + (b/2*(3e-7/tau)**2)**2 )

N = 5
size = 0.02
x = lpPulse(lambdas, amps, size, N)#LP(size, 800e-9, N)#
x.AddDispersion(b**2/2, 4)
#x.AddDispersion(-4000000000, 3)

    

ts = linspace(-20*tau, 20*tau, 100)
intensities = []
fields = []

for t in ts:
    Isum = x.Intensity(0, t)
    field = x.Field(t)
    intensities.append(Isum)
    fields.append(field)
    
#t x y
Iswapped = swapaxes(intensities, 0, 2) #y x tuhg
taum = density_var(Iswapped[N//2][N//2])*mean(diff(ts))

plt.plot(ts, Iswapped[N//2][N//2] / Iswapped[N//2][N//2][len(ts)//2], 'o')

plt.plot(ts, exp( -ts**2/tau**2 ))
plt.plot(ts, exp( -ts**2/taum**2/2 ))

plt.show()


#
#plt.plot(ts, Iswapped[N//2][N//2] -  Iswapped[N//2][N//2][len(ts)//2]*exp(- ts**2/ (2.71*tau)**2/2 ))


#t x y
#Fswapped = swapaxes(fields, 0, 2) #y x t
#plt.plot(ts, Fswapped[N//2][N//2])
#plt.show()
