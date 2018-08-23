# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 13:13:18 2018

@author: Basil
"""

import collections
import numpy
import matplotlib.pyplot as plt

Spectrum = collections.namedtuple('Spectrum', ['lambdas', 'amps', 'sigma', 'lambda0', 'specN', 'tau'])

def SpecGenerate(lambda0, sigma, specN, doplot = False):
    lambdas = numpy.linspace(lambda0 - 3*sigma, lambda0 + 3*sigma, specN) #linspace(800e-9 - 3*sigma, 800e-9 + 3*sigma, 50) #sort(random.uniform(800e-9 - 3*sigma, 800e-9 + 3*sigma, 50)) #random.normal(mean(lambdas), sigma/24, 500) #[800e-9]#
    amps = numpy.exp(-(lambdas - lambda0 )**2 / sigma**2/2 )#[1]
    tau = numpy.mean(lambdas)**2/sigma/(2*numpy.pi) 
    if(doplot):
        plt.plot(lambdas, amps, '.') 
        plt.show()
    return Spectrum(lambdas,amps,sigma,lambda0,specN,tau)

