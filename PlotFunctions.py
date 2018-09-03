# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 17:47:50 2018

@author: Basil
"""
import numpy
import matplotlib.pyplot as plt

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
        