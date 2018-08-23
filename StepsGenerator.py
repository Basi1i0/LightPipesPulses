# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 14:52:42 2018

@author: Basil
"""

import collections
import numpy
import matplotlib.pyplot as plt

Spectrum = collections.namedtuple('Steps', ['start', 'end', 'steps', 'delta', 'points'])

def StepsGenerate(start, end, steps):
    delta=(end-start)/steps
    zs = numpy.linspace(start, end, steps, endpoint=False)
    return Spectrum(start,end,steps,delta,zs)

