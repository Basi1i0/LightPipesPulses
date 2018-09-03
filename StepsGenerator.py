# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 14:52:42 2018

@author: Basil
"""

import collections
import numpy
import matplotlib.pyplot as plt

Steps = collections.namedtuple('Steps', ['points', 'start', 'end', 'steps', 'delta'])
GridsAll = collections.namedtuple('GridsAll', ['x', 'z', 't'])

def StepsGenerate(start, end, steps):
    delta=(end-start)/steps
    zs = numpy.linspace(start, end, steps, endpoint=False)
    return Steps(zs,start,end,steps,delta)

