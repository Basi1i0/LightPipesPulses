# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 16:29:45 2018

@author: Basil
"""

import numpy

import StepsGenerator
import SpecGenerator

def DumpParamsToDisk(Izt0yx, s, lp, cp, g, dirname):

    numpy.savetxt(dirname + '\\' + 'lp' + '.txt', lp, newline = '\r\n')
    numpy.savetxt(dirname + '\\' + 'cp' + '.txt', cp, newline = '\r\n')
    
    numpy.save(dirname + '\\' + 'Izt0yx' + '.npy', Izt0yx)
    numpy.save(dirname + '\\' + 's' + '.npy', s)
    numpy.save(dirname + '\\' + 'gt' + '.npy', g.t)
    numpy.save(dirname + '\\' + 'gx' + '.npy', g.x)
    numpy.save(dirname + '\\' + 'gz' + '.npy', g.z)
    return (dirname)


def ReadDumpFromDisk(dirname, LaunchParams, ComputParmas):
    lp_d = LaunchParams( *numpy.loadtxt(dirname + '\\' + 'lp' + '.txt') )
    lp = LaunchParams( int(lp_d.n_jobs), int(lp_d.Nx), int(lp_d.Nz), int(lp_d.Nt), *lp_d[4:])
    cp = ComputParmas( *numpy.loadtxt(dirname + '\\' + 'cp' + '.txt') )
    
    g = StepsGenerator.GridsAll( StepsGenerator.Steps(*numpy.load(dirname + '\\' + 'gx' + '.npy')),
                                 StepsGenerator.Steps(*numpy.load(dirname + '\\' + 'gz' + '.npy')),
                                 StepsGenerator.Steps(*numpy.load(dirname + '\\' + 'gt' + '.npy')) )
    s = SpecGenerator.Spectrum( *numpy.load(dirname + '\\' + 's' + '.npy') )
    Izt0yx = numpy.load(dirname + '\\' + 'Izt0yx' + '.npy')
    
    return Izt0yx,s,lp,cp,g