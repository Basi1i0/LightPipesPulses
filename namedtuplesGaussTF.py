# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 18:52:53 2018

@author: Basil
"""

import collections

LaunchParams = collections.namedtuple('LaunchParams', ['n_jobs', 'Nx', 'Nz', 'Nt', 'size', 'f', 'w1', 'alpha', 'b'])
ComputParmas = collections.namedtuple('ComputParmas', ['kw2', 'AdS', 'BdS2', 'D2'])