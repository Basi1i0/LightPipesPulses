# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 14:10:38 2018

@author: Basil
"""

from joblib import Parallel, delayed
import multiprocessing
    
# what are your inputs, and what operation do you want to 
# perform on each input. For example...
inputs = range(10) 
def processInput(i):
	return i * i / i**1.99

num_cores = 1
inputs = range(1, 300000) 
   
start_time = timeit.default_timer()
results = Parallel(n_jobs=num_cores)(delayed(processInput)(i) for i in inputs)
elapsed = timeit.default_timer() - start_time
print('\nIt took', elapsed)
    

