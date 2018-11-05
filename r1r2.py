# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 18:59:21 2018

@author: Basil
"""

def r1(z, cp):
    Z = z / cp.kw2
    return 1 + cp.D2**2 *(-1 + cp.D2*Z)**2 *(1 + cp.AdS**2)


def r2(z, cp):
    Z = z / cp.kw2
    return (1 + cp.BdS2*cp.D2*(-1 + cp.D2*Z))**2 + \
           (cp.BdS2 - cp.D2*(-1 + cp.D2*Z)*(1 + cp.AdS**2))**2
    
def z0(b, cp, s):
    BdS2 = b/((s.tau/2.998e-7)**2)
    return cp.kw2*BdS2*cp.AdS**2/(cp.D2**2 * ( (1+cp.cp.AdS**2)**2 + BdS2**2))