# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 07:59:18 2022

@author: Group-7
"""

import numpy as np

def std_norm_dist_gen(k_arm):
    
    np.random.seed(7)
    distribution = np.random.normal(loc=10,scale=1,size=k_arm)
    
    return distribution