# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 07:59:18 2022

@author: Group-7
"""

import numpy as np
from matplotlib import pyplot as plt

def testbed_plot(distribution):
    plt.plot([0,12],[0,0],linestyle='--')
    plt.plot(np.arange(10)+1,distribution,'ro',label='$ \ mean \ (\mu$)')
    plt.errorbar(np.arange(10)+1,distribution,yerr=np.ones(10),fmt='none', label='Standard deviation $\ (\sigma$)')
    plt.title('10-armed testbed')
    plt.ylim(min(distribution)-2, max(distribution)+2)
    plt.xlabel('Action')
    plt.ylabel('Reward Distribution')
    plt.legend()

    
def std_norm_dist_gen(k_arm):
    
    np.random.seed(7)
    distribution = np.random.normal(loc=0,scale=1,size=k_arm)
    
    return distribution



if __name__ == '__main__':
    testbed_plot(std_norm_dist_gen(10))
