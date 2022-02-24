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

    
def std_norm_dist_gen(mean,std,k):
    
    # np.random.seed(7)
    distribution = np.random.normal(mean,std,k)
    
    return distribution



if __name__ == '__main__':
    
    
    for _ in range(2000):
        
        actArr = std_norm_dist_gen(0,1,10)
        valEstimates = np.zeros(10)
        kAction = np.zeros(10)         
        rSum = np.zeros(10) 
        scoreArr = np.zeros((1000, 1))
        for step in range(1000):
            
            maxAction = np.argmax(valEstimates)     # Find max value estimate
            # identify the corresponding action, as array containing only actions with max
            action = np.where(valEstimates == np.argmax(valEstimates))[0]

            # If multiple actions contain the same value, randomly select an action
            if len(action) == 0:
                actionT = maxAction
            else:
                actionT = np.random.choice(action)
            reward = std_norm_dist_gen(actArr[actionT],1,1)[0]
            
            kAction[actionT] += 1       # Add 1 to action selection
            rSum[actionT] += reward    # Add reward to sum array
    
            # Calculate new action-value, sum(r)/ka
            valEstimates[actionT] = rSum[actionT]/kAction[actionT]
            
            scoreArr[step]=reward