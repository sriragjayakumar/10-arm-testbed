# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 07:59:18 2022

@author: Group-7
"""

import numpy as np
from matplotlib import pyplot as plt

def testbed_plot(distribution):
    # plt.subplot(2,1,1)
    plt.plot([0,12],[0,0],linestyle='--')
    plt.plot(np.arange(10)+1,distribution,'ro',label='$ \ mean \ (\mu$)')
    plt.errorbar(np.arange(10)+1,distribution,yerr=np.ones(10),fmt='none', label='Standard deviation $\ (\sigma$)')
    plt.title('10-armed testbed')
    plt.ylim(min(distribution)-2, max(distribution)+2)
    plt.xlabel('Action')
    plt.ylabel('Reward Distribution')
    plt.legend()
    plt.show()

    
def std_norm_dist_gen(mean,std,k):
    
    # np.random.seed(7)
    distribution = np.random.normal(mean,std,k)
    
    return distribution

def play(epsilon):
    scoreArr = np.zeros((1000, 1))
    for _ in range(2000):
        
        actArr = std_norm_dist_gen(0,1,10)
        valEstimates = np.zeros(10)
        kAction = np.zeros(10)         
        rSum = np.zeros(10) 
        
        for step in range(1000):
            
            
            randomProb = np.random.uniform(low=0, high=1)
            if randomProb < epsilon:
                actionT = np.random.choice(10)    
            else:
                # Greedy Method
                maxAction = max(valEstimates)     # Find max value estimate
                # identify the corresponding action, as array containing only actions with max
                action = np.where(valEstimates == maxAction)[0]
    
                # If multiple actions contain the same value, randomly select an action
                if len(action)>1:

                    actionT = np.random.choice(action)
                else:
                    actionT = action[0]
                    
            reward = std_norm_dist_gen(actArr[actionT],1,1)[0]
            
            kAction[actionT] += 1       # Add 1 to action selection
            rSum[actionT] += reward    # Add reward to sum array
    
            # Calculate new action-value, sum(r)/ka
            valEstimates[actionT] = rSum[actionT]/kAction[actionT]
            
            scoreArr[step]+=reward
                
    return scoreArr/2000

if __name__ == '__main__':
        
        testbed_plot(std_norm_dist_gen(0,1,10))
        
        scoreAvg_0=play(0)
        scoreAvg_01=play(0.1)
        scoreAvg_001=play(0.01)

        
        plt.title("10-Armed TestBed - Average Rewards")
        plt.subplot(2,1,2)
        plt.plot(scoreAvg_0,'r')
        plt.plot(scoreAvg_01,'b')
        plt.plot(scoreAvg_001,'g')
        plt.ylabel('Average Reward')
        plt.xlabel('Plays')
        plt.show()