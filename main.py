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
    plt.show()

    
def std_norm_dist_gen(mean,std,k):
    
    distribution = np.random.normal(mean,std,k)
    
    return distribution

def play(epsilon):
    
    scoreArr = np.zeros((1000, 1))
    optArr=np.zeros((1000,1))
    
    for _ in range(2000):
        
        actArr = std_norm_dist_gen(0,1,10)
        valEstimates = np.zeros(10)
        nAction = np.zeros(10)         
        rSum = np.zeros(10) 
        
        for step in range(1000):
                        
            randomProb = np.random.uniform(low=0, high=1)
            
            if randomProb < epsilon:
                actionTaken = np.random.choice(10)    
            else:
                # Greedy Method
                maxAction = max(valEstimates)    
                possibleActions = np.where(valEstimates == maxAction)[0]
    
                if len(possibleActions)>1:

                    actionTaken = np.random.choice(possibleActions)
                else:
                    actionTaken = possibleActions[0]
                    
            reward = std_norm_dist_gen(actArr[actionTaken],1,1)[0]
            
            nAction[actionTaken] += 1       
            rSum[actionTaken] += reward        
            valEstimates[actionTaken] = rSum[actionTaken]/nAction[actionTaken]            
            scoreArr[step]+=reward
                
            if actionTaken == np.argmax(actArr):
                optArr[step] += 1
    
    return scoreArr/2000,optArr/2000

if __name__ == '__main__':
        
        testbed_plot(std_norm_dist_gen(0,1,10))
        
        avgReward_0,optAct_0 = play(0)
        avgReward_01,optAct_01 = play(0.1)
        avgReward_001,optAct_001 = play(0.01)
        
        plt.title("10-Armed TestBed - Average Rewards")
        plt.plot(avgReward_0,'r', label='Ɛ=0')
        plt.plot(avgReward_01,'b', label='Ɛ=0.1')
        plt.plot(avgReward_001,'g', label='Ɛ=0.01')
        plt.ylabel('Average Reward')
        plt.xlabel('Plays')
        plt.legend(['Ɛ=0', 'Ɛ=0.1', 'Ɛ=0.01'])
        plt.show()
        
        plt.title("10-Armed TestBed - % Optimal Action")
        plt.plot(optAct_0*100,'r', label='Ɛ=0')
        plt.plot(optAct_01*100,'b', label='Ɛ=0.1')
        plt.plot(optAct_001*100,'g', label='Ɛ=0.01')
        plt.ylabel('% Optimal Action')
        plt.xlabel('Plays')
        plt.legend(['Ɛ=0', 'Ɛ=0.1', 'Ɛ=0.01'])
        plt.show()