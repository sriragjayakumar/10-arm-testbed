"""
@author: Group-7
Srirag Jayakumar
Jishnu Prakash Kunnanath Poduvattil
Akhil Raj
"""
#Importing libraries
import numpy as np
import matplotlib.pyplot as plt

#Methods
def testbedPlot(distribution):
    """
    Plot input distribution
    """
    plt.plot([0,12],[0,0],linestyle='--')
    plt.plot(np.arange(10)+1,distribution,'ro',label='$ \ mean \ (\mu$)')
    plt.errorbar(np.arange(10)+1,distribution,yerr=np.ones(10),fmt='none', label='Standard deviation $\ (\sigma$)')
    plt.title('10-armed testbed')
    plt.ylim(min(distribution)-2, max(distribution)+2)
    plt.xlabel('Action')
    plt.ylabel('Reward Distribution')
    plt.legend()
    plt.show()
    
def genStdNormDist(mean,std,k):
    """
    Generate standard normal distribution
    mean --> mean for distribution
    std --> standard deviation for distribution
    k --> required number of values
    """
    return np.random.normal(mean,std,k)

def play(epsilon):
    """
    Performs 10-armed test bed problem with input epsilon value
    """
    #Expected reward values
    scoreArr = np.zeros((1000, 1))
    #Optimal Action values
    optArr=np.zeros((1000,1))
    #Loop over 2000 times
    for _ in range(2000):
        actArr = genStdNormDist(0,1,10) #Action array
        valEstimates = np.zeros(10) #Value states
        nAction = np.zeros(10) #Number of actions taken
        rSum = np.zeros(10) #Reward sum
        #Loop oveer 1000 steps
        for step in range(1000):
            #Getting random value from unifrom distribution 
            randomProb = np.random.uniform(low=0, high=1)

            #Compare with epsilon
            if randomProb < epsilon:
                #Select random action
                actionTaken = np.random.choice(10)
            else:
                # Greedy Method 
                possibleActions = np.where(valEstimates == max(valEstimates))[0]
                if len(possibleActions)>1:
                    #Select random action if multiple actions present
                    actionTaken = np.random.choice(possibleActions)
                else:
                    actionTaken = possibleActions[0]

            #Get reward for action taken (as mean) from normal distribution
            reward = genStdNormDist(actArr[actionTaken],1,1)[0]
            #Update number of action array            
            nAction[actionTaken] += 1
            #Update sum of rewards
            rSum[actionTaken] += reward
            #Update state value
            valEstimates[actionTaken] = rSum[actionTaken]/nAction[actionTaken]
            #Update action array for each step
            scoreArr[step]+=reward
            
            if actionTaken == np.argmax(actArr):
                #Update optimal action array
                optArr[step] += 1
    #Return average of values
    return scoreArr/2000, optArr/2000

if __name__ == '__main__':
    #Plot Reward distribution for 10-arms
    print("Plotting reward distribution for 10-armed bandid problem")
    testbedPlot(genStdNormDist(0,1,10))
    
    #Greedy method
    print("Performing Greedy method (Ɛ=0)...")
    avgReward_0,optAct_0 = play(0)

    #Ɛ-Greedy method with Ɛ=0.1
    print("Performing Ɛ-Greedy method (Ɛ=0.1)...")
    avgReward_01,optAct_01 = play(0.1)

    #Ɛ-Greedy method with Ɛ=0.01
    print("Performing Ɛ-Greedy method (Ɛ=0.01)...")
    avgReward_001,optAct_001 = play(0.01)
    
    #Plotting average rewards vs plays
    print("Plotting Average rewards...")
    plt.title("10-Armed TestBed - Average Rewards")
    plt.plot(avgReward_0,'r', label='Ɛ=0')
    plt.plot(avgReward_01,'b', label='Ɛ=0.1')
    plt.plot(avgReward_001,'g', label='Ɛ=0.01')
    plt.ylabel('Average Reward')
    plt.xlabel('Plays')
    plt.legend(['Ɛ=0', 'Ɛ=0.1', 'Ɛ=0.01'])
    plt.show()
    
    #Plotting Optimal action values vs plays
    print("Plotting Average rewards...")
    plt.title("10-Armed TestBed - % Optimal Action")
    plt.plot(optAct_0*100,'r', label='Ɛ=0')
    plt.plot(optAct_01*100,'b', label='Ɛ=0.1')
    plt.plot(optAct_001*100,'g', label='Ɛ=0.01')
    plt.ylabel('% Optimal Action')
    plt.xlabel('Plays')
    plt.legend(['Ɛ=0', 'Ɛ=0.1', 'Ɛ=0.01'])
    plt.show()