"""
@author: Group-7
Srirag Jayakumar
Jishnu Prakash Kunnanath Poduvattil
Akhil Raj
"""
#Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Methods
def eGreedyPolicy(state, qTable, epsilon = 0.1):
    """
    Select action for epsilon greedy policy
    """
    #Decide explore or exploit
    if np.random.random() < epsilon:
        # UP = 0, LEFT = 1, RIGHT = 2, DOWN = 3
        action = np.random.choice(4)
    else:
        # Choose the action with largest Q-value (state value)
        action = np.argmax(qTable[:, state])
    return action
    
def agentWalk(agent, action):
    """
    Move agent with respect to action taken
    """
    # get position of the agent
    (x,y) = agent
    # UP
    if action == 0 and x > 0:
        x = x - 1
    # LEFT
    elif action == 1 and y > 0:
        y = y - 1
    # RIGHT
    elif action == 2 and y < 11:
        y = y + 1
    # DOWN
    elif action == 3 and x < 3:
        x = x + 1
    agent = (x, y)
    return agent

def getReward(state):
    """
    Decides rewards and game state (terminates or continues)
    """
    # game continues
    endGame = False
    # all states except cliff have -1 value
    reward = -1
    # goal state
    if(state == 47):
        endGame = True
        reward = 10
    # cliff
    if(state >= 37 and state <= 46):
        endGame = True
        # Penalize the agent if agent encounters a cliff
        reward = -100
    return reward, endGame

def updateQTable(qTable, state, action, reward, followedStateValue, gammaDiscount = 1, alpha = 0.1):
    """
    Estimates Action value 
    Q(S, A) <- Q(S, A) + [alpha * (reward + (gamma * maxValue(Q(S', A'))) -  Q(S, A) ]
    """
    qValue = qTable[action, state] + alpha * (reward + (gammaDiscount * followedStateValue) - qTable[action, state])
    qTable[action, state] = qValue
    return qTable    

def sarsa(episodes = 500, gammaDiscount = 1, alpha = 0.1, epsilon = 0.1):
    """
    SARSA method implementation
    """
    #Initialize q values as 0
    qTable = np.zeros((4, 48))
    storedSteps, storedRewards = [],[]
    # start iterating through the episodes
    for episode in range(0, episodes):
        agent = (3, 0) # starting from left down corner
        endGame = False
        rewardSum = 0 # cumulative reward of the episode
        totalSteps = 0 # keeps number of iterations untill the end of the game
        # choose action using policy
        state =  12*agent[0] + agent[1]
        action = eGreedyPolicy(state, qTable, epsilon)
        while(endGame == False):
            # move agent to the next state
            agent = agentWalk(agent, action)
            totalSteps += 1
            # observe next state value
            followedState = 12*agent[0] + agent[1]
            # observe reward and determine whether game ends
            reward, endGame = getReward(followedState)
            rewardSum += reward 
            # choose the next action using policy and next state
            followedAction = eGreedyPolicy(followedState, qTable, epsilon)
            # update qTable
            followedStateValue = qTable[followedAction][followedState] # differs from q-learning uses the next action determined by policy
            qTable = updateQTable(qTable, state, action, reward, followedStateValue, gammaDiscount, alpha)
            # update the state and action
            state = followedState
            action = followedAction # differs q-learning both state and action must updated
        storedRewards.append(rewardSum)
        storedSteps.append(totalSteps)
        if(episode > 498):
            print("Agent trained with SARSA after 500 iterations")
    return storedRewards

def qLearning(episodes = 500, gammaDiscount = 1, alpha = 0.1, epsilon = 0.1):
    """
    Q-Learning method implementation
    """
    # initialize all states to 0
    # Terminal state cliff walking ends
    storedRewards = list()
    storedSteps = list()
    qTable = np.zeros((4, 48))
    agent = (3, 0) # starting from left down corner
    # start iterating through the episodes
    for episode in range(0, episodes):
        env = np.zeros((4, 12))
        agent = (3, 0) # starting from left down corner
        endGame = False
        rewardSum = 0 # cumulative reward of the episode
        stepSum = 0 # keeps number of iterations untill the end of the game
        while(endGame == False):
            # get the state from agent's position
            state = 12*agent[0] + agent[1]
            # choose action using epsilon-greedy policy
            action = eGreedyPolicy(state, qTable, epsilon)
            # move agent to the next state
            agent = agentWalk(agent, action)
            stepSum += 1

            # observe next state value
            followedState = 12*agent[0] + agent[1]
            maxFollowedStateValue = np.amax(qTable[:, int(state)])

            # observe reward and determine whether game ends
            reward, endGame = getReward(followedState)
            rewardSum += reward 
            # update qTable
            qTable = updateQTable(qTable, state, action, reward, maxFollowedStateValue, gammaDiscount, alpha)

            # update the state
            state = followedState
        storedRewards.append(rewardSum)
        if(episode > 498):
            print("Agent trained with Q-learning after 500 iterations") 
        storedSteps.append(stepSum)
    return storedRewards
    
def plotRewardSumNormalised(qLearningRewards, sarsaRewards):
    """
    Plots reward sum from SARSA and Q-Learning after normalsing the values
    """
    rewardsQLearning = []
    meanReward = np.array(qLearningRewards).mean()
    stdReward = np.array(qLearningRewards).std()
    count = 0 # used to determine the batches
    batchReward = 0 # accumulate reward for the batch
    for cache in qLearningRewards:
        count = count + 1
        batchReward += cache
        if(count == 10):
            # normalize the sample
            normReward = (batchReward - meanReward)/stdReward
            rewardsQLearning.append(normReward)
            batchReward = 0
            count = 0
            
    rewardsSarsa = []
    meanReward = np.array(sarsaRewards).mean()
    stdReward = np.array(sarsaRewards).std()
    count = 0 # used to determine the batches
    batchReward = 0 # accumulate reward for the batch
    for cache in sarsaRewards:
        count = count + 1
        batchReward += cache
        if(count == 10):
            # normalize the sample
            normReward = (batchReward - meanReward)/stdReward
            rewardsSarsa.append(normReward)
            batchReward = 0
            count = 0      
    # prepare the graph    
    plt.plot(rewardsQLearning, label = "Q-learning")
    plt.plot(rewardsSarsa, label = "SARSA")
    plt.ylabel('Cumulative Rewards')
    plt.xlabel('Batches of Episodes (sample size 10) ')
    plt.title("Q-Learning/SARSA Convergence of Cumulative Reward")
    plt.legend(loc='lower right', ncol=2, mode="expand", borderaxespad=0.)
    plt.show()

if __name__ == '__main__':
    sarsaRewards = sarsa(episodes = 500, gammaDiscount = 1, alpha = 0.1, epsilon = 0.1)
    qLearningRewards = qLearning(episodes = 500, gammaDiscount = 1, alpha = 0.1, epsilon = 0.1)
    plotRewardSumNormalised(qLearningRewards,sarsaRewards)
