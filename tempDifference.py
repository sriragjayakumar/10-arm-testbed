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
def eGreedyPolicy(state, q_table, epsilon = 0.1):
    """
    Select action for epsilon greedy policy
    """
    #Decide explore or exploit
    if np.random.random() < epsilon:
        # UP = 0, LEFT = 1, RIGHT = 2, DOWN = 3
        action = np.random.choice(4)
    else:
        # Choose the action with largest Q-value (state value)
        action = np.argmax(q_table[:, state])
    return action
    
def agentWalk(agent, action):
    """
    Move agent with respect to action taken
    """
    # get position of the agent
    (posX , posY) = agent
    # UP
    if action == 0 and posX > 0:
        posX = posX - 1
    # LEFT
    if action == 1 and posY > 0:
        posY = posY - 1
    # RIGHT
    if action == 2 and posY < 11:
        posY = posY + 1
    # DOWN
    if action == 3 and posX < 3:
        posX = posX + 1
    agent = (posX, posY)
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

def updateQTable(q_table, state, action, reward, next_state_value, gamma_discount = 1, alpha = 0.1):
    """
    Estimates Action value
    Q(S, A) <- Q(S, A) + [alpha * (reward + (gamma * maxValue(Q(S', A'))) -  Q(S, A) ]
    """
    q_value = q_table[action, state] + alpha * (reward + (gamma_discount * next_state_value) - q_table[action, state])
    q_table[action, state] = q_value
    return q_table    

def sarsa(num_episodes = 500, gamma_discount = 1, alpha = 0.1, epsilon = 0.1):
    """
    SARSA method implementation
    """
    #Initialize q values as 0
    q_table = np.zeros((4, 48))
    step_cache, reward_cache = [],[]
    # start iterating through the episodes
    for episode in range(0, num_episodes):
        agent = (3, 0) # starting from left down corner
        endGame = False
        rewardSum = 0 # cumulative reward of the episode
        totalSteps = 0 # keeps number of iterations untill the end of the game
        # choose action using policy
        state =  12*agent[0] + agent[1]
        action = eGreedyPolicy(state, q_table, epsilon)
        while(endGame == False):
            # move agent to the next state
            agent = agentWalk(agent, action)
            totalSteps += 1
            # observe next state value
            next_state = 12*agent[0] + agent[1]
            # observe reward and determine whether game ends
            reward, endGame = getReward(next_state)
            rewardSum += reward 
            # choose next_action using policy and next state
            next_action = eGreedyPolicy(next_state, q_table, epsilon)
            # update q_table
            next_state_value = q_table[next_action][next_state] # differs from q-learning uses the next action determined by policy
            q_table = updateQTable(q_table, state, action, reward, next_state_value, gamma_discount, alpha)
            # update the state and action
            state = next_state
            action = next_action # differs q_learning both state and action must updated
        reward_cache.append(rewardSum)
        step_cache.append(totalSteps)
        if(episode > 498):
            print("Agent trained with SARSA after 500 iterations")
    return q_table, reward_cache, step_cache

def qLearning(num_episodes = 500, gamma_discount = 1, alpha = 0.1, epsilon = 0.1):
    """
    Q-Learning method implementation
    """
    # initialize all states to 0
    # Terminal state cliff_walking ends
    reward_cache = list()
    step_cache = list()
    q_table = np.zeros((4, 48))
    agent = (3, 0) # starting from left down corner
    # start iterating through the episodes
    for episode in range(0, num_episodes):
        env = np.zeros((4, 12))
        agent = (3, 0) # starting from left down corner
        endGame = False
        rewardSum = 0 # cumulative reward of the episode
        stepSum = 0 # keeps number of iterations untill the end of the game
        while(endGame == False):
            # get the state from agent's position
            state = 12*agent[0] + agent[1]
            # choose action using epsilon-greedy policy
            action = eGreedyPolicy(state, q_table, epsilon)
            # move agent to the next state
            agent = agentWalk(agent, action)
            stepSum += 1

            # observe next state value
            next_state = 12*agent[0] + agent[1]
            max_next_state_value = np.amax(q_table[:, int(state)])

            # observe reward and determine whether game ends
            reward, endGame = getReward(next_state)
            rewardSum += reward 
            # update q_table
            q_table = updateQTable(q_table, state, action, reward, max_next_state_value, gamma_discount, alpha)

            # update the state
            state = next_state
        reward_cache.append(rewardSum)
        if(episode > 498):
            print("Agent trained with Q-learning after 500 iterations") 
        step_cache.append(stepSum)
    return q_table, reward_cache, step_cache
    
def plotRewardSumNormalised(reward_cache_qlearning, reward_cache_SARSA):
    """
    Plots reward sum from SARSA and Q-Learning after normalsing the values
    """
    cum_rewards_q = []
    rewards_mean = np.array(reward_cache_qlearning).mean()
    rewards_std = np.array(reward_cache_qlearning).std()
    count = 0 # used to determine the batches
    cur_reward = 0 # accumulate reward for the batch
    for cache in reward_cache_qlearning:
        count = count + 1
        cur_reward += cache
        if(count == 10):
            # normalize the sample
            normalized_reward = (cur_reward - rewards_mean)/rewards_std
            cum_rewards_q.append(normalized_reward)
            cur_reward = 0
            count = 0
            
    cum_rewards_SARSA = []
    rewards_mean = np.array(reward_cache_SARSA).mean()
    rewards_std = np.array(reward_cache_SARSA).std()
    count = 0 # used to determine the batches
    cur_reward = 0 # accumulate reward for the batch
    for cache in reward_cache_SARSA:
        count = count + 1
        cur_reward += cache
        if(count == 10):
            # normalize the sample
            normalized_reward = (cur_reward - rewards_mean)/rewards_std
            cum_rewards_SARSA.append(normalized_reward)
            cur_reward = 0
            count = 0      
    # prepare the graph    
    plt.plot(cum_rewards_q, label = "q_learning")
    plt.plot(cum_rewards_SARSA, label = "SARSA")
    plt.ylabel('Cumulative Rewards')
    plt.xlabel('Batches of Episodes (sample size 10) ')
    plt.title("Q-Learning/SARSA Convergence of Cumulative Reward")
    plt.legend(loc='lower right', ncol=2, mode="expand", borderaxespad=0.)
    plt.show()

if __name__ == '__main__':
    q_table_SARSA, reward_cache_SARSA, step_cache_SARSA = sarsa()
    q_table_qlearning, reward_cache_qlearning, step_cache_qlearning = qLearning()
    plotRewardSumNormalised(reward_cache_qlearning,reward_cache_SARSA)
