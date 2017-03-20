# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 18:35:55 2017

@author: alienware
"""
#importing libraries
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K



#Creating the model
def createModel(inp_size, out_size):
    model = Sequential()
    # Input layer with 64 nodes
    model.add(Dense(64, input_dim=inp_size, activation='tanh')) 
    # Hidden layer with 128 nodes
    model.add(Dense(128, activation='tanh'))
    # Hidden layer with 128 nodes
    model.add(Dense(128, activation='tanh'))
    #Output layer with number of neurons equal to number of states
    model.add(Dense(out_size, activation='linear'))
    # Create the model using mean square error as the final loss
    model.compile(loss= 'mse',optimizer='adam')
    return model

#Suggest Action
def suggestAction(state, action_size, randomSelectTrigger, model):
    #choosing a random action with randomSelectTrigger probability
    if np.random.rand() <= randomSelectTrigger:
        #random action select
        return [random.randrange(action_size),0]
    else:
        #Use neural network to select action
        return [np.argmax(model.predict(state)),1]

def retrainModel(model, data, action_size, batch_size=50):
    #Get number of obsertations to train the model
    batch_size = min(batch_size, len(data))
    #Select random observations
    minibatch = random.sample(data, batch_size)
    X = np.zeros((batch_size, len(minibatch[0][0][0])))
    Y = np.zeros((batch_size, action_size))
    for i in range(batch_size):
        state, action, reward, next_state, done = minibatch[i]
        #implement the bellman equation
        target = model.predict(state)[0]
        if done:
            target[action] = 0
        else:
            #Calculate future reward
            a = np.argmax(model.predict(next_state)[0])
            t = model.predict(next_state)[0]
            target[action] = reward + .9 * t[a]
            
        X[i], Y[i] = state, target
    #fit the model
    model.fit(X, Y, nb_epoch=1, verbose=0)
    return model
    
    
if __name__ == "__main__":
    #number of game trials
    trials = 5000
    #number of game steps in each trial
    steps = 1000
    #initialize environment
    env = gym.make('CartPole-v0')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    action_model = createModel(state_size,action_size)
    memory = deque(maxlen=50000)
    #Probabilty of choosing a random action
    epsilon = 1.0
    #Minimum probabilty of choosing a random action
    e_min = .05
    #Amount to reduce the probability of random selection
    self_decay = .996
    env = gym.make('CartPole-v0')
    #Iterating over the games
    for trial in range(trials):
        #reset of the game
        strt_state = env.reset()
        strt_state = np.reshape(strt_state, [1, state_size])
        curr_state = strt_state
        
        DNN_action=0
        rewards=0
        for step in range(steps):
            #Uncomment env.render to see the action. Slows down the execution.
            #env.render()
            action, type_action = suggestAction(curr_state, action_size, epsilon, action_model)
            DNN_action = DNN_action+type_action
            #passing the chosen action to env to get the next state and reward
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            #saving observatipn to memory
            memory.append((curr_state,action,reward,next_state,done))
            rewards = rewards+reward
            curr_state = next_state
            if done or step==999:
                print("Game finished: {}/{}, score: {}, % of DNN action: {:.2}, epsilon: {:.2}"
                        .format(trial, trials, rewards, float(DNN_action/step), float(epsilon)))
                break
        #train the model after each game    
        action_model = retrainModel(action_model, list(memory),action_size)
        #reduce the probability of randomly selecting action
        if epsilon > e_min:
            epsilon *= self_decay
        
    