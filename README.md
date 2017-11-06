# Reinforcement-Learning-Agent
This project used the openai gym environment to train a neural network for playing openai games. https://www.linkedin.com/pulse/create-self-learning-ai-deep-reinforcement-shubhankar-mitra/

# Article Text below
## Article Link: https://www.linkedin.com/pulse/create-self-learning-ai-deep-reinforcement-shubhankar-mitra/
So I recently got introduced to the apenai (https://gym.openai.com/) gym. They have provided the environment to run simple 2-D games in python with programmatic inputs and state output. This makes it possible to run simulations while controlling the game agent and observing the game environment. In this tutorial, I will go through the code and concepts of reinforcement learning to train a deep neural network to play a simple CartPole game. If you want to directly jump to the code: https://github.com/shubhankar90/Reinforcement-Learning-Agent/blob/master/openai-player.py .


CartPole-v0

The objective of this game is to hold the pole straight for as long as possible by controlling the movement of the cart. If the pole tilts by 15 degrees or the cart reaches the end, the game is over. Each time the stick does not fall over, we earn a reward.

Google Deep Q-learning

We will train a feedforward neural network to learn the future possible reward obtained at particular state on choosing a particular action. This is based on Google's deep Q-learning algorithm. (Link: https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)

Q-Learning and Bellman equation

The main objective of Q learning is to learn a matrix Q which gives us a value of expected reward when we choose an action at a particular state of the world and assume the optimal action is taken from the next state. For most environments, the number of states can become huge such that a memory based approach of creating a Q matrix becomes impractical. This is where neural networks come into the picture. They are able to learn this matrix representation given enough data generated through trials conducted in the environment. So now we need a way to create a dataset with input as states and output as the future reward on taking a particular action. Bellman equation can help us in that. This states that the expected long-term reward for a given action is equal to the immediate reward from the current action combined with the expected reward from the best future action taken at the following state. More details: https://en.wikipedia.org/wiki/Q-learning or https://neuro.cs.ut.ee/demystifying-deep-reinforcement-learning/

Architecture

So the overall architecture consists of conducting experiments in the environment to record the effect and training the neural network with the data being collected through these experiments. We will be going through multiple game iterations. Each game ends when either the pole tilts by15 degrees or the cart reaches the boundary of the environment or the count of rewards reaches 1000 (our network becomes a pro!). In each step of a single game we will be randomly selecting an action but after a while, once our neural network is intelligent enough we will be asking the neural network to choose the action (but we still will be choosing a random action based on a low probability to record new experiences). Note, the neural network takes input as the state values and returns the future possible reward for each possible action. At each step we will be storing our state, the action we took, the next environment state and reward in a memory queue. After each complete game, we will be retraining the model using a random batch of samples from our memory queue. Choosing a random batch helps us in avoiding correlation among our samples and avoids training on only the recent memory.

Let's start building:

I will be using Keras (with theano backend) for the deep neural network part. The code is in Python. The openai gym can be installed via pip. This is the main part of the code. It creates the environment, initializes the neural network, runs the games, calls the function to choose action, fills the memory with experience and at the end of each run calls the function to train the neural network. Uncomment the env.render() line to see the AI in action. Rendering will slows down the execution, though.

#importing libraries
﻿import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K

﻿#number of game trials
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
Neural Network architecture

This function returns a feedforward neural network with the input layer neurons equal to the number of elements in the state and output layer neural equal to the number of possible actions.

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
The function below chooses either a random action for exploration or the optimal action given by our neural network. The probability of choosing a random action is given by epsilon initialized in the first code snippet and passed to this function argument of randomSelectTrigger.

#Suggest Action
def suggestAction(state, action_size, randomSelectTrigger, model):
    #choosing a random action with randomSelectTrigger probability
    if np.random.rand() <= randomSelectTrigger:
        #random action select
        return [random.randrange(action_size),0]
    else:
        #Use neural network to select action
        return [np.argmax(model.predict(state)),1]
Experience Replay

This is the part which trains the model from experiences stored in memory. This step is called experience replay in the deep Q network google paper: https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

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
Results

On running the complete code (https://github.com/shubhankar90/Reinforcement-Learning-Agent/blob/master/openai-player.py), the network will initially randomly select the action and keep recording the observations. after some games, most of the action will be suggested from the neural network.

Initially, it will only be able to get rewards around 20-30, which is expected of random guessing. After playing the 200th game it should start to get a score of above 100.

Game finished: 88/5000, score: 23.0, % of DNN action: 0.5, epsilon: 0.7
Game finished: 89/5000, score: 21.0, % of DNN action: 0.2, epsilon: 0.7
Game finished: 90/5000, score: 53.0, % of DNN action: 0.27, epsilon: 0.7
Game finished: 91/5000, score: 20.0, % of DNN action: 0.26, epsilon: 0.69
Game finished: 92/5000, score: 31.0, % of DNN action: 0.33, epsilon: 0.69
Game finished: 93/5000, score: 13.0, % of DNN action: 0.42, epsilon: 0.69
Game finished: 94/5000, score: 13.0, % of DNN action: 0.25, epsilon: 0.69
Game finished: 95/5000, score: 18.0, % of DNN action: 0.35, epsilon: 0.68
Game finished: 96/5000, score: 25.0, % of DNN action: 0.21, epsilon: 0.68
Game finished: 97/5000, score: 25.0, % of DNN action: 0.33, epsilon: 0.68



Around the 400th game, you should start seeing scores above 400. Sometimes it even reaches scores of 1000 (which is the maximum) at this point.

Game finished: 458/5000, score: 239.0, % of DNN action: 0.88, epsilon: 0.16
Game finished: 459/5000, score: 305.0, % of DNN action: 0.86, epsilon: 0.16
Game finished: 460/5000, score: 245.0, % of DNN action: 0.84, epsilon: 0.16
Game finished: 461/5000, score: 187.0, % of DNN action: 0.87, epsilon: 0.16
Game finished: 462/5000, score: 187.0, % of DNN action: 0.83, epsilon: 0.16
Game finished: 463/5000, score: 227.0, % of DNN action: 0.84, epsilon: 0.16
Game finished: 464/5000, score: 174.0, % of DNN action: 0.88, epsilon: 0.16
Game finished: 465/5000, score: 170.0, % of DNN action: 0.8, epsilon: 0.16
Game finished: 466/5000, score: 257.0, % of DNN action: 0.84, epsilon: 0.15
Game finished: 467/5000, score: 588.0, % of DNN action: 0.85, epsilon: 0.15
Game finished: 468/5000, score: 546.0, % of DNN action: 0.88, epsilon: 0.15
Game finished: 469/5000, score: 445.0, % of DNN action: 0.83, epsilon: 0.15
Game finished: 470/5000, score: 399.0, % of DNN action: 0.85, epsilon: 0.15
Game finished: 471/5000, score: 591.0, % of DNN action: 0.85, epsilon: 0.15
Game finished: 472/5000, score: 719.0, % of DNN action: 0.87, epsilon: 0.15
Game finished: 458/5000, score: 239.0, % of DNN action: 0.88, epsilon: 0.16
Game finished: 459/5000, score: 305.0, % of DNN action: 0.86, epsilon: 0.16
Game finished: 460/5000, score: 245.0, % of DNN action: 0.84, epsilon: 0.16
Game finished: 461/5000, score: 187.0, % of DNN action: 0.87, epsilon: 0.16
Game finished: 462/5000, score: 187.0, % of DNN action: 0.83, epsilon: 0.16
Game finished: 463/5000, score: 227.0, % of DNN action: 0.84, epsilon: 0.16
Game finished: 464/5000, score: 174.0, % of DNN action: 0.88, epsilon: 0.16
Game finished: 465/5000, score: 170.0, % of DNN action: 0.8, epsilon: 0.16
Game finished: 466/5000, score: 257.0, % of DNN action: 0.84, epsilon: 0.15
Game finished: 467/5000, score: 588.0, % of DNN action: 0.85, epsilon: 0.15
Game finished: 468/5000, score: 546.0, % of DNN action: 0.88, epsilon: 0.15
Game finished: 469/5000, score: 445.0, % of DNN action: 0.83, epsilon: 0.15
Game finished: 470/5000, score: 399.0, % of DNN action: 0.85, epsilon: 0.15
Game finished: 471/5000, score: 591.0, % of DNN action: 0.85, epsilon: 0.15
Game finished: 472/5000, score: 719.0, % of DNN action: 0.87, epsilon: 0.15
On some runs, it might only reach a maximum score of 150-200. There are quite a few improvements which you will find on reading up from more sources, which stabilize the algorithm. Reinforcement learning is quite an interesting field and is in active research. The pace of new discoveries is accelerating. Hope this article kindles your interest. Do comment and let me know if you find this interesting.

