
from environment_defense import Env
import numpy as np
from math import sin,atan2,cos
from time import sleep
import random
import matplotlib.pyplot as plt 
import copy
import pylab
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
from keras import backend as K
from keras.models import load_model
from collections import deque
from keras.layers import Dense, Dropout, Flatten , Input
from keras.models import Sequential
from keras.layers import Conv2D, Conv3D, MaxPooling2D
from keras.models import Model
from keras import optimizers , metrics
from keras.layers.core import Activation
from keras.layers.merge import Concatenate

class Agent_in_Env():
    def __init__(self):
        self.load_model = False
        # actions which agent can do
        self.action_space = [0, 1, 2, 3]
        # get size of state and action
        self.action_size = len(self.action_space)
        self.state_size = 12
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.model = self.build_model()
        self.optimizer = self.optimizers()
        self.states, self.actions, self.rewards = [], [], []

        # if self.load_model:
        #     self.model.load_weights('./save_model/reinforce_trained.h5')

    # state is input and probability of each action(policy) is output of network
    def build_model(self):
        
        inputA=Input(shape=(self.state_size,))       
        x_1= Dense(24,activation="relu")(inputA)
        x1 = Dense(16,activation="relu")(x_1)
        #x1=Flatten()(x1)
        
        x1 = Dense(16,activation="relu")(x1)
        x1 = Dense(8,activation="relu")(x1)
        
        # inputB=Input(shape=(self.input_shape_B))
        # x_2=Conv2D(2,(1,1),activation="relu")(inputB)        
        # x2=Conv2D(1,(3,3),activation="relu")(x_2)         
        # x2=Flatten()(x2)
        
        x=Concatenate(axis=-1)([x1,x_1])
        x=Dense(24)(x)
        x=Activation("softmax")(x)
        x=Dense(16)(x)
        x=Activation("softmax")(x)
        x=Dense(self.action_size)(x)
        x=Activation("softmax")(x)
        model=Model(inputs=inputA,outputs=x)
        
        # model = Sequential()
        # model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        # model.add(Dense(24, activation='relu'))
        # model.add(Dense(24, activation='tanh'))
        # model.add(Dense(24, activation='relu'))
        # model.add(Dense(24, activation='tanh'))
        # model.add(Dense(self.action_size, activation='softmax'))
        # model.summary()
        return model

    # create error function and training function to update policy network
    def optimizers(self):
        action = K.placeholder(shape=[None, 4])
        discounted_rewards = K.placeholder(shape=[None, ])

        # Calculate cross entropy error function
        action_prob = K.sum(action * self.model.output, axis=1)
        cross_entropy = K.log(action_prob) * discounted_rewards
        loss = -K.sum(cross_entropy)

        # create training function
        optimizer = Adam(lr=self.learning_rate)
        updates = optimizer.get_updates(loss,self.model.trainable_weights)
        train = K.function([self.model.input, action, discounted_rewards], [],
                           updates=updates)

        return train

    # get action from policy network
    def get_action(self, state):
        policy = self.model.predict(state)[0]
        #print(policy)
        return np.random.choice(self.action_size, 1, p=policy)[0]

    # calculate discounted rewards
    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.discount_factor + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    # save states, actions and rewards for an episode
    def append_sample(self, state, action, reward):
        self.states.append(state[0])
        self.rewards.append(reward)
        act = np.zeros(self.action_size)
        act[action] = 1
        self.actions.append(act)

    # update policy neural network
    def train_model(self):
        discounted_rewards = np.float32(self.discount_rewards(self.rewards))
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= 0.00001 + np.std(discounted_rewards)

        self.optimizer([self.states, self.actions, discounted_rewards])
        self.states, self.actions, self.rewards = [], [], []



if __name__ == '__main__':   
    env = Env()
    #agents = Agent_in_Env()
    while True:
        env.opponent_model()
        env.agent_cotrol()
        for i in range (len(env.opponent_pos)-1):
            plt.plot(env.opponent_pos[i][0],env.opponent_pos[i][1],'r.')
            # plt.show()
        for i in range (10):
            plt.plot(env.agent_pos_vel[i][0],env.agent_pos_vel[i][1],'g.')    
        plt.axis([-50,50,-50,50])
        plt.pause(0.01)
        plt.clf()
        sleep(0.0100)