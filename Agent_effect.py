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


class Env():
    def __init__(self):
        self.arena_size = [100,100]             # [length,breadth] magnitude
        self.arena_diagonal = [[-50,50],[50,-50]]
        self.gravity_point = [0,0]
        # attack or defense
        self.force_direction = +1 
        # regulator is to control the offense and defense
        # relative distance is the distance of the vehicle from the gravity point
        self.regulator = 1
        self.K = 1               # gravitational constant
        self.relative_distance = 1
        self.field_force_magnitude = self.K*(self.regulator)/(self.relative_distance)                     # the force magnitude to regulate the path
        # opponents force
        self.opponent_pos = []
        self.number_of_opponents = 10
        self.opponent_force_magnitude = 0
        self.opponent_radial_velocity = 5
        self.dt = 0.5          ##discreet time in sec

################################################
    ### Effect on the agents
    ## Format of agent_pos -> [x coordinate , y coordinate]
    def field_agent_force(self, agent_pos):       

        theta = atan2((agent_pos[1] - self.gravity_point[1]),(agent_pos[0] - self.gravity_point[0]))        #angle of the opponent with the gravity point        
        force  = self.field_force_magnitude - self.opponent_force_magnitude

        force_x = (self.force_direction)*force*cos(theta)
        force_y = (self.force_direction)*force*sin(theta)

        # agent_pos[2] = agent_pos[2] + force_x*self.dt
        # agent_pos[3] = agent_pos[3] + force_y*self.dt

        # agent_pos[0] = agent_pos[0] + agent_pos[2]*self.dt
        # agent_pos[1] = agent_pos[1] + agent_pos[3]*self.dt

        return force_x,force_y

#################################################
###############****************Opponents section of the code********************#######################

    def opponent_model(self):

        if len(self.opponent_pos) != self.number_of_opponents:         
            for j in range (self.number_of_opponents - len(self.opponent_pos)):
                print("hello",j)
                initial_pos = self.opponent_restart()
                self.opponent_pos.append(initial_pos)
        #else:
        for i in range (len(self.opponent_pos)):
            (v_x,v_y) = self.opponent_control(self.opponent_pos[i])

            #dynamics of the opponent
            self.opponent_pos[i][0] = self.opponent_pos[i][0] + v_x*self.dt 
            self.opponent_pos[i][1] = self.opponent_pos[i][1] + v_y*self.dt 

            #check if the target is nullified
        #opponent_pos_copy = copy.deepcopy(self.opponent_pos)    
        for i in range (len(self.opponent_pos)):    
            nullify_status = self.opponent_check_if_nullified(self.opponent_pos[i])
            if nullify_status == 1:
                self.opponent_pos[i] = 0
                # initial_pos = self.opponent_restart()
                # self.opponent_pos.append(initial_pos)

        self.opponent_pos[:] = (value for value in self.opponent_pos if value != 0)        

        return self.opponent_pos        

    def opponent_check_if_nullified(self,pos):                 ##to check if nullified or not

        nullify_status = 0

        # close to the point of gravity
        if (pos[0] - self.gravity_point[0])**2 + (pos[1] - self.gravity_point[1])**2 <= 2:
            nullify_status = 1
  
        # close to gaurd team

          ###TO BE WRITTEN

        return nullify_status


    def opponent_restart(self):

        random_corner = random.randrange(0,2)
        random_side = random.randrange(0,2)
        if random_side == 0:                
            initial_pos = [self.arena_diagonal[random_corner][0] , random.randrange(-50,50)]            ##Need to change this to more general
        if random_side == 1:                
            initial_pos = [random.randrange(-50,50) , self.arena_diagonal[random_corner][1]] 

        return initial_pos    


    def opponent_control(self,pos):

        theta = atan2((pos[1] - self.gravity_point[1]),(pos[0] - self.gravity_point[0]))        #angle of the opponent with the gravity point
        velocity_x = -(self.force_direction)*self.opponent_radial_velocity*cos(theta)
        velocity_y = -(self.force_direction)*self.opponent_radial_velocity*sin(theta)

        return velocity_x,velocity_y


class Agent_in_Env():
    def __init__(self,size):
        self.arena_size = [100,100]             # [length,breadth] magnitude
        self.arena_diagonal = [[-50,50],[50,-50]]
        self.gravity_point = [0,0]
        # attack or defense
        self.force_direction = +1 
        # regulator is to control the offense and defense
        # relative distance is the distance of the vehicle from the gravity point
        self.regulator = 1
        self.K = 1               # gravitational constant
        self.relative_distance = 1
        self.field_force_magnitude = self.K*(self.regulator)/(self.relative_distance)                     # the force magnitude to regulate the path
        # opponents force
        self.opponent_pos = []
        self.number_of_opponents = 10
        self.opponent_force_magnitude = 0
        self.opponent_radial_velocity = 5
        self.dt = 0.5          ##discreet time in sec

################################################
    ### Effect on the agents
    ## Format of agent_pos -> [x coordinate , y coordinate]
    def field_agent_force(self, agent_pos):       

        theta = atan2((agent_pos[1] - self.gravity_point[1]),(agent_pos[0] - self.gravity_point[0]))        #angle of the opponent with the gravity point        
        force  = self.field_force_magnitude - self.opponent_force_magnitude

        force_x = (self.force_direction)*force*cos(theta)
        force_y = (self.force_direction)*force*sin(theta)

        # agent_pos[2] = agent_pos[2] + force_x*self.dt
        # agent_pos[3] = agent_pos[3] + force_y*self.dt

        # agent_pos[0] = agent_pos[0] + agent_pos[2]*self.dt
        # agent_pos[1] = agent_pos[1] + agent_pos[3]*self.dt

        return force_x,force_y


# this is REINFORCE Agent for GridWorld
class ReinforceAgent:
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
        updates = optimizer.get_updates(self.model.trainable_weights, [],
                                        loss)
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
    while True:
        env.opponent_model()
        for i in range (len(env.opponent_pos)-1):
            plt.plot(env.opponent_pos[i][0],env.opponent_pos[i][1],'r.')
            # plt.show()
        plt.axis([-50,50,-50,50])
        plt.pause(0.01)
        plt.clf()
        sleep(0.0100)

