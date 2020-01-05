import numpy as np
from math import sin,atan2,cos,pi
from time import sleep
import random
import matplotlib.pyplot as plt 
import copy
from environment_defense import Env

class Agent_in_Env():
    def __init__(self):
        self.numer_of_agents = 10
        self.gravity_point = Env().gravity_point
        self.dt = 1
        self.agent_pos_vel = self.agent_initialize()        ## Format -> [x coord, y coord, velocity_x, velocity_y]

################################################
    ### Effect of field on the agents
    def agent_cotrol(self): 

        ## Due to control decison      
        #theta = atan2((agent_pos_vel[1] - self.gravity_point[1]),(agent_pos_vel[0] - self.gravity_point[0]))        #angle of the opponent with the gravity point        
        
            ## TO BE WRITTEN
        
        ## Due to field
        for i in range (self.numer_of_agents):
            field_effect = Env().field_agent_force(self.agent_pos_vel[i])

            self.agent_pos_vel[i][2] = self.agent_pos_vel[i][2] + field_effect[0]*self.dt
            self.agent_pos_vel[i][3] = self.agent_pos_vel[i][3] + field_effect[1]*self.dt

            self.agent_pos_vel[i][0] = self.agent_pos_vel[i][0] + self.agent_pos_vel[i][2]*self.dt
            self.agent_pos_vel[i][1] = self.agent_pos_vel[i][1] + self.agent_pos_vel[i][3]*self.dt

        return self.agent_pos_vel

    def agent_initialize(self):
        
        initial_pos = []
        #random_theta = random.randrange(0,2)
        initial_radius = 6

        for i in range (self.numer_of_agents):
            random_theta = random.randrange(0,2*314)/100
            initial_pos.append([initial_radius*cos(random_theta),initial_radius*sin(random_theta),0,0])

        return initial_pos  

    # def agent_model(self):    
    #     for i in range (self.numer_of_agents):



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