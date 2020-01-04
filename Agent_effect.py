import numpy as np
from math import sin,atan2,cos
from time import sleep
import random
import matplotlib.pyplot as plt 
import copy

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
    def field_radial_force(self):
        force_radial = self.field_force_magnitude - self.opponent_force_magnitude
        return force_radial

    def field_agent_control(self, agent_pos):
        theta = atan2((pos[1] - self.gravity_point[1]),(pos[0] - self.gravity_point[0]))        #angle of the opponent with the gravity point
        velocity_x = -(self.force_direction)*self.opponent_radial_velocity*cos(theta)
        velocity_y = -(self.force_direction)*self.opponent_radial_velocity*sin(theta)

        return velocity_x,velocity_y



#################################################
    ### Opponents section of the code

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

###########################*********************** Above part is the Opponent Section of the code ***************###############################
			
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

