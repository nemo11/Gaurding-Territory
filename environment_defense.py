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
        self.K = 0.1               # gravitational constant
        self.relative_distance = 1
        self.field_force_magnitude = self.K*(self.regulator)/(self.relative_distance)**2                     # the force magnitude to regulate the path
        # opponents force
        self.opponent_pos = []
        self.number_of_opponents = 10
        self.opponent_force_magnitude = 0
        self.opponent_radial_velocity = 5
        self.dt = 0.5          ##discreet time in sec
        self.numer_of_agents = 10
        self.agent_pos_vel = self.agent_initialize()        ## Format -> [x coord, y coord, velocity_x, velocity_y]


###############**************** Agents section of the code ********************#######################

    def agent_cotrol(self): 

        ## Due to control decison          
        
            ## TO BE WRITTEN
        
        ## Due to field
        for i in range (self.numer_of_agents):
            field_effect = self.field_agent_force(self.agent_pos_vel[i])

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

    ## Effect on the agents
    ## Format of agent_pos -> [x coordinate , y coordinate]
    def field_agent_force(self, agent_pos):       

        theta = atan2((agent_pos[1] - self.gravity_point[1]),(agent_pos[0] - self.gravity_point[0]))        #angle of the opponent with the gravity point        
        force  = self.field_force_magnitude - self.opponent_force_magnitude

        force_x = (self.force_direction)*force*cos(theta)
        force_y = (self.force_direction)*force*sin(theta)

        return force_x,force_y

###############****************Opponents section of the code********************#######################

    def opponent_model(self):

        if len(self.opponent_pos) != self.number_of_opponents:         
            for j in range (self.number_of_opponents - len(self.opponent_pos)):
                initial_pos = self.opponent_restart()
                self.opponent_pos.append(initial_pos)

        for i in range (len(self.opponent_pos)):
            (v_x,v_y) = self.opponent_control(self.opponent_pos[i])
            #dynamics of the opponent
            self.opponent_pos[i][0] = self.opponent_pos[i][0] + v_x*self.dt 
            self.opponent_pos[i][1] = self.opponent_pos[i][1] + v_y*self.dt 

        #check if the target is nullified  
        for i in range (len(self.opponent_pos)):    
            nullify_status = self.opponent_check_if_nullified(self.opponent_pos[i])
            if nullify_status == 1:
                self.opponent_pos[i] = 0

        self.opponent_pos[:] = (value for value in self.opponent_pos if value != 0)        

        return self.opponent_pos        

    def opponent_check_if_nullified(self,pos):                 ##to check if nullified or not

        nullify_status = 0

        # close to the point of gravity
        if (pos[0] - self.gravity_point[0])**2 + (pos[1] - self.gravity_point[1])**2 <= 2:
            nullify_status = 1
  
        # close to gaurd team
        for i in range (self.numer_of_agents):
            if (pos[0] - self.agent_pos_vel[i][0])**2 + (pos[1] - self.agent_pos_vel[i][1])**2 <= 30:
                print("hello")
                nullify_status = 1
                break

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