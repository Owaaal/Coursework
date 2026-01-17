#The below import are all that should be used in this assignment. Additional libraries are not allowed.
import numpy as np 
import math 
import scipy.sparse.csgraph 
import matplotlib.pyplot as plt 
import random 
import argparse
import collections
import sys
import csv

def rrt(map_file, step_size, num_points):

    def read_map_from_file(map_file):
        '''
        Just like the first method, this functions reads a csv file describing a map and returns the map data
        Inputs:
            - filename (string): name of the file to read
        Outputs:
            - map (tuple): A map is a tuple of the form (grid_size, start_pos, goal_pos, [obstacles])
                grid_size is an tuple (length, height) representing the size of the map
                start_pos is a tuple (x, y) representing the x,y coordinates of the start position
                goal_pos is a tuple (x, y) representing the x,y coordinate of the goal position
                obstacles is a list of tuples. Each tuple represents a single  circular obstacle and is of the form (x, y, radius).
                    x is an integer representing the x coordinate of the obstacle
                    y is an integer representing the y coordinate of the obstacle
                    radius is an integer representing the radius of the obstacle
        '''
        def read_file(filename):
            with open(filename, 'r') as file:
                file = csv.reader(file)
                numbers = list(file)
                return numbers
        data_list = read_file(map_file)
        # ^Reades and creates a list ot of data

        #
        map_size = tuple(map(int, data_list[0]))
        start_point = tuple(map(int, data_list[1]))
        end_point = tuple(map(int, data_list[2]))
        def object_values():
            object_list = []
            for ii in data_list[3:6]:
                obj_x = tuple(map(int, ii))
                object_list.append(obj_x)
            return(object_list)
        # ^Turns list into interger and tuple form while seperating key values (eg. start position) from objects
        # Use of map/tuple function to turn values from list of strings to tuple of intergers

        #
        obj_values = list(object_values())
        map_data = tuple([map_size,start_point,end_point, obj_values])
        # ^Formats the values into the correct format
        return(map_data)

    class Node:
        '''Update'''
        def __init__(self, coords, parent=None):
            self.position = coords  # (x, y)
            self.parent = parent
            self.children = []
        
        def add_child(self, child_node):
            self.children.append(child_node)

    class Graph:
        def __init__(self, map_data, step_size, obstacle_data):
            self.length = map_data[0][0]
            self.start = map_data[1]
            self.end = map_data[2]
            self.step_size = step_size
            self.obstacle_data = obstacle_data
            self.trial_points = []
            self.nodes = [(self.start)]
            
        def add_trial_point_and_node(self):
            '''
            This function randomly generates a point on the map
            Inputs
                - map data
            Outputs
                - Random point on map
            '''

            trial_point = self.generate_trial_point()
            self.trial_points.append(trial_point)
            
            closest_node = self.find_closest_node(trial_point)
            trial_point_node = self.step_to_trial_point(closest_node, trial_point)
            self.nodes.append(trial_point_node)
            '''if self.obstacle_check(trial_point_node) == 0:
                self.nodes.append(trial_point_node)'''

        def generate_trial_point(self):
            trial_point_x = random.randint(0, self.length-1)
            trial_point_y = random.randint(0, self.length-1)
            trial_point = (trial_point_x,trial_point_y)
            return trial_point

        def obstacle_check(self, point):   
            point_x = (point[0])
            point_y = (point[1])
            list = []
            '''
            Finds the map size from map_data, then creates a zeros array using the map size
            '''

            def obstacle_finder(obstacle_number):
                '''
                xx
                '''
                obstacle_list = self.obstacle_data
                mid_point_x = (obstacle_list[obstacle_number-1][0])
                mid_point_y = (obstacle_list[obstacle_number-1][1])
                radius = (obstacle_list[obstacle_number-1][2])

                return [[mid_point_x,mid_point_y,radius]]
                #Have to return a list of lists to then extract them

            for i in range(len(self.obstacle_data)):
                obs_data = obstacle_finder(i+1)
                for x, y, radius in obs_data:
                        if math.sqrt((point_x - x)**2 + (point_y - y)**2) <= radius:
                            list.append(1)
                        else:
                            list.append(0)

            return max(list)
            
        def plot_trial_point(self,point_no):
            x = self.trial_points[point_no][0]
            y = self.trial_points[point_no][1]

            xx = self.nodes[point_no][0]
            yy = self.nodes[point_no][1]
            colors = ['green','blue','orange','pink','purple','orange','cyan']
            for ii in range(len(colors)):
                plt.scatter(x,y, color= colors[point_no % len(colors)], marker='+', s=30, linewidths=1)
                plt.scatter(xx,yy, color= colors[point_no % len(colors)], marker='.', s=30, linewidths=0.5)

            if self.obstacle_check((x,y)) == 1:
                plt.scatter(x,y, color= 'black', marker='o', s=30, linewidths=1)

        def find_closest_node(self, trial_point):
            '''
            Finds closest node to the trial point
            Inputs:
                ONLY the random trial point
            Output:
                The coordinates of the closest node that is in bounds
            '''
            node_list = []

            for node in self.nodes:
                direction_vec = np.array(trial_point) - np.array(node)
                distance = math.sqrt(direction_vec[0]**2 + direction_vec[1]**2)
                node_list.append((node,distance))
            
            node_list.sort(key=lambda x: x[1])

            for node in node_list:
                child_x, child_y = self.step_to_trial_point(node[0], (trial_point))
                if 0 <= child_x <= self.length:
                    if 0 <= child_y <= self.length:
                        return (node[0])
            print('defaulting',(node_list[1][0]))
            return (node_list[1][0])
        
        def step_to_trial_point(self, closest_node, trial_point):
            '''
            Inputs:
                coordinates of the closest valid node
                coordinates of the trial point
            Outputs
                coordinates of the new node
            '''
            node_coords = closest_node
            direction_vec = np.array(trial_point) - np.array(node_coords)
            distance_vec = math.sqrt(direction_vec[0]**2 + direction_vec[1]**2)
            # ^Extracts data

            if distance_vec == 0:
                return list(node_coords)

            unit_vec = direction_vec / distance_vec
            x = node_coords[0] + self.step_size * unit_vec[0]
            y = node_coords[0] + self.step_size * unit_vec[1]
            node_array = np.array([int(x),int(y)])
            new_node = tuple(node_array.tolist())

            return new_node

    def plot_map(ax, map_data):
        '''
        This function plots a map given a description of the map
        Inputs:
        ax (matplotlib axis) - the axis the map should be drawn on
        map_data - a tuple describing the map. See definition in read_map_from_file function for details.
        '''
        if map_data:
            start_pos = map_data[1]
            goal_pos = map_data[2]
            obstacles = map_data[3]

            ax.plot(goal_pos[0], goal_pos[1], 'r*')
            ax.plot(start_pos[0], start_pos[1], 'b*')

            for obstacle in obstacles:
                #Obstacle[0] is x position, [1] is y position and [2] is radius
                c_patch = plt.Circle((obstacle[0], obstacle[1]), obstacle[2], color='red')
                ax.add_patch(c_patch)
        else:
            print("No map data provided- have you implemented read_map_from_file?")

    '''
    To do
        make sure node is in graph range - problem with find_closest_node()
        convert node list to tuples?
        plot stuff
    '''

    def rrt_main():
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_axis_off()

        map_data = read_map_from_file(map_file)
        print(map_data)
        plot_map(ax, map_data)
        graph = Graph(map_data, step_size, obstacle_data=map_data[3])
        for ii in range(num_points):
            graph.add_trial_point_and_node()
            graph.plot_trial_point(ii)
        nodes = graph.nodes
        print(nodes)
        plt.show()

    rrt_main()


rrt('map1 copy.csv',step_size=5,num_points=100)