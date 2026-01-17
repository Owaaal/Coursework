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
        def __init__(self, coords, parent=None):
            '''
            This class adds nodes to the graph
            Inputs
                - Node we want to add coordinates (x,y)
            Output
                - A node with attached parents and room for children
                - .coords function that we can call to find node coords
                - intially set parents to none, so we know when we have hit the start node (useful in the future!)
            '''
            self.coords = coords
            self.parent = parent
            self.children = []
        
        def add_child(self, child_node):
            '''
            This function add children to a node
            Inputs:
                - Child node coordinates
            Output:
                - Child node added to parent node
            '''
            self.children.append(child_node)

    class Graph:
        def __init__(self, map_data, step_size, obstacle_data):
            '''
            This is the main object and stores the tree data
            All points should be explainable by name, nodes is a dict due to the fact that we need to call and store coordinates
            '''
            self.length = map_data[0][0]
            self.start = map_data[1]
            self.end = map_data[2]
            self.step_size = step_size
            self.obstacle_data = obstacle_data
            self.trial_points = []
            self.nodes = {}
            self.path = []

        def add_node(self, node_coords, parent_coords):
            '''
            This add a node into the graph's node dictionary with a parent reference
            Inputs:
                - node coordinates that we want to add
                - parent node attached to it
            Outputs:
                - node added to the dict with a parent reference (where key = coordinates)
            '''
            parent_node = self.nodes[parent_coords]
            new_node = Node(node_coords, parent_node) # creates a node object
            parent_node.add_child(new_node)           # makes sure the parent node is updated too! - v important to find path
            self.nodes[node_coords] = new_node

        def find_trial_point_and_node(self):
            '''
            This function finds a randomly generated point and the closest (valid-ish) node to it,
            It then uses this to generate the next node on the tree/graph
            Inputs
                - self.generate_trial_point()
                - find_closest_node()
                - step_to_trial_point()            
            Outputs
                - new node on the graph with satisfying all conditions (most of the time)
            '''

            trial_point = self.generate_trial_point()
            self.trial_points.append(trial_point)
            # finds a trial point for function reference
            
            closest_node = self.find_closest_node(trial_point)
            trial_point_node = self.step_to_trial_point(closest_node, trial_point)
            # the new node = trial_point_node, with the parent node being the closest one to it
            self.add_node(trial_point_node, closest_node)

        def generate_trial_point(self):
            '''
            This function randomly generates a point on the map (in boundary)
            Inputs
                - N/A
            Outputs
                - random point on map
            '''
            trial_point_x = random.randint(0, self.length-1)
            trial_point_y = random.randint(0, self.length-1)
            trial_point = (trial_point_x,trial_point_y)
            return trial_point

        def obstacle_finder(self, obstacle_number):
                '''
                Same as occupancy method, simply find the obstacle data of each of obstacle number
                Inputs:
                    - obstacle reference number
                Outputs:
                    - circle data in form [midpoint, radius]
                '''
                obstacle_list = self.obstacle_data
                mid_point_x = (obstacle_list[obstacle_number-1][0])
                mid_point_y = (obstacle_list[obstacle_number-1][1])
                radius = (obstacle_list[obstacle_number-1][2])

                return [[mid_point_x,mid_point_y,radius]]
                #Have to return a list of lists to then extract them

        def obstacle_point_check(self, point):   
            '''
            Checks whether a given point is in any of the obstacles
            Inputs:
                - point in the form (x,y)
            Outputs
                - a 0 if the point isn't in the obstacles
                - a 1 if it is
            '''
            point_x = (point[0])
            point_y = (point[1])
            list_of_obs_checks = []
            # have a list as we want to check if it's in any of the obstacles (using max function at the bottom)

            for i in range(len(self.obstacle_data)):
                obs_data = self.obstacle_finder(i+1)
                for x, y, radius in obs_data:
                        if math.sqrt((point_x - x)**2 + (point_y - y)**2) <= radius:
                            list_of_obs_checks.append(1)
                        else:
                            list_of_obs_checks.append(0)

            return max(list_of_obs_checks)

        def obstacle_vector_check(self, closest_node, trial_point_node):
            '''
            Finds if the vector between two points crosses on obstacle using parametic equations
            Inputs:
                - parent node and potential new node
            Outputs:
                - a 0 if the vector between the two doesn't interact with the obstacle
                - a 1 if it does

            '''
            from_x = closest_node[0]
            from_y = closest_node[1]
            to_x = trial_point_node[0]
            to_y = trial_point_node[1]

            for i in range(len(self.obstacle_data)):
                obs_data = self.obstacle_finder(i+1)
                for obs_x, obs_y, radius in obs_data:
                        if ((2 * ((to_x - from_x) * (from_x - obs_x) + (to_y - from_y) * (from_y - obs_y)))**2 - 
                            4*((to_x - from_x)**2 + (to_y - from_y)**2)*((from_x - obs_x)**2 + (from_y - obs_y)**2 - radius**2)) < 0:
                            # this is the discriminant, i know it is ugly!
                            return 0
                        else:
                            return 1

                '''
                    Converting points to cartesian eg.(x= from_x - t(to_x - from_x))
                    Substitute that into circle equation to get quadratic equation in terms of t, (0 <= t <= 1)
                    If the discriminant; b**2 - 4ac is >= 1 then it interacts with the circle
                '''

        def plot_trial_point(self,point_no):
            '''
            For my own reference to check why the graph might not be working/debugging
            Simpy plots all of the trial points used
            '''
            x = self.trial_points[point_no][0]
            y = self.trial_points[point_no][1]

            colors = ['green','blue','orange','pink','purple','orange','cyan']
            for ii in range(len(colors)):
                plt.scatter(x,y, color= colors[point_no % len(colors)], marker='+', s=30, linewidths=1)
                # cycles through set colours

            if self.obstacle_point_check((x,y)) == 1:
                plt.scatter(x,y, color= 'black', marker='o', s=30, linewidths=1)
                # plots a black dots if the random point interacts with an obstacle

        def find_closest_node(self, trial_point):
            '''
            Finds closest node to the trial point, or if that node is invalid goes to the next closest
            If both fails tries a random node to see if that is valid, and if all three are invalid assumes that no path is possible
            If there is only one possible parent node, defaults to it (start node)
            Inputs:
                ONLY the random trial point
            Output:
                The coordinates of the closest node that meets all conditions
            Additional comments:
                This is the best method I could figure out with the available time, it's not perfect and is the main reason for bugs in code
                However it is pretty good and relatively fast, and should be easily improvable and easy to bugfix(-ish)
            '''
            node_list = []

            for node in self.nodes:
                direction_vec = np.array(trial_point) - np.array(node)
                distance = math.sqrt(direction_vec[0]**2 + direction_vec[1]**2)
                node_list.append((node,distance))
                # creates a node list with structure: ((direction = x,y),distance)
            
            node_list.sort(key=lambda x: x[1])
            # sorts the node list by distance

            for node in node_list:
                child_x, child_y = self.step_to_trial_point(node[0], (trial_point))
                if 0 <= child_x <= self.length:
                    if 0 <= child_y <= self.length:
                        if self.obstacle_vector_check((node[0]),(child_x,child_y)) == 0:
                            if self.obstacle_point_check((child_x,child_y)) == 0:
                                return (node[0])
            # checks if the closest node is in the boundaries and not interacting with an obstacle
            # if it is - return it, if not - go to first fall back method

            if len(node_list) > 1:      # if it isn't just the start node
                try:
                    if self.obstacle_point_check((node_list[1][0])) == 0:
                        if self.obstacle_vector_check((node[0]),(node_list[1][0])) == 0:
                            print('defaulting to next available node', (node_list[1][0]))                       
                            return (node_list[1][0])
                    # this tries the next furthest node away

                    if self.obstacle_point_check((node_list[1][0])) == 1:
                        desperate_interger = random.randint(2,len(node_list))
                        print('trying desperate',desperate_interger)
                    # ^ if the next furthest node away also goes through an obstacle

                        if self.obstacle_point_check((node_list[desperate_interger][0])) == 0:
                            if self.obstacle_vector_check((node[0]),(node_list[desperate_interger][0])) == 0:
                                return (node_list[desperate_interger][0])
                        # tries a random interger out of desperation

                        else:
                            print('Cant find suitable path')
                            raise ValueError('Could not create graph, too many points in obstacles')
                        # if all fail, print error message
                    
                except Exception as e:
                    print('Error: {e}')
                    raise 
                        
            else:
                print('defaulting to start node')
                return (5,5) # if all else fails default to the start node!
                            # only happens when there is 1 node in total so that is why it makes sense

        def step_to_trial_point(self, closest_node, trial_point):
            '''
            Takes the trial point and creates a new node in that direction using step size (from the closest node)
            Inputs:
                - coordinates of the closest valid node
                - coordinates of the trial point
            Outputs:
                - coordinates of the new node
            '''
            node_coords = closest_node
            direction_vec = np.array(trial_point) - np.array(node_coords)
            distance_vec = math.sqrt(direction_vec[0]**2 + direction_vec[1]**2)
            # extract data

            if distance_vec == 0:
                return tuple(node_coords)
            # if there is no distance between the trial point and closest node default make new node = closest node
            # avoids bugs although erases certain points

            unit_vec = direction_vec / distance_vec
            x = node_coords[0] + self.step_size * unit_vec[0]
            y = node_coords[1] + self.step_size * unit_vec[1]
            # unit vector calculates relative direction in (x,y) space
            # the performs simple maths to keep step size constant

            node_array = np.array([int(x),int(y)])
            new_node = tuple(node_array.tolist())
            # makes sure node is in form - (x,y)
            # where x,y both are INTERGERS

            return new_node

        def go_to_end_point(self):
            '''
            Makes sure graph reaches end point from the tree
            Inputs:
                - N/A
            Outputs:
                - end point made a node object with parent node = closest node
                - add end point to node dict
            '''

            end_point = self.end
            # calls self.end from intialisation to find end point

            final_node = self.find_closest_node(end_point)
            self.add_node(end_point, final_node)
            # parent node = final_node

        def find_path(self, end):
            '''
            This function takes the end point and the final tree and finds the path using parent nodes
            Inputs:
                - end point
            Outputs:
                - a list of tuples (x,y) that creates a path based on valid nodes in the tree
            '''
            index_node = self.nodes.get(end)
            # starts at the end point

            while index_node is not None:
                self.path.append(index_node.coords)
                index_node = index_node.parent
                # while the node and it's parent exists, append it to the list and the move to it's parent
            
            print('Found path:',self.path)
        
        def plot_tree(self, map_data):
            '''
            Plots the tree and shortest path onto a graph, with obstacles 
            Inputs:
                - map data
            Outputs:
                - matplot lib with:
                    - red circles representing obstacles
                    - multicoloured dots representing nodes
                    - green line representing tree
                    - purple line representing shortest path
            '''

            fig, ax = plt.subplots(figsize=(8, 8))
            # immediatley define ax, (fig is not needed for this plot)

            start_pos = self.start
            goal_pos = self.end
            obstacles = map_data[3]
            for obstacle in obstacles:
                # Obstacle[0] is x position, [1] is y position, [2] is radius
                c_patch = plt.Circle((obstacle[0], obstacle[1]), obstacle[2], color='red', alpha=0.5)
                ax.add_patch(c_patch)
            ax.plot(goal_pos[0], goal_pos[1], 'r*')
            ax.plot(start_pos[0], start_pos[1], 'b*')
            # ^^^ same as occupancy method plot_map()

            for node in self.nodes.values():
                # cycle through each node in the tree
                if node.parent is not None:
                    parent_x, parent_y = node.parent.coords
                    x, y = node.coords
                    # finds parent coords and it's own coords

                    ax.plot([parent_x, x], [parent_y, y], linewidth=2, color = 'green')
                    # plot line using parent-node edge
                ax.scatter(node.coords[0], node.coords[1], s=10)
                # plot node, realised that I didn't need to cycle through a colour list at this point!

            x_coord_list = []
            y_coord_list = []
            for point in self.path:
                x_coord_list.append(point[0])   # here my x_coordinates are second due to numpy flipping the graph axis
                y_coord_list.append(point[1])   # vice versa with y
            ax.plot(x_coord_list, y_coord_list, linewidth=1, color = 'purple')
            # ^^^ again, same as occupancy method plot_path()

            ax.set_xlim(0, self.length) # set limits
            ax.set_ylim(0, self.length)
            ax.set_xlabel('x') # so no confusion with numpy array
            ax.set_ylabel('y')
            ax.grid(True) # make sure grid lines are on
                   
    def rrt_main():
        '''
        Main function to make sure everything runs in right order
        '''
        
        map_data = read_map_from_file(map_file)
        graph = Graph(map_data, step_size, obstacle_data=map_data[3]) # define intial graph object
        graph.nodes[graph.start] = Node(graph.start)                  # define intial node object in graph
        for ii in range(num_points):                                  # loops through the (main?) function of the graph
            graph.find_trial_point_and_node()
            graph.plot_trial_point(ii)
        graph.go_to_end_point()
        graph.find_path(graph.end)
        graph.plot_tree(map_data)
        plt.show()

    rrt_main() # finally, runs main!



rrt('map1 copy.csv',step_size=5,num_points=100)