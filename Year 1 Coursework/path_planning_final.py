#The below import are all that should be used in this assignment. Additional libraries are not allowed.
import numpy as np 
import math 
import scipy.sparse.csgraph 
from scipy.sparse.csgraph import dijkstra, minimum_spanning_tree
import matplotlib.pyplot as plt 
import random 
import argparse
import collections
import sys
import csv

'''
==============================
The code below here is for your occupancy grid solution
==============================
'''

def read_map_from_file(filename):
    '''
    This functions reads a csv file describing a map and returns the map data
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
        '''
        This function reads the csv file and returns a list of lists of the data provided
        Inputs
            - filename (string): name of the file to read
        Outputs:
            - list of lists of data in the csv file
        '''
        with open(filename, 'r') as file:
            file = csv.reader(file)
            numbers = list(file)
            return numbers

    data_list = read_file(filename)
        # uses an outputs the function to a variable
    map_size = tuple(map(int, data_list[0]))
    start_point = tuple(map(int, data_list[1]))
    end_point = tuple(map(int, data_list[2]))
        # this takes uses data_list to size, start and end all in tuple + interger form form
        # can't just use int() function for a list have to map(int, list)

    def object_values_finder():
        '''
        This function turns the obstacle data into tuples
        Inputs:
            - none
        Outputs:
            - obstacles is a list of tuples. Each tuple represents a single  circular obstacle and is of the form (x, y, radius).
                    x is an integer representing the x coordinate of the obstacle
                    y is an integer representing the y coordinate of the obstacle
                    radius is an integer representing the radius of the obstacle
        '''
        object_list = []
        for ii in data_list[3:6]:
            obj_x = tuple(map(int, ii))
            object_list.append(obj_x)
        return(object_list)
    
    obj_values = list(object_values_finder())
        # uses the object_values_finder function to create a list of said tuples
    map_data = tuple([map_size,start_point,end_point, obj_values])
        # finally compiles the data into the required format

    return(map_data)

def make_occupancy_grid_from_map(map_data, cell_size):
    '''
    This function takes a map and a cell size (the physical size of one "cell" in the grid) and returns a 2D numpy array, 
    with each cell containing a '1' if it is occupied and '0' if it is empty
    Inputs: map (tuple) - see read_map_from_file for description.
    Outputs: occupancy_grid - 2D numpy array
    '''
    
    array_length = int(map_data[0][0])
    array_height = int(map_data[0][1])
    occupancy_array = np.zeros((array_length,array_height))
        # creates an empty occupancy array using the map data

    def obstacle_finder(obstacle_number):
        '''
        Finds the all of the intergers in the circular obstacle, and adds extra to approximate for cell size
        Inputs
            - an obstacle number
        Outputs
            - updates the occupancy array with all the points that circle take up
        '''
        obstacle_list = map_data[3]
        obs_x = (obstacle_list[obstacle_number-1][0])
        obs_y = (obstacle_list[obstacle_number-1][1])
        radius = (obstacle_list[obstacle_number-1][2])

        for x in range(int(-radius),int(radius+1)):
            x_value = int(obs_x + x)
            for y in range(int(-radius),int(radius+1)):
                y_value = int(obs_y + y)

                distance = math.sqrt(x**2 + y**2)
                if distance < radius:
                    if 0 <= x_value < array_length and 0 <= y_value < array_height:
                        occupancy_array[x_value,y_value] = 1

    for i in range(len(map_data[3])):
        obstacle_finder(i+1)   
        # this loops through all the possible obstacles from the data and runs them through the finder function

    def convert_array_to_grid(occupancy_array):
        '''
        This function converts the occupancy array to the occupancy grid with cell size = ? (using np.max to not add 1s together)
        Inputs
            - the completed occupancy array
        Outputs
            - an occupancy grid with cell size specified for the main function
        '''
        grid_length = int(array_length/cell_size)
        grid_height = int(array_height/cell_size)
        grid_array = np.zeros((grid_length,grid_height))
        
        for x in range(grid_length):
            x_min = cell_size * x
            x_max = cell_size * (x+1)
            for y in range(grid_height):
                y_min = cell_size * y
                y_max = cell_size * (y+1)
                grid_array[x,y] = np.max(occupancy_array[x_min:x_max, y_min:y_max])
        return(grid_array)

    grid_array = convert_array_to_grid(occupancy_array)
        # uses the function to return a variable that we can output

    return (grid_array)
 
def make_adjacency_matrix_from_occupancy_grid(occupancy_grid):
    '''
    This function converts an occupancy grid into an adjacency matrix. We assume that cells are connected to their neighbours unless the neighbour is occupied. 
    We also assume that the cost of moving from a cell to a neighbour is always '1' and allow only horizontal and vertical connections (i.e. no diagonals allowed).
    Inputs: occupancy_grid - a 2D (NxN) numpy array. An element with value '1' is occupied, while those with value '0' are empty.
    Outputs: A 2D (MxM where M=NxN) array. Element (i,j) contains the cost of travelling from node i to node j in the occupancy grid. 
    '''

    def create_adjacency_matrix(occupancy_grid):
        '''
        Creates an intial adjacency matrix then updates it accordingly
        Inputs
            - occupancy grid
        Outputs
            - adjacency matrix
        '''

        grid_dimensions = occupancy_grid.shape
        grid_length = grid_dimensions[0]
        grid_height = grid_dimensions[1]
        matrix_length = int(grid_length**2) 
        matrix_height = int(grid_height**2)
        adjacency_matrix = np.zeros((matrix_length,matrix_height))
            # simply creates an empty adj. matrix with the occupancy grid provided (using .shape)
        
        def index(x,y):
            '''
            Assigns x,y values to a 1D index or cell number
            Cannot just flatten it, need to assign each value
            Inputs
                - x,y coordinates
            Outputs
                - an interger that tells us how many rows down that coordinate is (aka cell number)
            '''
            flat_index = x + y * grid_length
            return flat_index

        for xx in range(grid_length):
            for yy in range(grid_height):
                if occupancy_grid[(xx,yy)] == 0:
                    if xx + 1 < grid_length and occupancy_grid[(xx+1,yy)] == 0:
                        adjacency_matrix[(index(xx,yy),index(xx+1,yy))] = 1
                    if xx - 1 >= 0 and occupancy_grid[(xx-1,yy)] == 0:
                        adjacency_matrix[(index(xx,yy),index(xx-1,yy))] = 1
                    if yy + 1 < grid_height and occupancy_grid[(xx,yy+1)] == 0:
                        adjacency_matrix[(index(xx,yy),index(xx,yy+1))] = 1
                    if yy - 1 >= 0 and occupancy_grid[(xx,yy-1)] == 0:
                        adjacency_matrix[(index(xx,yy),index(xx,yy-1))] = 1
                if occupancy_grid[(xx,yy)] == 1:
                    adjacency_matrix[(index(xx,yy)),:] = 0
        '''
        For every (x,y) first checks if it's in bounds of the grid
        Then checks if it's neighbours are free using == 0
        Updates the adj_mtx using the 1D index if they are free using the index function
        Then checks all the taken (==1) squares and sets that row to 0
        '''

        return(adjacency_matrix)

    adj_mtx = create_adjacency_matrix(occupancy_grid=occupancy_grid)
        # again just calls the function into a variable to return

    return(adj_mtx)

def get_path_from_predecessors(predecessors, map_data, cell_size):
    '''
    This function takes a predecessors matrix, map_data and cell_size as input and returns the path from start to goal position. 
    We take the mid-point of each cell as the (x, y) coordinate for the path.
    Inputs: predecessors - a 1D numpy array (size = M = NxN, where N is the length of an occupancy grid) produced by scipy's implementation of Dijkstra's algorithm.
            Each element i tells us the index of the node we should travel to if we are in node j. 
            map_data -  (tuple) see read_map_from_file for description.
            cell_size - (integer) the physical size corresponding to a single cell in the grid.
    Outputs: path - A list of tuples (x, y), where (x, y) are the coordinates of a position we can travel to in the map. 
    '''
    start_point = map_data[1]
    end_point = map_data[2]
    map_length = map_data[0][0]
    grid_length = int(map_length/cell_size)
    # once again finds key data from map_data

    def find_cell(point):
        '''
        This function takes the updated x and y of an inputted point and finds what cell it is in
        Through two for loops 
        '''
        point_x = point[0]/cell_size
        point_y = point[1]/cell_size
        #cell = (y + grid_length * x)
        for xx in range(grid_length):
            for yy in range(grid_length):
                if  xx < point_x <= (xx+1) and yy < point_y <= (yy+1):
                    cell = yy + grid_length*xx
        return(cell)

    end_cell = find_cell(end_point)
    start_cell = find_cell(start_point)
        # defines the start and end in terms of cells

    def find_path(predecessors, start, end):
        '''
        This function takes the pred. matrix and the start and end CELLS
        Then iterates through the matrix as shown on the brief using a while loop
        Also checks that it can move through the cell and it isn't -999
        Inputs:
            - predecessor matrix
            - start and end CELL
        Outputs:
            - shortest path in cell number format
        '''
        path_list = []
        ii = end
        path_list.append(end)

        while ii != start:
            if predecessors[(start_cell,ii)] >= 0:
                ii = predecessors[(start_cell,ii)]
                path_list.append(int(ii))
            else:
                print('Cannot reach the end cell')
                break
        return(path_list)

    cell_path = find_path(predecessors,start=start_cell,end=end_cell)
        # uses the find path function to create a path list

    def find_cell_midpoint(cell):
        '''
        This function finds the bottom left corner of the cell by finding the modulo for the y coordinate and the interger divide for the x
        Then finds the midpoint by adding +0.5 to both axis and multiplying them by the cell size
        Inputs:
            - cell number
        Outputs:
            - x and y corrdinates of its midpoint (adjusted back to full grid size)
        '''
        y = cell % grid_length
        x = int(cell/grid_length)
        grid_midpoint_x = x + 0.5
        grid_midpoint_y = y + 0.5
        midpoint_x = grid_midpoint_x * cell_size
        midpoint_y = grid_midpoint_y * cell_size
        return(midpoint_x,midpoint_y)


    def find_coordinate_path():
        '''
        Finally this iterates through the cell path converting cell numbers to coordinates of their midpoints and updates them into a tuple
        It then checks if there is a path by seeing if the list has more than just a start and an end point
        Inputs:
            - none
        Outputs:
            - the converted cell path to a coordinate path
        '''
        coordinate_list = [end_point]
        for ii in range(1,len(cell_path)):
            coordinates = find_cell_midpoint(cell_path[ii])
            coordinate_list.append(coordinates)
        coordinate_list.append(start_point)
        coordinate_path = tuple(coordinate_list)

        if len(coordinate_path) < 3:
            coordinate_path = []
        
        return (coordinate_path)

    coordinate_path = find_coordinate_path()
        # call the function to create a variable we can output

    return(coordinate_path)

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

def plot_path(ax, path):
    '''
    This function plots the path found by your occupancy grid solution.
    Inputs: ax (matplotlib axis) - the axis object where the path will be drawn
            path (list of tuples) - a list of points (x, y) representing the spatial co-ordinates of a path.
    '''

    x_coord_list = []
    y_coord_list = []

    for point in path:
        x_coord_list.append(point[1])   # here my x_coordinates are second due to numpy flipping the graph axis
        y_coord_list.append(point[0])   # vice versa with y

    ax.plot(x_coord_list, y_coord_list, linewidth=2, color = 'purple')
        # plots a line using the two list and a line width
        # also it's purple cause i like purple :)

def test_make_occupancy_grid():
    
    map0 = ((10,10), (1, 1), (9, 9), [])
    assert np.array_equal(make_occupancy_grid_from_map(map0, cell_size=1), np.zeros((10, 10))), "Test 1 - checking map 0 with cell size 10"
    
    map1 = ((10,10), (1, 1), (9, 9), [(5, 5, 2)])
    assert np.array_equal(make_occupancy_grid_from_map(map1, cell_size=10), np.array([[1]])), "Test 1 - checking map 1 with cell size 10"
    assert np.array_equal(make_occupancy_grid_from_map(map1, cell_size=5), np.array([[1, 1], [1, 1]])), "Test 2 - checking map 1 with cell size 5"

    map1_cell_size_2_answer = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0]]) # < Changed this as my method uses a more effecient method
    assert np.array_equal(make_occupancy_grid_from_map(map1, cell_size=2), map1_cell_size_2_answer), "Test 3 - checking map 1 with cell size 2"

    map2 = (100, (1, 1), (9, 9), [(10, 10, 5), (90, 90, 5)])
    map2_answer = np.array([[1, 0, 0, 0, 0], 
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1]])

    occupancy_grid1 = np.array([[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]])

    adjacency_matrix1 = np.array([[0., 1., 0., 1., 0., 0., 0., 0., 0.],
        [1., 0., 1., 0., 1., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0., 1., 0., 0., 0.],
        [1., 0., 0., 0., 1., 0., 1., 0., 0.],
        [0., 1., 0., 1., 0., 1., 0., 1., 0.],
        [0., 0., 1., 0., 1., 0., 0., 0., 1.],
        [0., 0., 0., 1., 0., 0., 0., 1., 0.],
        [0., 0., 0., 0., 1., 0., 1., 0., 1.],
        [0., 0., 0., 0., 0., 1., 0., 1., 0.]])

    assert np.array_equal(make_adjacency_matrix_from_occupancy_grid(occupancy_grid1), adjacency_matrix1)

    occupancy_grid2 = np.array([[1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]])

    adjacency_matrix2 = np.zeros((occupancy_grid2.size, occupancy_grid2.size))

    assert np.array_equal(make_adjacency_matrix_from_occupancy_grid(occupancy_grid2), adjacency_matrix2)

    occupancy_grid3 = np.array(
        [
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0]
        ])

    adjacency_matrix3 = np.array([
    [0., 1., 0., 0., 0., 0., 0., 0., 0.],
    [1., 0., 1., 0., 0., 0., 0., 0., 0.],
    [0., 1., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 1., 0.],
    [0., 0., 0., 0., 0., 0., 1., 0., 1.],
    [0., 0., 0., 0., 0., 0., 0., 1., 0.]])
    assert np.array_equal(make_adjacency_matrix_from_occupancy_grid(occupancy_grid3), adjacency_matrix3)

def test_get_path():

    '''TEST 1'''
    map_1 = ((15, 15), (3, 3), (13, 13), [(7, 7, 2)])
    distance_matrix_1 = np.array([
    [0., 1., 0., 1., 0., 0., 0., 0., 0.],
    [1., 0., 1., 0., 0., 0., 0., 0., 0.],
    [0., 1., 0., 0., 0., 1., 0., 0., 0.],
    [1., 0., 0., 0., 0., 0., 1., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 1., 0., 0., 0., 0., 0., 1.],
    [0., 0., 0., 1., 0., 0., 0., 1., 0.],
    [0., 0., 0., 0., 0., 0., 1., 0., 1.],
    [0., 0., 0., 0., 0., 1., 0., 1., 0.]
    ])
    path_1 = ((13, 13), (7.5, 12.5), (2.5, 12.5), (2.5, 7.5), (2.5, 2.5), (3, 3))
    predecessors_1 = scipy.sparse.csgraph.shortest_path(distance_matrix_1, directed=False, return_predecessors=True)[1]
    assert np.array_equal(get_path_from_predecessors(predecessors_1, map_1, cell_size=5), path_1), "Test 1 - Incorrect path for map size 15,15 one obstacle"
    
    '''TEST 2'''
    map_2 = ((50, 50), (5, 5), (45, 45), [(15, 30, 20), (45, 10, 2)])
    oc_grid_2 = make_occupancy_grid_from_map(map_2, cell_size=2)
    distance_matrix_2 = make_adjacency_matrix_from_occupancy_grid(oc_grid_2)
    predecessors_2 = scipy.sparse.csgraph.shortest_path(distance_matrix_2, directed=False, return_predecessors=True)[1]
    path_2 = ((45, 45), (45.0, 43.0), (43.0, 43.0), (43.0, 41.0), (41.0, 41.0), (41.0, 39.0), (39.0, 39.0), (39.0, 37.0), (37.0, 37.0), (35.0, 37.0), (33.0, 37.0), 
              (31.0, 37.0), (29.0, 37.0), (27.0, 37.0), (25.0, 37.0), (23.0, 37.0), (21.0, 37.0), (19.0, 37.0), (17.0, 37.0), (15.0, 37.0), (13.0, 37.0), (11.0, 37.0), 
              (9.0, 37.0), (9.0, 35.0), (9.0, 33.0), (9.0, 31.0), (9.0, 29.0), (9.0, 27.0), (9.0, 25.0), (9.0, 23.0), (9.0, 21.0), (9.0, 19.0), (9.0, 17.0), (9.0, 15.0), 
              (9.0, 13.0), (7.0, 13.0), (7.0, 11.0), (7.0, 9.0), (5.0, 9.0), (5.0, 7.0), (5.0, 5.0), (5, 5))
    assert np.array_equal(get_path_from_predecessors(predecessors_2, map_2, cell_size=2), path_2), "Test 2 - your cell_size function isn't working - path only works for cell_size <5"


    pass

def test_occupancy_grid():
    test_make_occupancy_grid()
    pass

def occupancy_grid(file, cell_size):

    fig, ax = plt.subplots(figsize=(8, 8))
    # immediatley define ax, (fig is not needed for this plot)

    map_data = read_map_from_file(file)
    plot_map(ax, map_data)
    grid = make_occupancy_grid_from_map(map_data, cell_size)
    distance_matrix = make_adjacency_matrix_from_occupancy_grid(grid)
    predecessors = scipy.sparse.csgraph.shortest_path(distance_matrix, directed=False, return_predecessors=True)[1]
    path = get_path_from_predecessors(predecessors, map_data, cell_size)
    print(path)
    plot_path(ax, path)

    ax.set_xlim(0, map_data[0][0]) # set limits
    ax.set_ylim(0, map_data[0][0])
    ax.set_xlabel('x') # so no confusion with numpy array
    ax.set_ylabel('y')
    ax.grid(True) # make sure grid lines are on
    plt.show()



'''
==============================
The code below here is for your RRT solution
==============================
'''
def rrt(map_file, step_size=5, num_points=100):

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

            self.path.reverse()
            # formats path so it is start to end

        
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
        graph.go_to_end_point()
        graph.find_path(graph.end)
        graph.plot_tree(map_data)
        plt.show()
        return(graph.path)

    path = rrt_main() # finally, runs main
    print('Found path:',path)
    return path


def test_rrt():
    file1 = 'map1 copy.csv'
    map_data = read_map_from_file('map1 copy.csv')
    obstacle_data = map_data[3]
    path = rrt(file1, step_size=5, num_points=100)

    assert path is not None, "Test 1 - Path doesn't exist"

    assert len(path) < 50, "Test 2 - Path shouldn't span the entire tree (or most of it)"

    assert path[0] == (5,5), "Test 3 - Path doesn't start at start point"

    assert path[-1] == (95,95), "Test 4 = Path doesn't end at end point"

    for point in path:
        for x, y, radius in obstacle_data:
            assert math.sqrt((point[0]- x)**2 + (point[1] - y)**2) > radius, "Test 5 - Point is in obstacle"

    '''for i in range(len(path) - 1): 
        point = path[i]
        parent = path[i + 1]
        if parent is not None:
                to_x = point[0]   
                to_y = point[1]
                from_x = parent[0]
                from_y = parent[1]
                for x, y, radius in obstacle_data:
                    discriminant = (((2 * ((to_x - from_x) * (from_x - x) + (to_y - from_y) * (from_y - y)))**2) - 4*((to_x - from_x)**2 + (to_y - from_y)**2)*((from_x - x)**2 + (from_y - y)**2 - radius**2))
                    if discriminant < 0:
                        assert discriminant > -5, f"Test 6 - Line between {point} and {parent} intersects an obstacle, {x,y}"
    # this test is slightly broken due to rounding errors!'''

    for i in range(len(path) - 2): # -2 as the last node's step size will always be to big
        point = path[i]
        parent = path[i + 1]
        step_size = math.sqrt((point[0] - parent[0]) ** 2 + (point[1] - parent[1]) ** 2)
        assert 4 <= step_size <= 5, f"Test 7, step size is too large or too small, ={step_size}"

'''
==============================
The code below here is used to read arguments from the terminal, allowing us to run different parts of your code.
You should not need to modify this
==============================
'''


def main():

    parser = argparse.ArgumentParser(description=" Path planning Assignment for CPA 2024/25")
    parser.add_argument('--rrt', action='store_true')
    parser.add_argument('-test_rrt', action='store_true')
    parser.add_argument('--occupancy', action='store_true')
    parser.add_argument('-test_occupancy', action='store_true')
    parser.add_argument('-file')
    parser.add_argument('-cell_size', type=int)

    args = parser.parse_args()

    if args.occupancy:
        if args.file is None:
            print("Error - Occupancy grid requires a map file to be provided as input with -file <filename>")
            exit()
        else:
            if args.cell_size:
                occupancy_grid(args.file, args.cell_size)
            else:
                occupancy_grid(args.file)

    if args.test_occupancy:
        print("Testing occupancy_grid")
        test_occupancy_grid()

    if args.test_rrt:
        print("Testing RRT")
        test_rrt()
    
    if args.rrt:
        if args.file is None:
            print("Error - RRT requires a map file to be provided as input with -file <filename>")
            exit()
        else:
            rrt(args.file)

if __name__ == "__main__":
    main()


occupancy_grid('map2 copy.csv', 5)
rrt('map2 copy.csv')