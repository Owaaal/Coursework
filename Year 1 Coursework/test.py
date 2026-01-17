import numpy as np 
import math 
import scipy.sparse.csgraph 
import matplotlib.pyplot as plt 
import random 
import argparse
import collections
import sys
import csv

class Node:
    def __init__(self, coords, parent=None):
        self.position = coords  # (x, y)
        self.parent = parent
        self.children = []
    
    def add_child(self, child_node):
        self.children.append(child_node)

class Graph:
    def __init__(self):
        self.trial_points = []
        self.nodes = {}  # Change from list to dictionary

    def add_node(self, position, parent_position):
        parent_node = self.nodes[parent_position]
        new_node = Node(position, parent_node)
        parent_node.add_child(new_node)
        self.nodes[position] = new_node  # Store the new node in dictionary

# Initialize graph and root node
graph = Graph()
graph.nodes[(1,1)] = Node((1,1))  # Manually add the root node

# Add a new node
graph.add_node((5,5), (1,1))

# Print the graph structure
for pos, node in graph.nodes.items():
    parent_pos = node.parent.position if node.parent else None
    print(f"Node: {pos}, Parent: {parent_pos}")
