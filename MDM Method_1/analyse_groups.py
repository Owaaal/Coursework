import csv
from matplotlib import pyplot as plt
import numpy as np
import sys
'''objective of the program:
read the grades from before they were put into groups
generate a histogram and find the variance and mean of that

read the grades of the groups from the iterations - find the variance and mean of the groups as a whole (taking into account their mean grade)
and for each iteration plot a separate histogram'''

GRADES_FILEPATH = "students.csv"

GROUPINGS_FILEPATH = "groupings_1.csv"


def read_initial_grades(filepath):
    '''reads the grades from the students from before they were grouped together
    and returns a numpy array of that'''
    ungrouped_grades = []
    with open(filepath, "r", newline="") as file:
        reader = csv.reader(file)
        for line in reader:
            ungrouped_grades.append(int(line[1]))
    return np.array(ungrouped_grades)

def generate_student_to_grade_mapping():
    '''generates a mapping between the ID of the student and the test score that it maps to'''
    mapping = dict()
    with open(GRADES_FILEPATH, "r", newline="") as file:
        reader = csv.reader(file)
        for line in reader:
            mapping[line[0]] = line[1]
    return mapping

def generate_group_of_groups(filepath):
    '''generates a numpy multi-dimensional array where each entry is the iteration, and each element of the iteraiton is a group'''
    group_of_groups = list()
    with open(filepath, "r", newline="") as file:
        reader = csv.reader(file)
        iteration = []
        for line in reader:
            if str(line[0]).__contains__("iter"):
                group_of_groups.append(iteration)
                iteration = list()
            else:
                iteration.append(line)
    return np.array(group_of_groups)


def find_mean_score_of_groups(group_of_groups, mapping):
    '''returns a multi_dimensional array of iteration x group mean'''
    matrix = []
    for iteration in group_of_groups:
        row = []
        for group in iteration:
            group = group.tolist()
            group = [int(mapping[elem]) for elem in group]
            row.append(sum(group) / len(group))
        matrix.append(row)
    return np.array(matrix)

def generate_array_of_average_score_by_iteration(matrix_of_averages):
    arr = list()
    for iteration in matrix_of_averages:
        iter = []
        for group in iteration:
            iter.append(group.mean())
        arr.append(iter)
    return np.array(arr)

def generate_group_average_by_iteration(array_of_averages):
    averages = []
    for iteration in array_of_averages:
        averages.append(iteration.var())
    return averages


def main():
    # sets the filepath from the command line
    if len(sys.argv) == 2:
        GROUPINGS_FILEPATH = sys.argv[1]
    
    
    
    # reads all the data from the supplied files, calculates important values such as the variance and the mean
    ungrouped_grades = read_initial_grades(GRADES_FILEPATH)
    student_to_grade_mapping = generate_student_to_grade_mapping()
    group_of_groups = generate_group_of_groups(GROUPINGS_FILEPATH)
    matrix_of_averages = find_mean_score_of_groups(group_of_groups, student_to_grade_mapping)
    array_of_averages = generate_array_of_average_score_by_iteration(matrix_of_averages)
    group_variance_by_iteration = generate_group_average_by_iteration(array_of_averages)
    
    print(ungrouped_grades.var())
    
    print(group_variance_by_iteration)
    
    
    
    # sets up the figure where all the data will be visualised
    fig, ax = plt.subplots(6, 1)
    
    # assigns axes to the data, generates titles and plots the data
    
    # plots the histogram for all of the students ungrouped
    ax[0].set_title("ungrouped students")
    ax[0].set_xlabel("student score")
    ax[0].set_ylabel("density")
    ax[0].hist(ungrouped_grades)
    
    for index in range(1, len(array_of_averages)+1):
        ax[index].hist(array_of_averages[index - 1])
        #ax[index].set_title("grouped students - iteration {}".format(index))
        ax[index].set_xlabel("group mean score")
        ax[index].set_ylabel("density")
        ax[index].plot()
            
    fig.subplots_adjust(hspace=1)
    plt.show()
    
    
if __name__ == "__main__":
    main()