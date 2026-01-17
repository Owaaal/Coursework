import numpy as np
from matplotlib import pyplot as plt
import csv
from collections import OrderedDict

### GLOBAL VARIABLES

FILENAME = "students.csv"

GROUP_FILENAME = "groupings_3.csv"

N = 50

S = 5

G = 5



def get_scores():
    scores = dict()
    with open(FILENAME, "r", newline="") as file:
        reader = csv.reader(file)
        for line in reader:
            scores[line[0]] = line[1]
    return scores
def sort_scores(scores):
    sorted_scores = dict(sorted(scores.items(), key=lambda item: item[1]))
    return sorted_scores
def make_matrix(scores):
    n_rows = G
    n_columns = N // G
    matrix = np.array(list(scores.keys()))
    matrix = matrix.reshape(n_rows, n_columns)
    return matrix
def randomise_rows(matrix):
    randomised_matrix = []
    for row in matrix:
        np.random.shuffle(row)
    return matrix
def shift_rows(matrix):
    new_matrix = []
    for index in range(G):
        new_matrix.append(np.roll(matrix[index], index))
    new_matrix = np.array(new_matrix)
    return new_matrix
def run_iterations(matrix):
    matrices = [matrix]
    for i in range(S - 1):
        matrices.append(shift_rows(matrices[i]))
    return np.array(matrices)

def make_groups_from_matrices(matrices):
    groups_matrix = []
    for matrix in matrices:
        groups = matrix.transpose()
        groups_matrix.append(groups)
    return groups_matrix
def make_rows(groups):
    rows = []
    for i in range(1, S+1):
        for j in range(N // G):
            rows.append(groups[i-1][j].tolist())
        rows.append(["iteration {}".format(i)])
    return rows
def write_to_group_file(rows):
    with open(GROUP_FILENAME, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(rows)

def main():
    scores = get_scores()
    scores = sort_scores(scores)
    matrix = make_matrix(scores)
    matrix = randomise_rows(matrix)
    matrices = run_iterations(matrix)
    groups = make_groups_from_matrices(matrices)
    rows = make_rows(groups)
    write_to_group_file(rows)
    
    
if __name__ == "__main__":
    main()
    