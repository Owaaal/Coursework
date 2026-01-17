from numpy import random as rd

# global variables - model parameters

NUM_STUDENTS = 50
NUM_SESSIONS = 5
GROUP_SIZE = 5
SHIFT = 0




def generate_score():
    '''generates a score according to a normal distribution using the numpy library'''
    rng = rd.default_rng()
    num = int(rng.normal(65, 15))
    return num

def generate_studens():
    '''generates a sorted list of test scored for each student'''
    scores = []
    for index in range(NUM_STUDENTS):
        scores.append(generate_score())
    scores = sorted(scores)
    return scores

def make_matrix(scores):
    num_groups = NUM_STUDENTS // GROUP_SIZE
    group_matrix = []
    for index in range(GROUP_SIZE):
        new_row = []
        for jdex in range(num_groups):
            new_row.append(scores.pop(0))
        if index % 2 == 0:
            new_row.reverse()
        group_matrix.append(new_row)
    return group_matrix

def make_groups(matrix):
    '''makes the groups by reading down the column of the matrix'''
    groups = []
    num_groups = NUM_STUDENTS // GROUP_SIZE
    for index in range(num_groups):
        new_group = []
        for row in matrix:
            new_group.append(row[index])
        groups.append(new_group)
    return groups

def badly_generate_students():
    scores = []
    for index in range(NUM_STUDENTS):
        scores.append(generate_score())
    return scores

def print_groups(groups):
    '''prints each group out sequentially with some information'''
    for group in groups:
        print("group {index}: {group}, sum = {sum}, mean = {mean}".format(index=groups.index(group)+1, group=group, sum=sum(group), mean = sum(group)/len(group)))

def shift_row(row):
    '''shifts the row of a matrix to the left by one, with wraparound'''
    first_elem = row.pop(0)
    row.append(first_elem)
    return row   

def shift_rows(matrix):
    '''shifts the rows of the matrix according to the algorithm in method.txt'''
    global SHIFT
    new_matrix = []
    for index in range(len(matrix)):
        new_row = matrix[index]
        if index * SHIFT == 0:
            new_matrix.append(new_row)
            continue
        for jdex in range(SHIFT * index):
            new_row = shift_row(new_row)
        new_matrix.append(new_row)
    SHIFT += 1
    return new_matrix
    
def run_iterations(matrix):
    for i in range(NUM_SESSIONS):
            print("Iteration {index}".format(index=i+1))
            matrix = shift_rows(matrix)
            groups = make_groups(matrix)
            print_groups(groups)
            print("------------------------------")

def main():
    students = generate_studens()
    #students = [i for i in range(1, NUM_STUDENTS + 1)]
    #students = badly_generate_students()
    print("------------------------------")
    matrix = make_matrix(students)
    run_iterations(matrix)
    
    
if __name__ == "__main__":
    main()