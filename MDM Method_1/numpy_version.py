import numpy as np
from matplotlib import pyplot as plt
# MODEL PARAMETERS

N = 50
S = 5
G = 5
ORDERED_DIMENSION = N // G
MAX_SCORE = 100
MIN_SCORE = 0


def generate_students():
    rng = np.random.default_rng()
    students = rng.normal(65, 7, N)
    students = [int(student) for student in students]
    #students = sorted(students)
    return np.array(students)


def resize_and_balance(students):
    students = students.reshape(G, ORDERED_DIMENSION)
    for index in range(len(students)):
        if index % 2 == 1:
            students[index] = np.flipud(students[index])
    return students

def main():
    
    fig = plt.figure()
    
    ax1 = fig.add_subplot(311)
    
    
    # generates 50 students as a sample from a normal distribution
    students = generate_students()
    
    ax1.hist(students)
        
        
    # finds the mean score of the students as a whole
    average_score = students.mean()
    
    # finds the variance of the students as a whole
    variance = students.var()
    print("the mean score is", average_score)
    
    students = resize_and_balance(students)    
    ax2 = fig.add_subplot(312)
    
    scores = []
    for index in range(len(students)):
        scores.append(float(students[index].sum() / len(students[index])))
    scores = np.array(scores)
    
    
    # finds the variance of the mean scores of the groups
    scores_variance = scores.var()
    
    
    print("the variance of the students as a whole is", variance)
    print("the variance of the groups is", scores_variance)
        
    ax2.hist(scores)
    
    #print(scores)
    plt.show()
    
main()