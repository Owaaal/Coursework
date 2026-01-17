import csv
import numpy as np

FILENAME = "students.csv"

N = 50

SEED = 50

MEDIAN = 70

VARIANCE = 5

def generate_names():
    '''generates the name for each student, basically student 2, student 43 etc..'''
    names = []
    for index in range(N):
        name = []
        name.append("student {}".format(index + 1))
        names.append(name)
    return names

def add_figure(rows):
    '''takes an array of arrays and adds them to a csv file'''
    with open(FILENAME, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(rows)
        
def generate_scores():
    '''generates a random sample of N scores from a normal distribution of MEDIAN and VARIANCE,
    makes them into an integer and then turns the list into an array of arrays'''
    rng = np.random.default_rng(SEED)
    scores = rng.normal(MEDIAN, VARIANCE, N).tolist()
    scores = [[int(score)] for score in scores]
    return scores

def make_rows(data_1, data_2):
    '''turns two arrays into an array of concatenated arrays'''
    rows = []
    for d1, d2 in zip(data_1, data_2):
        rows.append(d1 + d2)
    return rows

def main():
    names = generate_names()
    scores = generate_scores()
    rows = make_rows(names, scores)
    add_figure(rows)
if __name__ == "__main__":
    main()