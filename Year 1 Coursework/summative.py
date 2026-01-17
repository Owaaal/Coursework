from math import sin
from math import sqrt
# Use of csv library is optional- you can use it, but don't need to. 
import csv

'''
============================================
Portfolio exercise 1 - put your code for portfolio exercise 1 here. You should only need to change the '???' strings
============================================
'''

def comparisons(a, b):

    message = ''

    if a>b:
        message += "%.2f is greater than %.2f\n" % (a, b)
    if a<b:
        message += "%.2f is less than %.2f\n" % (a, b)
    if a==b:
        message += "%.2f is equal to %.2f\n" % (a, b)
    '''
    Note- we've changed this from being within 0.1% to just within 0.1. 
    You may need to change this code from previously submitted solutions.
    '''
    tol = 1e-1
    if a<= b+tol*b and a>=b-tol*b:
        message += " %.2f is within 0.1 of %.2f" % (a, b)

    return message


'''
============================================
Portfolio Exercise 2 - put your code for portfolio exercise 2 here.
============================================
'''

from math import sin
def kepler():

    M = 1.54
    e = 0.3
    tol=0.0001
    print('Mean anomaly M is',M)
    print('Eccentricity e is',e)

    # guess
    E = M

    for i in range(21):
        E = M + e*sin(E)
        if E!=M:
            print('Iteration',i,'Value of E',E)
        if abs(E-(M+e*sin(E))) < tol:
            print (E,M)
            break
    print('Eccentric anomaly E is',E)
    print('E - e sin E is', E - e*sin(E))

    return E


'''
============================================
Portfolio Exercise 3 - put your code for portfolio exercise 3 here.
============================================
'''

names = ["Martin", "Arthur", "Hemma", "Josh"]
Q1 = [6, 3, 7, 4]
Q2 = [1, 8, 4, 7]
Q3 = [4, 4, 5, 3]

def assign_grades():

    from math import sqrt
    import csv

    Martin = [Q1[0],Q2[0],Q3[0]]
    Arthur = [Q1[1],Q2[1],Q3[1]]
    Hemma = [Q1[2],Q2[2],Q3[2]]
    Josh = [Q1[3],Q2[3],Q3[3]]

    Q_1 = [Martin[0],Arthur[0],Hemma[0],Josh[0]]
    Q_2 = [Martin[1],Arthur[1],Hemma[1],Josh[1]]
    Q_3 = [Martin[2],Arthur[2],Hemma[2],Josh[2]]

    b=1
    for a in [Q_1,Q_2,Q_3]:
        average = sum(a)/len(a)
        for ii in range(1):
            print('Average for Q.',b,'is',average)
        b = b+1

    c =1
    TopHalf = []
    for s in [Q_1,Q_2,Q_3]:
        for a in range(len(Q_1)):
            X_minusAVG = (Q_1[a]-(sum(s)/len(s)))**2
            TopHalf.append(X_minusAVG)
        Stand_Dev = sqrt(sum(TopHalf)/4)
        for ii in range(1):
            print('Standard deviation for Q.',c,'is',Stand_Dev)
        c = c+1

    Total_marks = []
    for t in [Martin]:
        Total1 = sum(t)
        Total_marks.append(Total1)
        print('Martin has a total of',Total1)
    for t in [Arthur]:
        Total2 = sum(t)
        Total_marks.append(Total2)
        print('Arthur has a total of',Total2)
    for t in [Hemma]:
        Total3 = sum(t)
        Total_marks.append(Total3)
        print('Hemma has a total of',Total3)
    for t in [Josh]:
        Total4 = sum(t)
        Total_marks.append(Total4)
        print('Josh has a total of',Total4)
    for T in [Total_marks,]:
        average_T = sum(T)/len(T)
        print('The mean total mark achieved is',average_T)
    TopHalf2 = []
    for s in [Total_marks]:
        for a in range(len(Total_marks)):
            X_minusAVG2 = (Total_marks[a]-(sum(s)/len(s)))**2
            TopHalf2.append(X_minusAVG2)
        Stand_Dev_T = sqrt(sum(TopHalf2)/4)
        print('Standard deviation for total marks achieved is',Stand_Dev_T)
    #Total1 = M etc. (M,A,H,J)
    keyName = ['Martin','Arthur','Hemma','Josh']
    T_M_Dict = dict(zip(keyName,Total_marks))
    for key in T_M_Dict:
        T_M_Dict[key]=(T_M_Dict[key]-average_T)
    for key in T_M_Dict:
        if T_M_Dict[key] < -Stand_Dev_T:
            T_M_Dict[key]='Fail'
        elif T_M_Dict[key] < 0 and T_M_Dict[key] > -Stand_Dev_T:
            T_M_Dict[key]='C'
        elif T_M_Dict[key] < Stand_Dev_T and T_M_Dict[key] > 0:
            T_M_Dict[key]='B'
        elif T_M_Dict[key] >= Stand_Dev_T or T_M_Dict[key] == 0:
            T_M_Dict[key]='A'
    Mean = ('The mean total mark achieved is', average_T)
    S_D = ('The standard deviation between grades is', Stand_Dev_T)
    Names = list(T_M_Dict.keys())
    Grades = list(T_M_Dict.values())
    table = zip(Names, Grades)

    write_to = "grade_file.csv"
    with open(write_to, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(Mean)
        writer.writerow(S_D)
        writer.writerows(table)
        print(f"Data written to {write_to} successfully.")
    
    grade_file = "grade_file.csv"
    pass


'''
============================================
Portfolio Exercise 4 - put your code for portfolio exercise 3 here.
============================================
'''

word_list_file = 'word_list.txt'

def caesar_cipher():
    '''
    This function has the user input character/s into the terminal and then saves them into 'UserInput'
    It also displays the input.
    '''
    UserInput = ''
    def user_charinput():
        while True:
            UserInput = input("Enter text as input: ")
            if len(UserInput) <= 100:
                print(f'Your  input was: {UserInput}')
                return(UserInput)
            elif len(UserInput)>= 101:
                print("Enter an appropriate amount of letters as input: ")
    
    '''
    This then converts that input into ascii values
    '''
    Word_to_Char = list(user_charinput())
    ascii_values = list()
    for ii in range(len(Word_to_Char)):
        a_value = ord(Word_to_Char[ii])
        ascii_values.append(a_value)

    '''
    This checks if it's a single character or multiple, and also if it has to perform a wraparound
    It then shifts by an input in the terminal then displays the shifted word
    '''
    def user_shift():
        UserShift = ''
        shifted_list = list()
        Char_to_Word = list()
        '''
        This subfunction checks for non=negative values
        '''
        while True:
            UserShift = input("Enter a number to shift by: ")
            if UserShift.isdigit and int(UserShift) >= 0:
                UsrShftList = list(UserShift)
                while len(UsrShftList) != len(ascii_values):
                    UsrShftList.append(UserShift)
                for jj in range(len(ascii_values)):
                    shifted_n = int(ascii_values[jj]) + int(UsrShftList[jj])
                    shifted_list.append(shifted_n)
                
                '''
                This subfunction checks for wraparound, for both single and multiple characters
                It then shifts it accounting for the wraparound
                '''
                if any(value > 122 for value in shifted_list):
                        if shifted_list == str:
                            for evalue in shifted_list:
                                offset = int(97)
                                wraparound_no = chr((int(evalue) - offset + int(UserShift)) % 26 + offset)
                                wraparound_str = str(wraparound_no)
                                shifted_char = chr(wraparound_str)
                                Char_to_Word.append(shifted_char)
                        else:
                            for evalue in shifted_list:
                                offset = int(97)
                                wraparound_no = (int(evalue) - offset + int(UserShift) - int(UserShift))
                                wraparound_no2 = wraparound_no  % 26 + offset
                                wrap_char = chr(wraparound_no2)
                                Char_to_Word.append(wrap_char)
                else:
                    '''
                    If no wraparound is needed it simply shifts the characters
                    '''
                    for cc in range(len(shifted_list)):
                        c_value = chr(shifted_list[cc])
                        Char_to_Word.append(c_value)
                Word = ''.join(Char_to_Word)
                return(Word)
            
            elif UserShift != 0:
                '''
                This checks for negative values and accounts for those including wraparound
                Very similar to the other, just incorporates negatives
                '''
                UserShift = int(UserShift)
                UsrShftList = [int(UserShift) for digit in str(UserShift)]
                while len(UsrShftList) != len(ascii_values):
                    UsrShftList.append(UserShift)
                for jj in range(len(ascii_values)):
                    shifted_n = int(ascii_values[jj]) + int(UsrShftList[jj])
                    shifted_list.append(shifted_n)
                
                if any(value > 123 for value in shifted_list):
                        for evalue in shifted_list:
                            offset = int(97)
                            wraparound_no = chr((int(evalue) - offset + int(UserShift)) % 26 + offset)
                            wraparound_str = str(wraparound_no)
                            shifted_char = chr(wraparound_str)
                            Char_to_Word.append(shifted_char)
                else:
                    for cc in range(len(shifted_list)):
                        c_value = chr(shifted_list[cc])
                        Char_to_Word.append(c_value)
                
                '''
                Then it converts the ascii values to a word
                '''
                Word = ''.join(Char_to_Word)
                return(Word)

            else:
                '''
                Simple error function if the user tries to input a number greater than the alphabet
                '''
                print('Please input a number less than 26')
    
    print(f'Your new characters are: {user_shift()}')
    pass

def decode_message(word, word_list_file):
    '''
    This function opens a file when you give it a file name
    '''
    def open_list(path):
        '''
        This first bit opens the file and converts it to a list
        '''
        with open(path, 'r') as file:
            import csv
            file = csv.reader(file)
            words = list(file)
            word_list = [item for sublist in words for item in sublist]
            return word_list
    word_list = open_list('word_list.txt')

    '''
    This function shifts the entire word given to it by the number given to it
    '''
    def word_decoder(old_word, number):

            shifted_word_list = list()
            for char in old_word:
                ascii_value = ord(char) + 1
                if (ascii_value+number) < 123:
                    shift_value = ascii_value + number
                elif (ascii_value+number) >= 123:
                    ascii_value = ascii_value - 26
                    shift_value = ascii_value + number
                decoded_char = chr(shift_value)
                shifted_word_list.append(decoded_char)
            shifted_word = "".join(shifted_word_list)
            return shifted_word

    '''
    This brings the two together, shifting the word by one each time, and then matches it to the word_list
    '''
    word=word
    for number in range(25):
        final_word = word_decoder(word,number)
        if final_word in word_list:
            print('Your word is:', final_word)
    pass


'''
============================================
Test functions - code below this line must not be modified
============================================
'''

def test_comparisons():

    print("Testing Portfolio exercise 1")
    assert comparisons(1, 2) == "1.00 is less than 2.00\n", "Test 1 failed: Your output was %s" % comparisons(1, 2)
    assert comparisons(2, 1) == "2.00 is greater than 1.00\n", "Test 2 failed: Your output was %s" % comparisons(2, 1)
    assert comparisons(2, 2) == "2.00 is equal to 2.00\n 2.00 is within 0.1 of 2.00", "Test 3 failed: Your output was %s" % comparisons(2, 2)
    assert comparisons(2.05, 2) == "2.05 is greater than 2.00\n 2.05 is within 0.1 of 2.00", "Test 4 failed: Your output was %s" % comparisons(2.05, 2)
    print("Test complete")


def test_kepler():

    print("Testing Portfolio exercise 2")
    eccentricity = 0.3 
    mean_anomaly = 1.54

    ecc_anomaly = kepler()
    kepler_err = ecc_anomaly - eccentricity*sin(ecc_anomaly) - mean_anomaly

    assert kepler_err**2 < (1e-4*mean_anomaly)**2, f'Equation not close enough: error is {kepler_err}'
    print("Test complete")


def test_assign_grades():
    
    print("Testing Portfolio exercise 3")
    with open("grade_file.csv", "r") as f:
        data = f.readlines()

    assert len(data) == 6, "File should contain 6 lines, but actually contains %d" % len(data)
    
    mean = float(data[0].split(',')[1])
    assert mean == 14.0, "Mean should be 14.0, but is actually %.3f" % mean

    standard_deviation = float(data[1].split(',')[1])
    assert standard_deviation > 1.87 and standard_deviation < 1.88, "Standard deviation should be between 1.87 and 1.88, but is actually %.3f" % standard_deviation

    martin_grade = data[2].split(',')[1].strip()
    assert martin_grade == 'Fail', "Martin's grade should be Fail, but is actually %s" % martin_grade

    arthur_grade = data[3].split(',')[1].strip()
    assert arthur_grade == 'B', "Arthur's grade should be B, but is actually %s" % arthur_grade

    hemma_grade = data[4].split(',')[1].strip()
    assert hemma_grade == 'A', "Hemma's grade should be A, but is actually %s" % hemma_grade

    josh_grade = data[5].split(',')[1].strip()
    assert josh_grade == 'B', "Josh's grade should be A, but is actually %s" % josh_grade
    print("Test complete")

def test_caesar_cipher():

    #The assert function will raise an error if the expression contained within it doesn't evaluate to True
    assert caesar_cipher('a', 1) == 'b', "Test 1 failed"
    assert caesar_cipher('a', 2) == 'c', "Test 2 failed"
    assert caesar_cipher('z', 1) == 'a', "Test 3 failed"
    assert caesar_cipher('a', -1) == 'z', "Test 4 failed"
    assert caesar_cipher('hello', 3) == 'khoor', "Test 5 failed"
    assert caesar_cipher('khoor', -3) == 'hello', "Test 6 failed"

    print("Encoding tests passed - great job!")

def test_decode_message():

    assert decode_message("khoor", word_list_file) == "hello", "Test 1 failed. Output was %s" % decode_message("khoor", word_list_file)
    assert decode_message("Khoor", word_list_file) == "Invalid character in message", "Test 2 failed. Output was %s" % decode_message("Khoor", word_list_file)
    assert decode_message("khoor khoor", word_list_file) == "hello hello", "Test 3 failed. Output was %s" % decode_message("khoor khoor", word_list_file)
    assert decode_message("puppy", word_list_file) == "Can't decode word: puppy", "Test 4 failed. Output was %s" % decode_message("puppy", word_list_file) 

    print("Decoding tests passed - well done")

def test_cipher():

    print("Testing Portfolio exercise 4")
    test_caesar_cipher()
    test_decode_message()
    print("Test complete")

def main():

    marks = 20

    try:
        test_comparisons()
        print("Test passed - well done.")
    except Exception as error:
        marks = marks - 5 
        print ("Test failed with error")
        print(error)

    try:
        test_kepler()
        print("Test passed - well done.")
    except Exception as error:
        marks = marks - 5 
        print ("Test failed with error")
        print(error)

    try:
        assign_grades()
        test_assign_grades()
        print("Test passed - well done.")
    except Exception as error:
        marks = marks - 5 
        print ("Test failed with error")
        print(error)

    try:
        test_cipher()
        print("Test passed - well done.")
    except Exception as error:
        marks = marks - 5 
        print ("Test failed with error")
        print(error)

    print("All tests complete")
    print("Total mark: %d" % marks)

main()





