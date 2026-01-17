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

'''
The code below is a test function for your code. It will run automatically when you run the script. Do not modify it. 
To pass this assignment, your code must pass all the tests.
'''

def test_caesar_cipher():

	#The assert function will raise an error if the expression contained within it doesn't evaluate to True
	assert caesar_cipher('a', 1) == 'b', "Test 1 failed"
	assert caesar_cipher('a', 2) == 'c', "Test 2 failed"
	assert caesar_cipher('z', 1) == 'a', "Test 3 failed"
	assert caesar_cipher('a', -1) == 'z', "Test 4 failed"
	assert caesar_cipher('hello', 3) == 'khoor', "Test 5 failed"
	assert caesar_cipher('khoor', -3) == 'hello', "Test 6 failed"

	print("Encoding tests passed - great job!")

''' Task 2 - Decoding
Write your code for decoding a message when we don't know the shift. 
You are given a list of words that the message may be composed of. 
Your code should work by *brute force* i.e it should check all possible shifts until it finds a match
'''


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
    word_list = open_list(word_list_file)

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

def main():

    caesar_cipher()



main()
