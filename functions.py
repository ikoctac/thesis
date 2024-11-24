import pyttsx3
import csv

# setup pyttsx3 for talking bot.
engine = pyttsx3.init()

# termination phrases for each language. 
termination_phrases = {
    'en-US': ["stop", "terminate", "end", "quit"],
    'el-GR': ["σταμάτα", "τερματισμός", "τέλος", "εξοδος"]
}

# switch commands for user
switch_commands = {
        'en-US': ["switch", "change","sweets"],
        'el-GR': ["αλλαγή", "άλλαξε"]
    }

# use pyttsx3 to vocalize the entry prompt.
def speak_language_prompt():
    engine.say("Choose English or Greek.")
    engine.runAndWait()

# function to select the language.
def select_language():
    print("Select the language you'll be speaking:")
    print("1. English")
    print("2. Greek")
    
    while True:
        try:
            language_choice = int(input("Enter the number corresponding to your language choice: "))
            languages = {
                1: 'en-US',
                2: 'el-GR'
            }
            # Return the language if it's valid, otherwise prompt the user again
            return languages.get(language_choice, 'en-US')  # Default to Greek if choice is invalid
        except ValueError:
            print("Invalid input. Please enter a valid number (1 or 2).")

# use pyttsx3 to vocalize the choosing mode summarize or translate
def speak_mode():
    engine.say("Do you want to summarize or just translate to ASL?")
    engine.runAndWait()

# use pyttsx3 to vocalize type or speak
def speak_input():
    engine.say("Do you want to type or speak?")
    engine.runAndWait()

# use pyttsx3 to vocalize the termination prompt
def speak_termination_prompt():
    engine.say("Do you want to terminate the process? ")
    engine.runAndWait()

# check for termination phrase .
def check_termination_phrase(text, language):
    phrases = termination_phrases.get(language, [])
    
    # normalize text, convert it to lowercase, strip any surrounding whitespace.
    normalized_text = text.lower().strip()
    
     # check for match in normalized_text.
    return normalized_text in phrases

# switch the person talking, mainly used to define who said what in the csv(Users must wait before the transition)
def check_switch_command(text, language):
    
    # check switch commands to terminate program
    commands = switch_commands.get(language, [])
    
    # normalize text, convert it to lowercase, strip any surrounding whitespace
    normalized_text = text.lower().strip()
    
    # check for match in normalized_text
    return normalized_text in commands

# used the first time to create the csv file header for each language
def create_csv_file(filename):
    try:
        with open(filename, 'x', newline='', encoding='utf-8') as csvfile:
            fieldnames = ["timestamp", "speech", "speaker"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
    except FileExistsError:
        pass 

# in case there isnt a file and we try to save promp an error
def save_to_csv(filename, timestamp, text, simplified_text,  speaker):
    try:
        with open(filename, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([timestamp, text, simplified_text , speaker])
    except FileNotFoundError:
        print("File not found. Create the CSV file first.")









