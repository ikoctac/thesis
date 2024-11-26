import speech_recognition as sr
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # depends on comp hardware, in each case uncomment
from datetime import datetime
from functions import *
from mapping import simplify_text_for_asl,display_images_sequentially

# initialize recognizer
recognizer = sr.Recognizer()

# starting speaker
cur_speaker = "Person 1"

#set the delay for pictures
def_delay = 200

# directory for the csv later
cur_directory = os.path.dirname(os.path.abspath(__file__))

# where the csv is going to be saved, if the folder doesnt exist
directory_path = os.path.join(cur_directory, "csv_files")

# mode between simple translate and using a ML model for summarize big sentences
while True:
    speak_mode()
    mode_choice = input("Do you want to simplify (summarize) or just translate to ASL? (simplify/translate): ").strip().lower()#removes white space and lowers text
    if mode_choice in ['simplify', 'translate']:
        break
    else: #loop until valid input
        print("Invalid choice. Please choose 'simplify' or 'translate'.")

# choose type for convenience or speak to try the speech recognition
while True:
    speak_input()
    input_mode = input("Do you want to type or speak? (type/speak): ").strip().lower() #removes white space and lowers text
    if input_mode in ['speak', 'type']:
        break
    else: #loop until valid input
        print("Invalid choice. Please choose 'type' or 'speak'.")

if input_mode == 'speak':
    # use the default microphone as source, bot asks what language will the user use.
    with sr.Microphone() as source:
        speak_language_prompt()  # language selection 
        lang = select_language()
        print(f"Listening in {lang}...")

        # csv file creation
        csv_filename = os.path.join(directory_path, f"speech_data_{lang}.csv")

        # adjust for ambient noise if necessary
        recognizer.adjust_for_ambient_noise(source)

        # loop for listening to users input
        while True:
            try:
                # capture user audio
                audio = recognizer.listen(source)

                # convert speech to text using the selected language
                text = recognizer.recognize_google(audio, language=lang)
                
                # if the language is greek it will only translate, simplify cant work not enough datasets in greek trained
                if lang == 'el-GR':
                    simplified_text = text
                else:
                    # used to choose between translate or simplify
                    if mode_choice == 'simplify':
                        simplified_text = simplify_text_for_asl(text)
                    else:
                        simplified_text = text  # if anything else than simplify it will just translate the text

                # switch the speaker command
                if check_switch_command(text, lang):
                    print("Who is speaking?")
                    new_speaker = input("Enter the new speaker (e.g., Person 2): ")
                    cur_speaker = new_speaker
                    print(f"Switched to {new_speaker}")

                # termination function
                elif check_termination_phrase(text, lang):
                    speak_termination_prompt()
                    decision = input("Do you want to terminate the process? (yes/no): ")
                    if decision.lower() == 'yes':
                        print("Terminating the process.")
                        break
                    else:
                        print("Resuming...")

                # print procecced text
                print(f"Processed Text: {simplified_text}")

                #display images
                display_images_sequentially(simplified_text, lang, def_delay)

                # timestamp for csv
                timestamp = datetime.now().strftime("%H:%M:%S")

                # save data with the timestamp,text,simplified_text and the speaker
                save_to_csv(csv_filename, timestamp, text, simplified_text, cur_speaker)

            # used to see if the microphone is still capturing audio
            except sr.UnknownValueError:
                print("Could not understand audio.")
            except sr.RequestError as e:
                print(f"Error: {e}")

elif input_mode == 'type':
    speak_language_prompt()  # language selection
    lang = select_language()
    print(f"Typing in {lang}...")

    # csv creation 
    csv_filename = os.path.join(directory_path, f"speech_data_{lang}.csv")

    while True:
        # manualy input the user text
        text = input(f"{cur_speaker}, please type your input: ").strip()

        # switch user
        if check_switch_command(text, lang):
            print("Who is speaking?")
            new_speaker = input("Enter the new speaker (e.g., Person 2): ")
            cur_speaker = new_speaker
            print(f"Switched to {new_speaker}")
            continue

        # termination phrase
        elif check_termination_phrase(text, lang):
            speak_termination_prompt()
            decision = input("Do you want to terminate the process? (yes/no): ")
            if decision.lower() == 'yes':
                print("Terminating the process.")
                break
            else:
                print("Resuming...")

        # if the language is greek it will only translate, simplify cant work not enough datasets in greek trained
        if lang == 'el-GR':
            simplified_text = text
        else:
            # used to choose between translate or simplify
            if mode_choice == 'simplify':
                simplified_text = simplify_text_for_asl(text)
            else:
                simplified_text = text  # if anything else than simplify it will just translate the text  

        #print text to ASL
        print(f"Processed Text: {simplified_text}")

        # display the text to ASL images
        display_images_sequentially(simplified_text, lang, def_delay)

        # timestamps for csv
        timestamp = datetime.now().strftime("%H:%M:%S")

        # save data with the timestamp,text,simplified_text and the speaker 
        save_to_csv(csv_filename, timestamp, text, simplified_text, cur_speaker)

        
