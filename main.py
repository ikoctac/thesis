import speech_recognition as sr
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # ML Model
from datetime import datetime
from functions import *
from mapping import *  # importing necessary functions from the secondary program.

# initalize recognizer.
recognizer = sr.Recognizer()

# define current speaker.
current_speaker = "person 1"

# get the current directory of the script.
current_directory = os.path.dirname(os.path.abspath(__file__))

# directory path where you save the CSV file, this way we can create a dataset.
directory_path = os.path.join(current_directory, "csv_files")

# user chooses if he will type or speak. 
input_mode = input("do you want to type or speak? (type/speak): ").strip().lower()

if input_mode == 'speak':
    # use the default microphone as source, bot asks what language will the user speak and then prompt the available languages.
    with sr.Microphone() as source:
        speak_language_prompt()  # bot asks user what they will use (Function is in separate py file.)
        language = select_language()
        print(f"listening {language}...")

        # create the csv_filename
        csv_filename = os.path.join(directory_path, f"speech_data_{language}.csv")

        # adjust for ambient noise if necessary.
        recognizer.adjust_for_ambient_noise(source)

        # continuously listen for speech.
        while True:
            try:
                # capture users audio 
                audio = recognizer.listen(source)

                # converts speech to text using the selected language.
                text = recognizer.recognize_google(audio, language=language)

                # get timestamp for csv reference.
                timestamp = datetime.now().strftime("%H:%M:%S")

                # print what the user said. uncomment if you want the original text printed
                # print(f"{current_speaker} said:", text)

                # output of the trained model based on the dataset provided(the loaded model location is in mapping.py and is loaded from import.)
                simplified_text = simplify_text_for_asl(text)
                print(f"simplified Text: {simplified_text}")

                # save the data to CSV file with speaker identifier.
                save_to_csv(csv_filename, timestamp, text, simplified_text, current_speaker)

                # used to map the converted/translated text to the pictures in my files (function is imported from mapping.py and is loaded from import.)
                asl_images = map_text_to_asl_images(simplified_text, language)

                # display ASL images sequentially.
                display_images_sequentially(asl_images)

                # check for switch commands( used to differantiate between users for a cleaner dataset, also imported from functions.py.) 
                if check_switch_command(text, language):
                    print(" Who is speaking?")
                    new_speaker = input("Enter the new speaker (e.g., Person 2): ")
                    current_speaker = new_speaker
                    print(f"Switched to {new_speaker}")

                # check for termination phrase( only works when users says a termination phrase once.)
                elif check_termination_phrase(text, language):
                    speak_termination_prompt()
                    decision = input("Do you want to terminate the process? (yes/no): ")
                    if decision.lower() == 'yes':
                        print("Terminating the process.")
                        break  # termination of the program.
                    else:
                        print("Resuming...")

            # used to identify any audio problems when user is speaking ( it informs if it doenst catch what the user said.)
            except sr.UnknownValueError:
                print("Audio output has a problem.")
            except sr.RequestError as e:
                print("Error; {0}".format(e))

elif input_mode == 'type':
    # used when user types "type" in the start of the program to choose to manually enter what he is going to say( make it more functional and faster because the google recognizer needs fast internet access and thats not always the case.)
    speak_language_prompt()  # bot asks user what language they will use (Function is in a separate py file.)
    language = select_language() # uses the selected language.
    print(f"typing in {language}...") 

    # create the csv_filename
    csv_filename = os.path.join(directory_path, f"speech_data_{language}.csv")

    while True:
        # user manually inputs the paragraph he wants to translate.
        text = input(f"{current_speaker}, please type your input: ").strip()

        # get timestamp for csv reference
        timestamp = datetime.now().strftime("%H:%M:%S")

        # print what the user typed
        # print(f"{current_speaker} typed:", text)

        simplified_text = simplify_text_for_asl(text)
        print(f"simplified Text: {simplified_text}")

         # save the data to CSV file with speaker identifier.
        save_to_csv(csv_filename, timestamp, text, simplified_text, current_speaker)

        # used to map the text to asl images in my files(Function is imported from another file mapping.py)
        asl_images = map_text_to_asl_images(simplified_text, language)

        # display ASL images sequentially.
        display_images_sequentially(asl_images)

        # check for switch commands( only works when user types one time the command and not in a sentence.)
        if check_switch_command(text, language):
            print("Who is speaking?")
            new_speaker = input("Enter the new speaker (e.g., Person 2): ")
            current_speaker = new_speaker
            print(f"Switched to {new_speaker}")

        # check for termination phrase( only works when user types one time the command and not in a sentence.)
        elif check_termination_phrase(text, language):
            speak_termination_prompt()
            decision = input("Do you want to terminate the process? (yes/no): ")
            if decision.lower() == 'yes':
                print("Terminating the process.")
                break  # terminates the program .
            else:
                print("Resuming...")

# used to catch invalid input from the user so the program doesnt crash.
else:
    print("choose 'type' or 'speak'.")
