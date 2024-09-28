import os
from PIL import Image, ImageTk
import tkinter as tk
from transformers import pipeline
from transformers import T5ForConditionalGeneration, T5Tokenizer,BartForConditionalGeneration, BartTokenizer

# path of ASL photos, should change in case of swapping folders.
AS_IMAGE_DIRECTORY = r"C:\Users\theap\Desktop\project-main\project-main\ASL_Photos"
GS_IMAGE_DIRECTORY = r"C:\Users\theap\Desktop\project-main\project-main\GSL_Photos"


# token is used to map input text to smaller portions so the model can handle them better.
# used to load the bart model.
model_path = r"C:\Users\theap\Desktop\project-main\project-main\model-bart"
model = BartForConditionalGeneration.from_pretrained(model_path)
token = BartTokenizer.from_pretrained(model_path)

# unpin the part below and pin the above to load the t5-small model.
# model_path = r"C:\Users\theap\Desktop\project-main\project-main\model"
# model = T5ForConditionalGeneration.from_pretrained(model_path)
# token = T5Tokenizer.from_pretrained(model_path)


def simplify_text_for_asl(input_text):
    try:
        # split the input text to subwords to get a more effective result.
        input_ids = token.encode(input_text, return_tensors="pt")

        # used to generate the output of my pre-trained model.
        outputs = model.generate(input_ids, max_length=500, num_beams=4, early_stopping=True)

        # maps the output tokens to words so the text can be readable by the user.
        simplified_text = token.decode(outputs[0], skip_special_tokens=True)
        return simplified_text
    except Exception as e:
        print(f"Error: {e}")
        return input_text  # used to return original text if simplification fails 

# map each letter of users input to corresponding ASL sign image.
def map_text_to_asl_images(text, language):
    # convert text to uppercase to make the process easier.
    text = text.upper()

    # list to hold path of images to each letter in the input.
    image_paths = []

    if language == 'en-US':
        directory = AS_IMAGE_DIRECTORY
    elif language == 'el-GR':
        directory = GS_IMAGE_DIRECTORY
    else:
        print("Invalid choice, setting ASL as default.")
        directory = AS_IMAGE_DIRECTORY

    # iterate through each letter in the text.
    for letter in text:
        if letter in [' ', "'",'"',':','[',']','.',',']:  # for special characters.
            continue

        # construct filename of the corresponding ASL image.
        if isinstance(letter, str):
            image_name = f"Sign_language_{letter}.png"
        else:
            image_name = f"Sign_language_{str(letter)}.png"  # if images have diff extensions we might have a problem, the displayer is set for transparent images.

        # full path to the ASL image.
        image_path = os.path.join(directory, image_name)

        # check if image exists.
        if os.path.exists(image_path):
            # if yes, add it to the list.
            image_paths.append(image_path)
        else:
            # if not prompt the "error".
            print(f"No ASL sign image found for '{letter}'")

    # return list of image paths.
    return image_paths

# display images sequentially with preferred delay.
def display_images_sequentially(image_paths, delay=500):
    # Tkinter window creation.
    window = tk.Tk()
    window.title("ASL Viewer")

    # canvas for displaying images.
    canvas = tk.Canvas(window, width=400, height=400)
    canvas.pack()

    # load and display each image with delay.
    for i, image_path in enumerate(image_paths):
        # open the image using PIL.
        image = Image.open(image_path).convert("RGBA")  

        # resize image to fit the canvas.
        image = image.resize((400, 400), Image.LANCZOS)

        # convert the image to a format compatible with Tkinter.
        tk_image = ImageTk.PhotoImage(image)

        # clear previous image.
        canvas.delete("all")

        # display new image on the canvas.
        canvas.create_image(0, 0, anchor=tk.NW, image=tk_image)

        # update the canvas to display new image.
        canvas.update()

        # check for last image.
        if i == len(image_paths) - 1:
            # if its last close window.
            window.after(delay, window.destroy)
        else:
            # initiate desired delay.
            window.after(delay)

    window.mainloop()
