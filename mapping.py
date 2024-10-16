import os
from PIL import Image, ImageTk
import tkinter as tk
from transformers import T5ForConditionalGeneration, T5Tokenizer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Path of ASL and GSL photos
AS_IMAGE_DIRECTORY = os.getenv("AS_IMAGE_DIRECTORY")
GS_IMAGE_DIRECTORY = os.getenv("GS_IMAGE_DIRECTORY")

# Load T5 model and tokenizer
model_path = os.getenv("MODEL_T5_4EPOCH_PATH")
model = T5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = T5Tokenizer.from_pretrained(model_path)

# Function to simplify the input text with the pre-trained model
def simplify_text_for_asl(input_text):
    try:
        # Simplify the text using the pre-trained model
        input_ids = tokenizer.encode(input_text, return_tensors="pt")
        outputs = model.generate(input_ids, max_length=500, num_beams=4, early_stopping=True)
        simplified_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return simplified_text

    except Exception as e:
        print(f"Error: {e}")
        return input_text  # Return original text if simplification fails

# Map each letter of user's input to corresponding ASL sign image
def map_text_to_asl_images(text, language):
    text = text.upper()
    image_paths = []

    if language == 'en-US':
        directory = AS_IMAGE_DIRECTORY
    elif language == 'el-GR':
        directory = GS_IMAGE_DIRECTORY
    else:
        print("Invalid choice, setting ASL as default.")
        directory = AS_IMAGE_DIRECTORY

    for letter in text:
        if letter in [' ', "'", '"', ':', '[', ']', '.', ',']:
            continue
        image_name = f"Sign_language_{letter}.png"
        image_path = os.path.join(directory, image_name)

        if os.path.exists(image_path):
            image_paths.append(image_path)
        else:
            print(f"No ASL sign image found for '{letter}'")

    return image_paths

# Display images sequentially with a preferred delay
def display_images_sequentially(image_paths, delay=500):
    window = tk.Tk()
    window.title("ASL Viewer")
    canvas = tk.Canvas(window, width=400, height=400)
    canvas.pack()

    def display_image(i):
        if i >= len(image_paths):
            window.destroy()
            return

        image = Image.open(image_paths[i]).convert("RGBA")
        image = image.resize((400, 400), Image.LANCZOS)
        tk_image = ImageTk.PhotoImage(image)

        canvas.delete("all")
        canvas.create_image(0, 0, anchor=tk.NW, image=tk_image)
        canvas.image = tk_image

        window.after(delay, display_image, i + 1)

    display_image(0)
    window.mainloop()
