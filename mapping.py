import os
from PIL import Image, ImageTk
import tkinter as tk
from transformers import BartForConditionalGeneration, BartTokenizer
from dotenv import load_dotenv

# Load paths from .env
load_dotenv()

# Path for ASL AND GSL (still need to load from .env)
ASL_DIR = os.getenv("ASL")
GSL_DIR = os.getenv("GSL")

# unpin to use for bart model
model_dir = os.getenv("MODEL_BART")
model = BartForConditionalGeneration.from_pretrained(model_dir)
token = BartTokenizer.from_pretrained(model_dir)

# Function to simplify text (for summarization)
def simplify_text_for_asl(input_text):
    try:
        # Prepend the task prefix for summarization (for BART model)
        input_text = "summarize: " + input_text
        
        # Tokenize the input text
        input_ids = token.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
        
        # Generate summary ids with beam search and penalties to control repetitiveness
        summary_ids = model.generate(
            input_ids,
            num_beams=4,
            no_repeat_ngram_size=2,
            length_penalty=2.0,
            min_length=30,
            max_length=150,
            early_stopping=True
        )
        
        # Decode the generated summary
        simplified_text = token.decode(summary_ids[0], skip_special_tokens=True)
        return simplified_text

    except Exception as e:
        print(f"Error: {e}")
        return input_text  # Return original text if summarization fails

# Function to map input letters to each ASL/GSL needed
def map_text_to_asl_images(text, language):
    text = text.upper()  # Convert text to uppercase for consistency
    image_paths = []

    # Set the directory based on the language choice
    if language == 'en-US':
        directory = ASL_DIR
    elif language == 'el-GR':
        directory = GSL_DIR
    else:
        print("Invalid choice, setting ASL as default.")
        directory = ASL_DIR

    # Iterate over each character in the text
    for letter in text:
        # Skip characters that are not letters or numbers
        if letter in [' ', "'", '"', ':', '[', ']', '.', ',', '-', '_']:
            continue

        # Construct the image filename
        image_name = f"Sign_language_{letter}.png"
        image_path = os.path.join(directory, image_name)

        # Check if the image exists in the directory and add it to the list
        if os.path.exists(image_path):
            image_paths.append(image_path)
        else:
            print(f"No ASL sign image found for '{letter}'")

    return image_paths

# Function to display images sequentially with a delay
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

# Function to integrate all tasks
def process_and_display(input_text, language='en-US'):
    # Simplify the input text first
    simplified_text = simplify_text_for_asl(input_text)

    # Map the simplified text to ASL images
    image_paths = map_text_to_asl_images(simplified_text, language)

    # Display the images sequentially
    display_images_sequentially(image_paths)
