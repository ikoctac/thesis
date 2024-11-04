import os
from PIL import Image, ImageTk
import tkinter as tk
from transformers import T5ForConditionalGeneration, T5Tokenizer, BartTokenizer, BartForConditionalGeneration
from dotenv import load_dotenv

# load paths from .env
load_dotenv()

# path for ASL AND GSL
ASL_DIR = os.getenv("ASL")
GSL_DIR = os.getenv("GSL")

# # load prefered model and its tokens
# model_dir = os.getenv("MODEL_T5")
# model = T5ForConditionalGeneration.from_pretrained(model_dir)
# token = T5Tokenizer.from_pretrained(model_dir)

# unpin to use for bart model
model_dir = os.getenv("MODEL_BART")
model = BartForConditionalGeneration.from_pretrained(model_dir)
token = BartTokenizer.from_pretrained(model_dir)

def simplify_text_for_asl(input_text):
    try:
        # Prepend the task prefix for summarization
        input_text = f"summarize: {input_text}"

        # Tokenize the input text
        input_ids = token.encode(input_text, return_tensors='pt', max_length=512, truncation=True)

        # Generate summary ids
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
    
# # summary/simplify func for the choosen model
# def simplify_text_for_asl(input_text):
#     try:
#         # summary/simplify the input text
#         input_ids = token.encode(input_text, return_tensors="pt")
#         outputs = model.generate(input_ids, max_length=500, num_beams=4, early_stopping=True)
#         simplified_text = token.decode(outputs[0], skip_special_tokens=True)
#         return simplified_text

#     except Exception as e:
#         print(f"Error: {e}")
#         return input_text  # escape clause if summary fails

# map input letters to each ASL/GSL needed
def map_text_to_asl_images(text, language):
    text = text.upper()
    image_paths = []

    if language == 'en-US':
        directory = ASL_DIR
    elif language == 'el-GR':
        directory = GSL_DIR
    else:
        print("Invalid choice, setting ASL as default.")
        directory = ASL_DIR

    for letter in text:
        if letter in [' ', "'", '"', ':', '[', ']', '.', ',','-','-']:
            continue
        image_name = f"Sign_language_{letter}.png"
        image_path = os.path.join(directory, image_name)

        if os.path.exists(image_path):
            image_paths.append(image_path)
        else:
            print(f"No ASL sign image found for '{letter}'")

    return image_paths

# display image with delay
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
