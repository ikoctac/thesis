import os
from PIL import Image, ImageTk
import tkinter as tk
from transformers import T5ForConditionalGeneration, T5Tokenizer
from dotenv import load_dotenv

# Load paths from .env
load_dotenv()
current_window = None

# Path for ASL AND GSL (still need to load from .env)
ASL_DIR = os.getenv("ASL")
GSL_DIR = os.getenv("GSL")

model_name = os.getenv("MODEL_T5")
model = T5ForConditionalGeneration.from_pretrained(model_name)
token = T5Tokenizer.from_pretrained(model_name)

def simplify_text_for_asl(input_text):
    try:
        input_text = "simplify: " + input_text
        input_ids = token.encode(input_text, return_tensors="pt", max_length=512, truncation=True)

        summary_ids = model.generate(
            input_ids,
            num_beams=6,
            no_repeat_ngram_size=2,
            length_penalty=1.0,
            min_length=20,
            max_length=100,
            early_stopping=True
        )

        simplified_text = token.decode(summary_ids[0], skip_special_tokens=True)
        return simplified_text

    except Exception as e:
        print(f"Error: {e}")
        return input_text  # Return original text if simplification fails

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
        if letter in [' ', "'", '"', ':', '[', ']', '.', ',', '-', '_']:
            # Treat spaces as None and skip adding an image path for them
            image_paths.append(None)
            continue

        image_name = f"Sign_language_{letter}.png"
        image_path = os.path.join(directory, image_name)

        if os.path.exists(image_path):
            image_paths.append(image_path)
        else:
            print(f"No ASL sign image found for '{letter}'")
            image_paths.append(None)  # Append None if the image is not found

    return image_paths

def display_images_sequentially(text, language, delay, word_gap=1):
    
    # Generate image paths using the mapping function
    image_paths = map_text_to_asl_images(text, language)

    if not image_paths:
        print("No images to display. Check your text or language input.")
        return

    # Create the main window
    window = tk.Tk()
    window.title("Sign Language Viewer")
    window.geometry("1920x1080")

    # Main display canvas for the large image
    canvas = tk.Canvas(window, width=150, height=150, bg="white")
    canvas.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

    # Frame for thumbnails with scrollbar
    thumbnail_frame = tk.Frame(window)
    thumbnail_frame.grid(row=1, column=0, sticky="nsew")

    # Configure grid weights
    window.rowconfigure(0, weight=1)
    window.rowconfigure(1, weight=4)
    window.columnconfigure(0, weight=1)

    # Create a canvas inside the thumbnail frame
    thumbnail_canvas = tk.Canvas(thumbnail_frame, bg="lightgray")
    thumbnail_canvas.pack(side="left", fill="both", expand=True)

    # Add a scrollbar to the thumbnail canvas
    scrollbar = tk.Scrollbar(thumbnail_frame, orient="vertical", command=thumbnail_canvas.yview)
    scrollbar.pack(side="right", fill="y")

    thumbnail_canvas.configure(yscrollcommand=scrollbar.set)
    thumbnail_canvas.bind(
        "<Configure>",
        lambda e: thumbnail_canvas.configure(scrollregion=thumbnail_canvas.bbox("all")),
    )

    # Create an inner frame to hold thumbnails
    inner_frame = tk.Frame(thumbnail_canvas, bg="lightgray")
    thumbnail_canvas.create_window((0, 0), window=inner_frame, anchor="nw")

    # Thumbnail settings
    thumbnail_size = 30
    max_thumbnails_per_row = 33
    current_row = 0
    current_column = 0

    # Store references to avoid garbage collection
    image_refs = []

    def load_thumbnail(path):
        try:
            thumbnail = Image.open(path).convert("RGBA").resize((thumbnail_size, thumbnail_size), Image.LANCZOS)
            return ImageTk.PhotoImage(thumbnail)
        except Exception as e:
            print(f"Error loading thumbnail {path}: {e}")
            return None

    def display_image(i):
        nonlocal current_row, current_column

        if i >= len(image_paths):
            return

        # Skip None values in image_paths (for missing images or spaces)
        if image_paths[i] is None:
            if text[i] == " ":
                # If it's a space, add a gap between words
                current_column += word_gap
            # Continue to the next image in the sequence
            window.after(delay, display_image, i + 1)
            return

        # Display the main image
        try:
            main_image = Image.open(image_paths[i]).convert("RGBA")
            resized_image = main_image.resize((150, 150), Image.LANCZOS)
            tk_image = ImageTk.PhotoImage(resized_image)

            canvas.delete("all")
            canvas.create_image(0, 0, anchor=tk.NW, image=tk_image)
            canvas.image = tk_image  # Keep reference to avoid garbage collection
            image_refs.append(tk_image)

        except Exception as e:
            print(f"Error displaying main image: {e}")
            return

        # Display thumbnail
        thumb = load_thumbnail(image_paths[i])
        if thumb:
            thumbnail_label = tk.Label(inner_frame, image=thumb, bg="lightgray")
            thumbnail_label.grid(row=current_row, column=current_column, padx=5, pady=5)
            image_refs.append(thumb)

        # Add letter as text below thumbnail
        letter = os.path.splitext(os.path.basename(image_paths[i]))[0].split("_")[-1]
        text_label = tk.Label(inner_frame, text=letter, bg="lightgray", font=("Arial", 12), fg="black")
        text_label.grid(row=current_row + 1, column=current_column, padx=5, pady=5)

        current_column += 1
        if current_column >= max_thumbnails_per_row:
            current_column = 0
            current_row += 2

        # Schedule next image
        window.after(delay, display_image, i + 1)

    # Close the program when pressing 'Esc'
    window.bind("<Escape>", lambda event: window.destroy())

    display_image(0)
    window.mainloop()