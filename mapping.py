import os
from PIL import Image, ImageTk
import tkinter as tk
from transformers import T5ForConditionalGeneration, T5Tokenizer
from dotenv import load_dotenv

# load paths from .env, used to load paths from the env so changes can be made easier
load_dotenv()

# path for ASL AND GSL (still need to load from .env)
ASL_DIR = os.getenv("ASL")
GSL_DIR = os.getenv("GSL")

# loads model from path, model is downloaded t5 and pretrained on a dataset of normal to simple sentences to work better for summarize
model_name = os.getenv("MODEL_T5")
model = T5ForConditionalGeneration.from_pretrained(model_name) # loads model safetensors
token = T5Tokenizer.from_pretrained(model_name) # loads generated tokens

def simplify_text_for_asl(text):
    try:
        text = "simplify: " + text # used to prepare model to summarize given input text
        input_ids = token.encode(text, return_tensors="pt", max_length=512, truncation=True) # converts text into id tokens for the model and ensures with "pt" that token output is in pytorch format 

        # used to generate output from our model via given input
        summary_ids = model.generate(
            input_ids,  
            num_beams=6, # reduces risk to miss on hidden high probability from the word sequences, calculating probabilities for 6 words in the model's vocabulary 
            no_repeat_ngram_size=2, # no sequence of >2 words is repeated, usefull for generation but not when input text is repetitive
            length_penalty=1.0, # affects sentence length, a greater value will generate smaller sentences while this value encourages big sentences 
            min_length=0, # avoid error output for smaller sentences
            max_length=100, # will max out at 100 tokens long
            early_stopping=True #stop early once all beam hypotheses have produced an EOS token
        )

        simplified_text = token.decode(summary_ids[0], skip_special_tokens=True) # decodes produced token ids into readable text
        simplified_text = simplified_text.replace("simplify:", "").strip() # incase program includes the simplify from the start
        return simplified_text

    except Exception as e:
        print(f"Error: {e}")
        return text  # Return original text if simplification fails

def map_text_to_asl_images(text, lang):
    text = text.upper()
    
    image_paths = []  # creates a list of image paths based on the given input  

    if lang == 'en-US': # choose ASL if language is english
        directory = ASL_DIR
    elif lang == 'el-GR': # choose GSL if language is greek
        directory = GSL_DIR
    else:
        print("Invalid choice, setting ASL as default.") # defaults ASL cause it has more applications on my program
        directory = ASL_DIR

    for letter in text:
        if letter in [' ', "'", '"', ':', '[', ']', '.', ',', '-', '_','(',')']:
            # Treat spaces as None and skip adding an image path for them, used to avoid errors in display
            image_paths.append(None)
            continue

        image_name = f"Sign_language_{letter}.png" # used to load image names
        image_path = os.path.join(directory, image_name) # get image path 

        if os.path.exists(image_path): # searches image path
            image_paths.append(image_path)
        else:
            print(f"No ASL sign image found for '{letter}'")
            image_paths.append(None)  # Append None if the image is not found

    return image_paths

def display_images_sequentially(text, lang, delay, word_gap=1):
    
    # generate image_paths by getting input to according function
    image_paths = map_text_to_asl_images(text, lang)

    if not image_paths: # if image paths is empty or doesnt exist output error
        print("No images to display. Check your text or language input.")
        return

    window = tk.Tk() # create a tkinter window to display ASL or GSL images
    window.title("Sign Language Viewer") # window title
    window.geometry("1280x720") # windows geometry

    canvas = tk.Canvas(window, width=150, height=150, bg="white") # create widget to display the image in 
    canvas.grid(row=0, column=0, sticky="nsew", padx=5, pady=5) # position the canvas top left grid to display images in, also it fills the available space 

    thumbnail_frame = tk.Frame(window) # creates a frame widget which will be used to group other widgets together
    thumbnail_frame.grid(row=1, column=0, sticky="nsew") # places frame in row 1 and column 0 to display the main images after displaying them 

    # Configure grid weights
    window.rowconfigure(0, weight=1) # setup a small space for displaying main images
    window.rowconfigure(1, weight=4) # setupt the rest of the space for displaying the thumbnails
    window.columnconfigure(0, weight=1) # column to set images

    # Create a canvas inside the thumbnail frame
    thumbnail_canvas = tk.Canvas(thumbnail_frame, bg="lightgray") # used to create a canvas to display images
    thumbnail_canvas.pack(side="left", fill="both", expand=True) # used to manage the widgets displayed from left to right to fill the spaces

    # Add a scrollbar to the thumbnail canvas
    scrollbar = tk.Scrollbar(thumbnail_frame, orient="vertical", command=thumbnail_canvas.yview) # create a scrollbar in the thumbnail frame to create more space for bigger sentences
    scrollbar.pack(side="right", fill="y") # scrollbar positions

    thumbnail_canvas.configure(yscrollcommand=scrollbar.set)
    thumbnail_canvas.bind(
        "<Configure>",
        lambda e: thumbnail_canvas.configure(scrollregion=thumbnail_canvas.bbox("all")),
    ) # update scrollbar based on the canvas content and when the canvas is resized the scrollbar to be adjusted

    window.update_idletasks() # used to update window for scrollbar 

    # Create an inner frame to hold thumbnails
    inner_frame = tk.Frame(thumbnail_canvas, bg="lightgray")
    thumbnail_canvas.create_window((0, 0), window=inner_frame, anchor="nw")

    thumbnail_size = 30 # thumbnail size to get more thumbnails in a row
    max_thumbnails_per_row = 32 # how many thumbnails in a row
    current_row = 0 # starting row
    current_column = 0 # starting column

    image_refs = [] # cache images

    def load_thumbnail(path):
        try:
            thumbnail = Image.open(path).convert("RGBA").resize((thumbnail_size, thumbnail_size), Image.LANCZOS) # open image, resizes it and displays it in a tkinter compatible format
            return ImageTk.PhotoImage(thumbnail) # gives back image
        except Exception as e:
            print(f"Error loading thumbnail {path}: {e}")
            return None

    def display_image(i):
        nonlocal current_row, current_column # used to get the curr_row and curr_column to maintain the values when display_image() is used multiple times

        if i >= len(image_paths): # only processes valid image_paths, stops execution once images are displayed 
            return

        # if we get none in the image_paths which indicates that we get space or blank images
        if image_paths[i] is None:
            if text[i] == " ":
                gap_label = tk.Label(inner_frame, bg="lightgray", width=thumbnail_size // 10, height=1) # grey gap_label to seperate words 
                gap_label.grid(row=current_row, column=current_column, padx=5, pady=5) # place of the gap_label in the frame 
                current_column += word_gap  # adds gap
            window.after(delay, display_image, i + 1) # used to diplay images with a delay given from the user
            return

        # start diplaying the main images
        try:
            main_image = Image.open(image_paths[i]).convert("RGBA") # properly format for transparency handling (because they were displayed black)
            resized_image = main_image.resize((150, 150), Image.LANCZOS) # riseze to 150x150 for display purposes
            tk_image = ImageTk.PhotoImage(resized_image) # convert image to tkinter compatible object to fit the graphical interface
            canvas.delete("all") # delete previous main image to display the new one
            canvas.create_image(0, 0, anchor=tk.NW, image=tk_image) #creates a new image to be able to handle the new input later
            canvas.image = tk_image  # Keep reference to avoid garbage collection
            image_refs.append(tk_image) # images are stored in mem, avoid erros related to garbage and img stay vis in gui as long as needed

        except Exception as e:
            print(f"Error displaying main image: {e}")
            return

        # Display thumbnail
        thumb = load_thumbnail(image_paths[i]) # load images from image_paths for displaying in thumbnail
        window.update_idletasks() # load the window correctly
        if thumb: #prevents erros displaying none images
            thumbnail_label = tk.Label(inner_frame, image=thumb, bg="lightgray") # creates a widget in the inner frame to display the thumbnail image
            thumbnail_label.grid(row=current_row, column=current_column, padx=5, pady=5) # places the thumbanil label within the inner_frame
            image_refs.append(thumb) # cache the thumbnail image to refer in future

        # adds letters bellow thumbnails
        letter = os.path.splitext(os.path.basename(image_paths[i]))[0].split("_")[-1] # retrieves from each image the letter that we want to display 
        text_label = tk.Label(inner_frame, text=letter, bg="lightgray", font=("Times new roman", 12), fg="black") # the colour font and size of the text
        text_label.grid(row=current_row + 1, column=current_column, padx=5, pady=5) # position inside the inner frame

        current_column += 1 # goes to the next column so the new thumbnail wont overlap the last one 
        if current_column >= max_thumbnails_per_row: # used to see if we reached the max langth of the row so we go and write in the next row
            current_column = 0
            current_row += 2

        # the time we will display the next image, taking delay time and display image function
        window.after(delay, display_image, i + 1)

        window.update_idletasks() # load window correctly  

    # close the window to insert new input text in the program
    window.bind("<Escape>", lambda event: window.destroy())

    display_image(0) # starts the image display process from the first image in the list
    window.mainloop() # keeps tkinter window open and intercative so the loops end only when escape is pressed