import tkinter as tk
from PIL import Image, ImageTk

def display_images_with_words(image_paths, texts, delay=500):
    if not image_paths or not texts or len(image_paths) != len(texts):
        print("Image paths and texts must be non-empty and of the same length.")
        return

    # Create the main window
    window = tk.Tk()
    window.title("American Sign Language Viewer")
    window.geometry("800x600")

    # Main display canvas (for large images)
    canvas = tk.Canvas(window, width=400, height=300, bg="white")
    canvas.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

    # Frame for thumbnails and text
    thumbnail_frame = tk.Frame(window, bg="lightgray")
    thumbnail_frame.grid(row=1, column=0, sticky="nsew")

    # Configure grid weights
    window.rowconfigure(0, weight=1)  # Main display
    window.rowconfigure(1, weight=1)  # Thumbnails and text
    window.columnconfigure(0, weight=1)

    # Thumbnail size and positioning
    thumbnail_size = 80  # Thumbnail dimensions
    max_thumbnails_per_row = 10  # Maximum number of thumbnails per row

    # Initialize current_row and current_column
    current_row = 0
    current_column = 0

    # Store references to avoid garbage collection
    image_refs = []  # For main images and thumbnails
    thumbnails = []   # Store thumbnails for reuse

    def load_thumbnail(path):
        try:
            thumbnail = Image.open(path).convert("RGBA").resize((thumbnail_size, thumbnail_size), Image.LANCZOS)
            return ImageTk.PhotoImage(thumbnail)
        except Exception as e:
            print(f"Error loading thumbnail {path}: {e}")
            return None

    # Pre-load thumbnails to avoid repeated loading
    for path in image_paths:
        if path:
            thumb = load_thumbnail(path)
            if thumb is not None:
                thumbnails.append(thumb)
            else:
                thumbnails.append(None)
        else:
            thumbnails.append(None)

    def display_image(i):
        nonlocal current_row, current_column

        if i >= len(image_paths):
            return

        # Display the main image
        if image_paths[i]:
            try:
                main_image = Image.open(image_paths[i]).convert("RGBA")
                resized_image = main_image.resize((400, 300), Image.LANCZOS)
                tk_image = ImageTk.PhotoImage(resized_image)

                canvas.delete("all")
                canvas.create_image(0, 0, anchor=tk.NW, image=tk_image)
                canvas.image = tk_image  # Keep reference to avoid black image
                image_refs.append(tk_image)  # Keep reference for the main image

            except Exception as e:
                print(f"Error loading main image {image_paths[i]}: {e}")
                return

        # Add thumbnail and text
        if texts[i] == " ":
            current_row += 2
            current_column = 0
        else:
            if thumbnails[i]:
                thumbnail_label = tk.Label(thumbnail_frame, image=thumbnails[i], bg="lightgray")
                thumbnail_label.grid(row=current_row, column=current_column, padx=5, pady=5)

            text_label = tk.Label(thumbnail_frame, text=texts[i], bg="lightgray", font=("Arial", 12), fg="black")
            text_label.grid(row=current_row + 1, column=current_column, padx=5, pady=5)

            current_column += 1
            if current_column >= max_thumbnails_per_row:
                current_column = 0
                current_row += 2

        # Schedule the next image
        window.after(delay, display_image, i + 1)

    # Close the program when pressing 'Esc'
    window.bind("<Escape>", lambda event: window.destroy())

    display_image(0)
    window.mainloop()

# Example usage with your paths and texts.
image_paths = [
    r"C:\Users\theap\Desktop\thesis\ASL_Photos\Sign_language_H.png",
    r"C:\Users\theap\Desktop\thesis\ASL_Photos\Sign_language_E.png",
    r"C:\Users\theap\Desktop\thesis\ASL_Photos\Sign_language_L.png",
    r"C:\Users\theap\Desktop\thesis\ASL_Photos\Sign_language_L.png",
    r"C:\Users\theap\Desktop\thesis\ASL_Photos\Sign_language_O.png",
    None,
    r"C:\Users\theap\Desktop\thesis\ASL_Photos\Sign_language_W.png",
    r"C:\Users\theap\Desktop\thesis\ASL_Photos\Sign_language_O.png",
    r"C:\Users\theap\Desktop\thesis\ASL_Photos\Sign_language_R.png",
    r"C:\Users\theap\Desktop\thesis\ASL_Photos\Sign_language_L.png",
    r"C:\Users\theap\Desktop\thesis\ASL_Photos\Sign_language_D.png"
]

texts = ["H", "E", "L", "L", "O", " ", "W", "O", "R", "L", "D"]

# Call the function with your paths and texts.
display_images_with_words(image_paths, texts, delay=2000)