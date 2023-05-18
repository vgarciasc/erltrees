# Import the required libraries
from datetime import datetime
import tkinter as tk
import tkinter.ttk as ttk
import pydotplus
from PIL import ImageTk, Image
import io
from graph_utils import convert_tree_string_to_dot
from rich import print

decision_tree_string = """- maize growing stage <= 2
-- biomass above <= 104.309
--- -0.99670523
--- cumulative N app. <= 66.934
---- -0.89922157
---- -0.97816033
-- -0.99997996"""
last_valid_image = None

def get_current_image():
    global last_valid_image

    try:
        dot_string = convert_tree_string_to_dot(label_text.get("1.0", tk.END))
        graph = pydotplus.graph_from_dot_data(dot_string)
        image = Image.open(io.BytesIO(graph.create_png()))
        image.thumbnail((400, 400), Image.ANTIALIAS)
        last_valid_image = image
        return image
    except:
        return last_valid_image

def update_image():
    rendered_image = ImageTk.PhotoImage(get_current_image())
    tree_image["image"] = rendered_image

def save_image():
    image = get_current_image()
    image.save(f"tree_{timestamp}.png")
    print(f"Saved to {timestamp}.png")

# Get datetime as string format 2023-01-01_00-00-00
now = datetime.now()
timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

window = tk.Tk()
window.geometry("+550+200")

frame_left = tk.Frame(width=300)
frame_left.pack(side=tk.LEFT, expand=True, fill=tk.X)
frame_right = tk.Frame(width=200)
frame_right.pack(side=tk.LEFT, expand=True, fill=tk.X)

tree_image_container = tk.LabelFrame(frame_left, text="Decision Tree")
tree_image_container.pack(padx=10, pady=10, fill=tk.Y, expand=True)
tree_image = tk.Label(tree_image_container, image=tk.PhotoImage(), width=400, height=400, bg="white")
# tree_image.rendered_image = rendered_image
tree_image.pack(padx=10, pady=10)
tree_image_save_btn = ttk.Button(tree_image_container, text="Save Image", command=save_image)
tree_image_save_btn.pack(side=tk.LEFT, padx=10, pady=10)

label_text_container = tk.LabelFrame(frame_right, text="Decision Tree String", width=20)
label_text_container.pack(padx=10, pady=10, fill=tk.X, expand=True)
label_text = tk.Text(label_text_container, font=("Courier", 12), width=50)
label_text.insert(tk.END, decision_tree_string)
label_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
label_text.bind("<KeyRelease>", lambda event: update_image())

update_image()

window.mainloop()