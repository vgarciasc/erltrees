import tkinter as tk
from tkinter import scrolledtext
import graphviz
from PIL import ImageTk, Image
from erltrees.evo.evo_tree import Individual
from erltrees.rl.configs import get_config
from erltrees.viz.graph_utils import convert_tree_to_tree_string, convert_tree_string_to_dot
from erltrees.rl.utils import fill_metrics

# Create the Cartpole environment

config = get_config("cartpole")
tree_str = "\n- Pole Angular Velocity <= 0.074\n-- Pole Angle <= 0.022\n--- LEFT\n--- RIGHT\n-- RIGHT"

# config = get_config("mountain_car")
# tree_str = "\n- Car Velocity <= -0.00010\n-- Car Position <= -0.94656\n--- RIGHT\n--- LEFT\n-- Car Velocity <= 0.01796\n--- Car Position <= -0.38836\n---- RIGHT\n---- LEFT\n--- RIGHT"
# tree_str = """
# - Car Velocity <= -0.00700
# -- LEFT
# -- Car Velocity <= 0.40550
# --- Car Position <= -0.09594
# ---- RIGHT
# ---- LEFT
# --- RIGHT"""
#
config = get_config("lunar_lander")
tree_str = "\n- Leg 1 is Touching <= 0.50000\n-- Y Velocity <= -0.09085\n--- Angle <= -0.04364\n---- Y Velocity <= -0.25810\n----- Y Position <= 0.20415\n------ MAIN ENGINE\n------ Angular Velocity <= -0.18925\n------- LEFT ENGINE\n------- X Velocity <= -0.17510\n-------- MAIN ENGINE\n-------- Angular Velocity <= -0.02175\n--------- LEFT ENGINE\n--------- MAIN ENGINE\n----- X Velocity <= 0.02710\n------ RIGHT ENGINE\n------ LEFT ENGINE\n---- Y Velocity <= -0.28725\n----- X Velocity <= -0.39615\n------ RIGHT ENGINE\n------ MAIN ENGINE\n----- Angle <= 0.21595\n------ X Position <= -0.02269\n------- RIGHT ENGINE\n------- Angular Velocity <= 0.18515\n-------- LEFT ENGINE\n-------- MAIN ENGINE\n------ RIGHT ENGINE\n--- Y Position <= 0.00074\n---- NOP\n---- Angle <= 0.02441\n----- LEFT ENGINE\n----- RIGHT ENGINE\n-- Y Velocity <= -0.06200\n--- MAIN ENGINE\n--- Angle <= -0.21080\n---- LEFT ENGINE\n---- NOP"

agent = Individual.read_from_string(config, tree_str)
# agent.denormalize_thresholds()
# tree_str = str(agent)

# Create a Tkinter window
window = tk.Tk()
window.title("Decision Tree Agent Evaluation")
window.geometry("1300x800+400+100")
window.columnconfigure(0, weight=2, uniform="x")
window.columnconfigure(1, weight=1, uniform="x")
window.rowconfigure(0, weight=1, uniform="y")
window.rowconfigure(1, weight=1, uniform="y")

tree_rendered_frame = tk.Frame(window)
tree_rendered_frame.grid(row=0, column=0, columnspan=1, sticky="nsew")
# tree_rendered_frame.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
tree_rendered_container = tk.LabelFrame(tree_rendered_frame, text="Current Decision Tree Agent", borderwidth=1,
                                        relief="solid")
tree_rendered_container.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
tree_rendered_container_frame = tk.Frame(tree_rendered_container, background="white")
tree_rendered_container_frame.pack(fill=tk.BOTH, expand=True)
tree_rendered = tk.Label(tree_rendered_container_frame, image=tk.PhotoImage(), background="white")
tree_rendered.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)

gym_canvas_frame = tk.Frame(window)
gym_canvas_frame.grid(row=0, column=1, columnspan=1, sticky="nsew")
# gym_canvas_frame.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
gym_canvas_container = tk.LabelFrame(gym_canvas_frame, text="OpenAI Gym Environment")
gym_canvas_container.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
gym_canvas = tk.Label(gym_canvas_container, image=tk.PhotoImage(), background="black")
gym_canvas.pack(fill=tk.BOTH, expand=True)

bottom_left_frame = tk.Frame(window)
bottom_left_frame.grid(row=1, column=0, columnspan=1, sticky="nsew")
bottom_left_frame.columnconfigure(0, weight=2, uniform="x")
bottom_left_frame.columnconfigure(1, weight=1, uniform="x")

tree_string_frame = tk.Frame(bottom_left_frame)
tree_string_text = scrolledtext.ScrolledText(bottom_left_frame, font=("Courier", 12), fg="black", bg="white")
tree_string_text.insert(tk.INSERT, tree_str)
tree_string_text.grid(row=0, column=0, columnspan=1, sticky="nsew")
tree_string_text.bind("<KeyRelease>", lambda e: update_tree(tree_string_text.get("1.0", tk.END)))


def update_tree(tree_string):
    global agent
    try:
        agent = Individual.read_from_string(config, tree_string[:-1])  # remove trailing newline
    except:
        pass


state_frame = tk.Frame(bottom_left_frame)
state_frame.grid_columnconfigure(0, weight=1)
state_frame.grid(row=0, column=1, columnspan=1, sticky="nsew")
# state_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
attribute_labels = []

for i, (attribute, _, _) in enumerate(config["attributes"]):
    state_frame.columnconfigure(i//4, weight=1, uniform="x")
    state_frame.rowconfigure(i%4, weight=1, uniform="y")
    attribute_label = tk.LabelFrame(state_frame, text=attribute)
    attribute_label.grid(row=i%4, column=i//4, sticky="nsew", padx=5, pady=5)
    attribute_value_label = tk.Label(attribute_label, text="0.0", font=("Courier", 18))
    attribute_value_label.pack(side=tk.LEFT, padx=5, pady=5, expand=True, fill=tk.X)
    attribute_labels.append(attribute_value_label)

bottom_right_frame = tk.Frame(window)
bottom_right_frame.grid(row=1, column=1, columnspan=1, sticky="nsew")
bottom_right_frame.columnconfigure(0, weight=1, uniform="x")
bottom_right_frame.columnconfigure(1, weight=1, uniform="x")
bottom_right_frame.columnconfigure(2, weight=1, uniform="x")
bottom_right_frame.rowconfigure(0, weight=3, uniform="y")
bottom_right_frame.rowconfigure(1, weight=1, uniform="y")

log_frame = tk.Frame(bottom_right_frame)
log_frame.grid(row=0, column=0, columnspan=3)
log_label = scrolledtext.ScrolledText(log_frame, font=("Courier", 12), fg="white", bg="black")
log_label.insert(tk.END, "")
log_label.pack(fill=tk.BOTH, expand=True)

# scrollbar = tk.Scrollbar(log_frame)
# scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
# log_label.config(yscrollcommand=scrollbar.set)
# scrollbar.config(command=log_label.yview)

episodes_label = tk.Label(bottom_right_frame, text="Episodes", font=("Courier", 12))
episodes_label.grid(row=1, column=0, padx=5, pady=5, sticky="e")
episodes_text = tk.Text(bottom_right_frame, font=("Courier", 12), fg="black", bg="white", width=20, height=1)
episodes_text.insert(tk.INSERT, "100")
episodes_text.grid(row=1, column=1, padx=5, pady=5)
episodes_launch_button = tk.Button(bottom_right_frame, text="Launch", command=lambda : launch_episodes())
episodes_launch_button.grid(row=1, column=2, padx=5, pady=5, sticky="w")


def launch_episodes():
    try:
        episodes = int(episodes_text.get("1.0", tk.END))
    except:
        episodes = 10
        print("Invalid episode quantity. Using 10")
    
    agent = Individual.read_from_string(config, "\n" + tree_string_text.get("1.0", tk.END)[:-1])
    fill_metrics(config, [agent], alpha=0.001, episodes=episodes, task_solution_threshold=config["task_solution_threshold"])
    log_message(f"Ran for {episodes} episodes.\n" +
                f"Average reward: {agent.reward:.3f} +- {agent.std_reward:.3f}\n" +
                f"Success rate: {agent.success_rate:.3f}\n\n")
    

env = config["maker"]()


def log_message(msg):
    log_label.configure(state='normal')
    log_label.insert(tk.END, msg)
    log_label.configure(state='disabled')
    log_label.see(tk.END)


def log_state(state):
    for i, s in enumerate(state):
        attribute_labels[i].configure(text=f'{s:.2f}')


# Loop to update the environment and display frames
for episode in range(100):
    # Initialize the Cartpole environment
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        # Render the current frame in the Cartpole environment
        image = env.render(mode='rgb_array')
        image = Image.fromarray(image)
        # image.thumbnail((max(500, gym_canvas.winfo_width() - 10), max(500, gym_canvas.winfo_height() - 10)))
        image.thumbnail((500, 500))
        rendered_image = ImageTk.PhotoImage(image)
        gym_canvas["image"] = rendered_image

        # Randomly select an action
        action = agent.act(state)

        # Render the current tree
        tree_string = convert_tree_to_tree_string(config, agent, state)
        dot_string = convert_tree_string_to_dot(tree_string)
        graph = graphviz.Source(dot_string, format="png")
        graph.render(filename="tmp_image", cleanup=True)
        tree_image = Image.open("tmp_image.png")
        tree_image.thumbnail((900, 500))
        tree_image_tk = ImageTk.PhotoImage(tree_image)
        tree_rendered["image"] = tree_image_tk
        log_state(state)

        # Perform the action in the environment
        state, reward, done, info = env.step(action)
        total_reward += reward

        # Update the Tkinter window
        window.update()

    log_message(f"Episode #{str(episode + 1).ljust(2, ' ')}: Reward = {total_reward}\n")

# Close the Cartpole environment
env.close()

# Close the Tkinter window when the user closes it
# window.mainloop()
