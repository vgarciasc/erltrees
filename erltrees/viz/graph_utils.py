import pydotplus
from PIL import Image
import graphviz
import io
from erltrees.evo.evo_tree import Individual

COLORS = ["crimson", "blue", "green", "gold", "lightseagreen", "lightpink", "lightsalmon", "khaki", "indigo", "hotpink", "lightgoldenrod", "purple", "maroon"]

def convert_tree_string_to_dot(string):
    dot_string = "graph DecisionTree {"
    lines = [line.strip() for line in string.split("\n")]

    parents = [None for _ in lines]
    for idx, line in enumerate(lines):
        if "-" not in line:
            continue

        depth = line.rindex("- ") + 1
        node_label = line[depth:].strip()
        node_id = f"node_{idx}"
        parent_id = parents[depth - 1] if depth > 1 else None

        node_color = "grey95" if "<=" in node_label else "grey99"
        edge_color = "black"

        if "*" in node_label:
            node_color = "lightcoral"
            node_label = node_label[:-2] # remove " *"

        if "comm" in node_label:
            node_color = COLORS[int(node_label[-1]) - 1] if node_label[-1] != "X" else "white"
            node_label = node_label[:-5] # remove "commX"

        dot_string += f'{"  " * depth}{node_id}[label=\"{node_label}\", shape=box, style=filled, fillcolor={node_color}, color={edge_color}];\n'

        if parent_id:
            dot_string += f'{"  " * depth}{parent_id} -- {node_id}[color={edge_color}]; '

        parents[depth] = node_id

    dot_string += "}"
    dot_string = dot_string.replace("<=", "≤")
    return dot_string

def make_image_from_dot(dot_string):
    graph = graphviz.Source(dot_string)
    image_bytes = graph.pipe(format='png')
    return Image.open(io.BytesIO(image_bytes))

def convert_tree_to_tree_string(config, tree, state=None):
    should_color_nodes = (state is not None)

    stack = [(tree, 1, should_color_nodes)]
    output = ""

    while len(stack) > 0:
        node, depth, in_path = stack.pop()
        output += "-" * depth + " "

        if node.is_leaf():
            if config["action_type"] == "continuous":
                output += str(node.label)
            else:
                output += (config['actions'][node.label]).upper()
        else:
            output += config['attributes'][node.attribute][0]
            output += " <= "
            output += '{:.5f}'.format(node.threshold)

            if node.right:
                stack.append((node.right, depth + 1, should_color_nodes and in_path and state[node.attribute] > node.threshold))
            if node.left:
                stack.append((node.left, depth + 1, should_color_nodes and in_path and state[node.attribute] <= node.threshold))

        if in_path:
            output += " *"
        output += "\n"

    return output
def convert_tree_to_tree_string_with_communities(config, tree, communities=None):
    stack = [(tree, 1)]
    output = ""

    while len(stack) > 0:
        node, depth = stack.pop()
        output += "-" * depth + " "

        if node.is_leaf():
            if config["action_type"] == "continuous":
                output += str(node.label)
            else:
                output += (config['actions'][node.label]).upper()
        else:
            output += config['attributes'][node.attribute][0]
            output += " <= "
            output += '{:.5f}'.format(node.threshold)

            if node.right:
                stack.append((node.right, depth + 1))
            if node.left:
                stack.append((node.left, depth + 1))

        if node.is_leaf() and communities is not None:
            leaf_id = tree.get_node_list(get_leaves=True).index(node)
            group_id = [i for (i, c) in enumerate(communities) if str(leaf_id) in c]
            group_id = group_id[0] if len(group_id) > 0 else 'X'
            output += f" comm{group_id}"
        output += "\n"

    return output

if __name__ == "__main__":
    # decision_tree_string = "- Node1\n-- Node2\n--- Leaf1\n--- Leaf2\n-- Leaf3"
    decision_tree_string = """
        - Test 1
        -- LEFT
        -- Test 2
        --- Test 3
        ---- LEFT
        ---- RIGHT
        --- RIGHT"""

    dot_string = convert_tree_string_to_dot(decision_tree_string)
    image = make_image_from_dot(dot_string)
    image.show()

    print(dot_string)
