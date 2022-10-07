import pdb
from erltrees.rl.configs import get_config
from erltrees.evo.evo_tree import Individual
import erltrees.evo.mutations as mutations

if __name__ == "__main__":
    string = "\n- Car Velocity <= -0.03680\n-- LEFT\n-- Car Position <= -0.15192\n--- RIGHT\n--- Car Velocity <= 0.22881\n---- LEFT\n---- RIGHT"
    config = get_config("mountain_car")
    tree = Individual.read_from_string(config, string)
    print("Before mutation")
    print(tree)

    for _ in range(100):
        mutations.mutate_E(tree, verbose=True)
        print(tree)

    # for _ in range(10):
    #     node = tree.get_random_node(get_inners=False, get_leaves=True)
    #     node.mutate_is_leaf(verbose=True)
    #     print(tree)

    print(tree)
    pdb.set_trace()