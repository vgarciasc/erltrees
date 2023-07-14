from erltrees.evo.evo_tree import Individual
from erltrees.rl.configs import get_config
import gym

config = get_config("dssat")

string = """
- istage <= -0.66667
-- -0.9995175973967536
-- istage <= -0.44444
--- -0.8605915760993957
--- -0.9999794198152064
"""

tree = Individual.read_from_string(config, string)
tree.denormalize_thresholds()

env = gym.make('gym_dssat_pdi:GymDssatPdi-v0', **{'mode': 'fertilization'})
for leaf in tree.get_node_list(get_inners=False, get_leaves=True):
    leaf.label = (leaf.label + 1) * 100
print(tree)

# Take action 0 until episode terminates
state = env.reset()
done = False
dap = 0
while not done:
    dap += 1
    state, reward, done, info = env.step({'anfer': 0})
    print(dap, state['istage'])
env.close()
