import emulators.vizdoom.vizdoom_emulator as ve
import time
import itertools
import numpy as np
from argparse import Namespace
import matplotlib.pyplot as plt
from matplotlib import animation

args = Namespace(
    num_emulators=2,
    resource_folder='resources/vizdoom_scenarios/',
    game_name='simpler_basic',
    history_window=1
)

print('Arguments:', args)
emulators = [ve.VizdoomEmulator(i, args) for i in range(args.num_emulators)]
buttons = np.array(emulators[0].game.get_available_buttons())
legal_actions = emulators[0].get_legal_actions()
num_actions = len(legal_actions)
one_hot_acts = np.eye(num_actions,dtype=np.int32)
print('buttons:', buttons)
print(legal_actions)
#states is an array of shape (history_window, height, width)
states = [e.get_initial_state()[-1,:,:] for e in emulators]
is_done = np.array([False] * len(emulators))
rewards = np.zeros(len(emulators))
num_steps = np.zeros(len(emulators))
step = 0

fig = plt.figure()
images = [None]*args.num_emulators
data = np.random.rand(128, 128)

for i in range(args.num_emulators):
    ax = fig.add_subplot(1, args.num_emulators, i+1)
    images[i] = ax.imshow(states[i], animated=True)
    ax.set_title('#{0}'.format(i + 1))
    plt.axis('off')
#def_acts =  [[1],] + [[0],]*10

def update_states(val):
    global step
    step += 1
    print('=========== Step #%d =============' % step)
    acts = np.random.choice(num_actions, args.num_emulators, replace=True)
    #acts = def_acts[step-1]
    for i, em in enumerate(emulators):
        if not is_done[i]:
            states[i], r_i, is_done[i] = em.next(one_hot_acts[acts[i]])
            print('Em #{} R={}, is_done={}'.format(i, r_i, is_done[i]))
            rewards[i] += r_i
            num_steps[i] = step
            if not is_done[i]:
                images[i].set_array(states[i][-1,:,:])

    print('Number steps: ', num_steps)
    print('Total Rewards:', rewards)
    if is_done.all():
        print('Episodes have finished!')
        ani.event_source.stop()



ani = animation.FuncAnimation(fig, update_states, interval=200, frames=50, repeat=False)

plt.show()


