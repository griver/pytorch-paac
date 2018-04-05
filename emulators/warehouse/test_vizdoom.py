from argparse import Namespace

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

from emulators.warehouse import warehouse_tasks as wh_tasks
from emulators.warehouse.warehouse_emulator import WarehouseEmulator, create_json_config

args = Namespace(
    num_emulators=1,
    resource_folder='resources/vizdoom_scenarios/',
    game='warehouse',
    history_window=1,
    random_seed=3,
)


def transform_obs(obs):
    if obs.shape[0] == 1:
        return obs[-1,:,:]
    return obs.transpose(1,2, 0)# from (C,H,W) to (H,W,C) format


def task_manager():
    return wh_tasks.TaskManager(
        [wh_tasks.PickUp, wh_tasks.Drop, wh_tasks.Visit, wh_tasks.CarryItem]
    )

create_json_config()
print('Arguments:', args)

emulators = [WarehouseEmulator(i, args, task_manager()) for i in range(args.num_emulators)]
#emulators = [ve.VizdoomEmulator(i, args) for i in range(args.num_emulators)]

buttons = np.array(emulators[0].game.get_available_buttons())
legal_actions = emulators[0].get_legal_actions()
num_actions = len(legal_actions)
one_hot_acts = np.eye(num_actions,dtype=np.int32)


#states is an array of shape (channels, height, width)
obs, tasks  = [None]*args.num_emulators, [None]*args.num_emulators
for i, e in enumerate(emulators):
    obs[i], tasks[i] = e.get_initial_state()

is_done = np.array([False] * len(emulators))
rewards = np.zeros(len(emulators))
num_steps = np.zeros(len(emulators))
step = 0

fig = plt.figure()
images = [None]*args.num_emulators

for i in range(args.num_emulators):
    ax = fig.add_subplot(1, args.num_emulators, i+1)
    img = transform_obs(obs[i])
    images[i] = ax.imshow(img,  animated=True)
    ax.set_title('#{0}'.format(i + 1))
    plt.axis('off')
#def_acts =  [[1],] + [[0],]*10

def update_images(val):
    global step
    step += 1
    print('=========== Step #%d =============' % step)
    acts = np.random.choice(num_actions, args.num_emulators, replace=True)
    #acts = def_acts[step-1]
    for i, em in enumerate(emulators):
        if not is_done[i]:
            obs[i], r_i, is_done[i], tasks[i] = em.next(one_hot_acts[acts[i]])
            print('Em #{} R={}, is_done={}'.format(i, r_i, is_done[i]))
            rewards[i] += r_i
            num_steps[i] = step
            if not is_done[i]:
                images[i].set_array(transform_obs(obs[i]))

    print('Number steps: ', num_steps)
    print('Total Rewards:', rewards)
    if is_done.all():
        print('Episodes have finished!')
        ani.event_source.stop()



ani = animation.FuncAnimation(fig, update_images, interval=200, frames=100, repeat=False)

plt.show()


