#!/usr/bin/env python

#####################################################################
# This script presents labels buffer that shows only visible game objects
# (enemies, pickups, exploding barrels etc.), each with unique label.
# OpenCV is used here to display images, install it or remove any
# references to cv2
# Configuration is loaded from "../../scenarios/basic.cfg" file.
# <episodes> number of episodes are played.
# Random combination of buttons is chosen for every action.
# Game variables from state and last reward are printed.
#
# To see the scenario description go to "../../scenarios/README.md"
#####################################################################

from argparse import Namespace

import cv2
import numpy as np
from vizdoom import *

from emulators.warehouse import warehouse_tasks as wh_tasks
from emulators.warehouse.warehouse_emulator import VizdoomWarehouse

MODE = Mode.SPECTATOR #Mode.PLAYER

rnd_seed = 467#np.random.randint(500)
print('Random seed is', rnd_seed)


def create_actions(env):
    num_actions = len(env.get_legal_actions())
    actions = np.eye(num_actions)
    # [Button.MOVE_FORWARD, Button.MOVE_LEFT, Button.MOVE_RIGHT, Button.TURN_LEFT, Button.TURN_RIGHT, Button.USE, Button.DROP_SELECTED_ITEM]
    #buttons = env.game.get_available_buttons()
    return actions


def make_action(env, actions):
    if MODE is Mode.SPECTATOR:
        return env.watch_next()

    act = np.random.choice(actions)
    return env.next(act)


def task2str(task):
    info = env._map_info
    if isinstance(task, wh_tasks.Visit):
        if task.target_id == info['entry_room'].id:
            return '[{}] Visit Entry room!'.format(task.status._name_)
        return '[{}] Visit room with {}'.format(
            task.status._name_, info['textures'][task.property][1])
    elif isinstance(task, wh_tasks.CarryItem):
        if task.target_id == info['entry_room'].id:
            return '[{}] Bring the item in the entry room!'.format(task.status._name_)
        return '[{}] Bring the item in the room with {}'.format(
            task.status._name_, info['textures'][task.property][1])
    elif isinstance(task, wh_tasks.Drop):
        return "[{}] Drop the carried item!".format(task.status._name_)
    elif isinstance(task, wh_tasks.PickUp):
        return "[{}] Pick up a {}".format(
            task.status._name_, info['items'][task.target_id])


def dist(pt1, pt2):
    return np.linalg.norm(np.array(pt1) - np.array(pt2))


def task_manager():
    return wh_tasks.TaskManager(
        [wh_tasks.PickUp, wh_tasks.Drop, wh_tasks.Visit, wh_tasks.CarryItem],
        priorities=[2., 1.5, 1., 1.]
    )


kwargs = dict(
    resource_folder='resources/vizdoom_scenarios/',
    game='warehouse',
    skill_level = 2,
    visualize = True,
    history_window=1,
    random_seed=rnd_seed,
    task_manager=task_manager()
)

#create_json_config()

VizdoomWarehouse.MODE = MODE
VizdoomWarehouse.SCREEN_RESOLUTION = ScreenResolution.RES_640X480
env = VizdoomWarehouse(0, **kwargs)
actions = create_actions(env)
#env.game.set_labels_buffer_enabled(True)


episodes = 10

# Sleep time between actions in ms
sleep_time = 1000

for i in range(episodes):
    print("Episode #" + str(i + 1))

    # Not needed for the first episode but the loop is nicer.
    obs, info = env.reset()
    is_done = False
    prev_task = None
    while not is_done:
        # Gets the state and possibly to something with it
        cv2.waitKey(sleep_time)
        obs, r, is_done, info = make_action(env, actions)
        st_info = env._state_info
        room_id = st_info.room_id
        items = st_info.item_count
        task = env.task

        if prev_task and prev_task != task:
            print('\r#{}: {}, last_reward={}'.format(
                len(env._completed), task2str(prev_task),r))

        print('\rtask={} room_id={}, num_items={} info={}'.format(task2str(task),room_id, items, info),
            end='', flush=True)
        prev_task = task


    print("/nEpisode finished!")
    print("************************")

cv2.destroyAllWindows()


def old_cycle(state, pX,pY):
    if len(state.labels) == 0:
        print('\rplayer_pos:({0:.1f},{1:.1f}), object=None', end='', flush=True)
    else:
        print('\rplayer_pos:({0:.1f},{1:.1f})', end=',')
        for l in state.labels:
            objX, objY = l.object_position_x, l.object_position_y
            dist_to_obj = dist((pX, pY), (objX, objY))
            print(" obj={0}, dist=({1:.2f})".format(l.object_name, dist_to_obj), end=',')
        print(end='', flush=True)