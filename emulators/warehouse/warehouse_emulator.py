import copy
from collections import namedtuple

import numpy as np
from vizdoom import Button
from copy import deepcopy
from ..vizdoom import vizdoom_emulator as ve
from .warehouse_map_info import load_map_info
from .warehouse_tasks import DummyManager
import emulators.warehouse.warehouse_tasks as wh_tasks

ComplexityParams = namedtuple(
    "ComplexityParams",
    [
        "n_rooms", #a number of target rooms not counting the entry room.
        "items_limit", #maximum number of items that can be spawned
        "room_dist", #1 or 2. How far a target room can be from the entry room
    ])

def task2str(env, task):
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


class WarehouseEmulator(ve.VizdoomEmulator):

    class StateInfo(object):
        __slots__ = ['obs', 'room_id', 'item_id', 'rooms',
         'entry_room', 'agent_pos', 'item_count']

        def __init__(self, obs, room_id, item_id, rooms,
                     entry_room, agent_pos, item_count):
            self.obs=obs
            self.room_id=room_id
            self.item_id=item_id
            self.rooms = rooms
            self.entry_room = entry_room
            self.agent_pos = agent_pos
            self.item_count = item_count

    SKILL_PARAMS = (
        ComplexityParams(n_rooms=(1, 1), room_dist=1, items_limit=4),#skill=0
        ComplexityParams(n_rooms=(1, 2), room_dist=1, items_limit=6),#1
        ComplexityParams(n_rooms=(2, 2), room_dist=2, items_limit=8),#2
        ComplexityParams(n_rooms=(2, 3), room_dist=2, items_limit=9),#3
        ComplexityParams(n_rooms=(3, 4), room_dist=2, items_limit=11),#4
        ComplexityParams(n_rooms=(4, 5), room_dist=2, items_limit=13),#5
    )

    def __init__(self, emulator_id, game, resource_folder, task_manager=None, random_seed=3,
                 skill_level=1,  **kwargs):
        kwargs.setdefault('reward_coef', 1.)
        super(WarehouseEmulator, self).__init__(emulator_id, game,
                                                resource_folder, **kwargs)

        info_file_path = ve.join_path(resource_folder, game)
        self._map_info = load_map_info(info_file_path)
        random_seed = (emulator_id + 1) * random_seed
        self.rnd = np.random.RandomState(random_seed)
        self.skill = skill_level
        self.task_manager = deepcopy(task_manager) if task_manager else DummyManager()
        self.__check_task_manager()
        self.task = None
        self.emulator_id = emulator_id

    def __check_task_manager(self):
        for var in self.task_manager.required_state_info():
            if var not in self.StateInfo.__slots__:
                raise ValueError(
                    "task_manager requires information" + \
                    "that {} doesn't provide".format(type(self))
                )

    def _define_actions(self, initalized_game):
        """
        Defines a list of legal agent actions.
        An action doesn't have to meet an individual button press.
        :return: (list of legal actions, noop action)
        """
        buttons = initalized_game.get_available_buttons()
        num_buttons = len(buttons)
        # each agent action corresponds to one button press:
        legal_actions = np.eye(num_buttons, dtype=int).tolist()
        #exept two cases:
        #1. Agen can pickup an item only if it has non-zero speed
        move_forward_id = buttons.index(Button.MOVE_FORWARD)
        pickup_id = buttons.index(Button.USE)
        legal_actions[pickup_id][move_forward_id] = 1 # adds forward movement when use is pressed
        #2. the pass action:
        noop = [0]*num_buttons
        legal_actions.append(noop)

        return legal_actions, noop

    def reset(self):
        #print('====== Reset emulator #{} ====='.format(self.emulator_id))
        self._init_episode()
        doom_state = self.game.get_state()
        self._update_state_info(doom_state)
        self.task = self.task_manager.next(self._state_info, self.rnd)

        info = {'task_status':self.task.status.value, 'n_steps':self.task.n_steps}
        info.update(self.task.as_info_dict())

        return self._state_info.obs, info

    def _init_episode(self):
        self.game.new_episode()
        self._reset_episode_vars()
        params = self.SKILL_PARAMS[self.skill]
        rooms = self._select_rooms(params)
        self._state_info.rooms = rooms
        self._update_room_textures(rooms)
        player_spawn_spots = set()
        doors_to_open = set()

        for r_id, room in rooms.items():
            #doors blocking path between the room and entry_room:
            doors_to_open.update(room.doors)
            #possible spawn spots that was blocked by respective doors:
            player_spawn_spots.update(room.spawn_spots)

        for door in doors_to_open:
            self._execute('open_door', door)

        self._spawn_item_if_needed()
        #spawn_player:
        #print('Rooms:', list(rooms.keys()), 'spots:', list(player_spawn_spots))
        spawn_spot = self.rnd.choice(list(player_spawn_spots))
        #print('spawn spot:', spawn_spot)
        self._execute('spawn_player', spawn_spot)
        self.game.make_action(self.noop)#we need a tick for commands("spawn_player", .etc) to finish their work

    def _reset_episode_vars(self):
        self._state_info = self.StateInfo(
            obs=None,
            room_id=None,
            item_id=None,
            rooms=None,
            entry_room=self._map_info['entry_room'].id,
            agent_pos=None,
            item_count=None
        )
        self.task = None
        self._completed = [] #list that holds all tasks completed in the current episode
        self.__last_command_tick = 0 # see self._execute
        self._num_spawned = 0 # basicaly the total number of items in the map
        self._num_items = 0  # a number of items in the rooms(not in the hallways)
        self._can_spawn = True # spawn new item at the beginning of every episode

    def _select_rooms(self, skill_params):
        rooms = []
        entry_id = self._map_info['entry_room'].id
        for r_id,r in self._map_info['rooms'].items():
            if r.entry_dist <= skill_params.room_dist and r_id != entry_id:
                rooms.append(copy.copy(r))
                #map_info stores data about the map at the beginning of a new episode
                #therefore we don't want to compromize the data with episode specific
                #changes
        min_num, max_num = skill_params.n_rooms
        n_rooms = self.rnd.randint(min_num, max_num+1)
        #chouse n_rooms excluding the entry room
        rooms = self.rnd.choice(rooms, n_rooms, replace=False).tolist()
        rooms.append(self._map_info['entry_room'])
        return {r.id:r for r in rooms}

    def _update_room_textures(self, selected_rooms):
        entry_room_id = self._map_info['entry_room'].id
        entry_texture_id = self._map_info['default_texture_id']
        textures = [(t_id,name) for t_id, name in self._map_info['textures']
                    if t_id != entry_texture_id]
        textures = textures[:6]
        self.rnd.shuffle(textures)
        for r_id, r in selected_rooms.items():
            if r_id == entry_room_id: continue
            t_id, t_name = textures.pop()
            r.texture = t_id
            self._execute('set_room_texture', r.id, t_id)

    def _update_state_info(self, doom_state):
        doom_vars = doom_state.game_variables
        aX, aY, item_id, room_id, = doom_vars[:4]
        self._num_items, self._can_spawn = doom_vars[4:6]
        obs = self._preprocess(doom_state.screen_buffer, self.screen_size)
        self._state_info.obs = obs
        self._state_info.room_id = room_id
        self._state_info.item_id = item_id
        self._state_info.agent_pos = (aX,aY)
        self._state_info.item_count = doom_vars[6:]

    def _spawn_item_if_needed(self):
        if self._can_spawn:
            items_lim = self.SKILL_PARAMS[self.skill].items_limit
            if (self._num_items > items_lim) or (self._num_spawned > items_lim * 1.5):
                return
            item_id = self.rnd.choice(list(self._map_info['items'].keys()))
            self._execute('spawn_item', item_id)
            #print('Spawn', self._map_info['items'][item_id])
            self._num_spawned += 1

    def _execute(self, script_name, arg1=0,arg2=0, arg3=0):
        """
        Executes a script defined in the iwad file
        Scripts are written on ACS (https://en.wikipedia.org/wiki/Action_Code_Script),
        receive up to 3 integer arguments, and can't return values.
        """
        curr_tick = self.game.get_episode_time()
        # For some reason if you try to execute several commands in one tick
        # some of them might not work
        if self.__last_command_tick >= curr_tick:
            self.game.make_action(self.noop)
        self.__last_command_tick = self.game.get_episode_time()

        self.game.send_game_command(
          "pukename {} {} {} {}".format(script_name, arg1, arg2, arg3))


    def next(self, action):
        """
        Performs the given action.
        Returns the next state, reward, and terminal signal
        """
        self._spawn_item_if_needed()
        action = self.legal_actions[np.argmax(action)]
        reward = self.game.make_action(action, self.action_repeat)
        is_done = self.game.is_episode_finished()
        if is_done: #task_id 0 is a NoTask. Zero value for property has the same meaning.
            return self.terminal_obs, reward, is_done, {'task_status':-1, 'task_id':0, 'property':0, 'n_steps':0}
            #completed = [str(t) for t in self._completed] + [str(self.task)]
            #print('Emulator #{} completed_tasks: {}'.format(self._id, completed), flush=True)

        self._update_state_info(self.game.get_state())
        if self.task.finished():
            self._completed.append(self.task)
            self.task = self.task_manager.next(self._state_info, self.rnd)

        reward, _ = self.task.update(reward, is_done, self._state_info)
        reward = reward * self.reward_coef
        info = {'task_status': self.task.status.value, 'n_steps':self.task.n_steps}
        info.update(self.task.as_info_dict())
        #print('emulator#{} r={}, task={}, info={}'.format(self._id, reward, self.task, info))
        return self._state_info.obs, reward, is_done, info

    def watch_next(self):
        reward = 0.
        if not self.game.is_episode_finished():
            self._spawn_item_if_needed()
            self.game.advance_action(2)
            reward = self.game.get_last_reward()

        is_done = self.game.is_episode_finished()
        if is_done:
            next_obs = self.terminal_obs
            return next_obs, reward, is_done, {}

        doom_state = self.game.get_state()
        self._update_state_info(doom_state)
        reward, _ = self.task.update(reward, is_done, self._state_info)
        reward = reward * self.reward_coef
        info = {'task_status':self.task.status.value}
        info.update(self.task.as_info_dict())

        if self.task.finished():
            self._completed.append(self.task)
            self.task = self.task_manager.next(self._state_info, self.rnd)

        return self._state_info.obs, reward, is_done, info

