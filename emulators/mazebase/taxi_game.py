from mazebase import games
import mazebase.items as maze_items
import copy
from mazebase.utils.mazeutils import choice, MazeException
from mazebase.termcolor import cprint
from mazebase.utils import creationutils
from .taxi_featurizers import  LocalViewFeaturizer, GlobalViewFeaturizer, FewHotEncoder


from .taxi_game_objects import Passenger, TaxiAgent, RestrainedMultiTaskTaxiAgent, rnd
from .taxi_tasks import TaskManager, TaskStatus, LifelongTaskManager

from collections import namedtuple
from enum import Enum, IntEnum
import numpy as np


Relation = Enum('Relation', ['INSIDE', 'NEAR', 'FAR'])
TaxiResetConfig = namedtuple('TaxiResetConfig', ['passenger_taxi', 'passenger_destination', 'taxi_destination'])



class Taxi(games.WithWaterAndBlocksMixin):
    ItemRelation = Relation
    InitState = TaxiResetConfig
    TaskManager = TaskManager

    @staticmethod
    def get_reset_configs():
        configs = [
            # passenger starts inside the taxi
            TaxiResetConfig(passenger_taxi=Relation.INSIDE,
                            passenger_destination=Relation.FAR,
                            taxi_destination=Relation.FAR),
            # passenger and taxi start in the same locations:
            TaxiResetConfig(passenger_taxi=Relation.NEAR,
                            passenger_destination=Relation.FAR,
                            taxi_destination=Relation.FAR),
            # passenger, taxi and destination are assigned to separate locations:
            TaxiResetConfig(passenger_taxi=Relation.FAR,
                            passenger_destination=Relation.FAR,
                            taxi_destination=Relation.FAR),
            # passenger starts inside the taxi, and the taxi already at the destination location
            TaxiResetConfig(passenger_taxi=Relation.INSIDE,
                            passenger_destination=Relation.NEAR,
                            taxi_destination=Relation.NEAR),
            # taxi starts starts at the destination location
            TaxiResetConfig(passenger_taxi=Relation.FAR,
                            passenger_destination=Relation.FAR,
                            taxi_destination=Relation.NEAR),
            ]
        return configs

    def __init__(self,
                 random_seed,
                 max_episode_steps=300,
                 finish_action=False,
                 success_reward=1.1,
                 fail_reward=0.,
                 single_task_episodes=False,
                 tasks=None,
                 **kwargs):
        preliminary_env = kwargs.pop('preliminary_env', False) #no difference there, but this parameter could be useful in subclasses
        self.max_episode_steps = max_episode_steps
        self.reset_configs = self.get_reset_configs()

        finish_action = 'pass' if finish_action else None
        #if not preliminary_env:
        #    print('ENV TASKS:',
        #          tasks if tasks else ['pickup','find_p','convey_p'])

        self.task_manager = self.TaskManager(
            tasks if tasks else ['pickup','find_p','convey_p'],
            extra_task_kwargs={"finish_action":finish_action}
        )
        self.single_task = single_task_episodes #episode equals one task
        self.completion_reward = success_reward
        self.fail_reward = fail_reward
        self.agent_cls = RestrainedMultiTaskTaxiAgent
        self.current_task = None
        self.episode_steps = 0
        self.rnd = np.random.RandomState(random_seed)
        if len(kwargs['map_size']) == 2: #if map is square why specify 4 values?
            kwargs['map_size'] = list(kwargs['map_size'])*2
        self.future_map_size = kwargs['map_size'] # map_size we'll use in the next episode
        #BaseMaseGame.__init__ calls self.reset(), so we need to create all feilds before the call
        #kwargs['waterpct'] = 0.1
        #kwargs['blockpct'] = 0.25
        super(Taxi, self).__init__(**kwargs)
        # Here we directly modify BaseMazeGame.__all_possible_features property:
        features = super(Taxi, self).all_possible_features()
        features.extend(Passenger.all_features())
        features.sort()

    def _choose_reset_config(self, params=None):
        return choice(self.reset_configs)

    def _reset(self):
        self.map_size = self.future_map_size
        super(Taxi, self)._reset()
        #print('=============RESET====================')
        self.current_task = None
        init_state = self._choose_reset_config()

        loc_agent = choice(creationutils.empty_locations(self, bad_blocks=[maze_items.Block]))
        self.agent = self.agent_cls(location=loc_agent)
        self._add_agent(self.agent, "TaxiAgent")

        loc_destination, loc_passenger = self._get_placement_locs(self.agent, 2)


        # check relationship between locations of the passenger and the taxi locations
        if init_state.passenger_taxi == Relation.FAR:
            self.passenger = Passenger(location=loc_passenger)
            self._add_item(self.passenger)
        else:
            self.passenger = Passenger(location=loc_agent)
            self._add_item(self.passenger)
            if init_state.passenger_taxi == Relation.INSIDE:
                self.agent.forced_pickup()
                assert self.passenger.is_pickedup, "Can't put a passenger into a taxi for init_state={}".format(init_state)

        if init_state.taxi_destination == Relation.NEAR:
            self.target = maze_items.Goal(location=loc_agent)
        else:
            self.target = maze_items.Goal(location=loc_destination)
        self._add_item(self.target)

        self.episode_steps = -1 # see the self._step() comment

        self._info = {}
        self._tasks_history = []

    def _get_placement_locs(self, obj_on_map, n_required, remove_limit_per_attempt=2):
        """
        Return list of  n_required random locations that reachable from the location of
        obj_on_map and don't contain walls on them.
        If there aren't enough reachable locations the function tries to remove
        several adjacent blocks to free up more space.

        :param obj_on_map: the object on a map from which we search locations
        :param n_required: number of locations to return
        :param remove_limit_per_attempt: if there is not enough free space, the algorithm
                                       removes (no more than)remove_limit_per_attempt blocks and count free space again!
        """
        num_attempts = 20 #tries to free more space this number of times
        obj_loc = obj_on_map.location
        for _ in range(num_attempts):
            visited, _ = creationutils.dijkstra(self, obj_loc,
                                                creationutils.agent_movefunc)
            empty_locs = set(creationutils.empty_locations(self, bad_blocks=[maze_items.Block, type(obj_on_map)]))
            reachable_locs = list(empty_locs & set(visited))
            n_lack = max(n_required - len(reachable_locs), 0)
            if n_lack == 0:
                break
            num_remove = min(n_lack, remove_limit_per_attempt)
            self._remove_adjacent_blocks(reachable_locs + [obj_loc, ], num_remove)
        else:
            raise MazeException('There is no enough space to place game items')

        placement_locs = rnd.sample(reachable_locs, n_required)
        return placement_locs

    def _remove_adjacent_blocks(self, reachable_locs, num_blocks):
        moves = [(0,-1), (0,1), (-1,0), (1,0)]
        blocks = set()
        for x,y in reachable_locs:
            for move_x, move_y in moves:
                    loc_new= (x+move_x, y+move_y)
                    if self._in_bounds(loc_new):
                        block = self._tile_get_block(loc_new, maze_items.Block)
                        if block is not None:
                            blocks.add(block)
        if num_blocks < len(blocks):
            blocks = rnd.sample(blocks, num_blocks)
        for b in blocks:
            self._remove_item(b.id)

    def task(self):
        return self.current_task

    def _get_reward(self, id):
        reward = super(Taxi, self)._get_reward(id)

        if self._info['task_status'] == TaskStatus.SUCCESS:
            reward += self.completion_reward
        elif self._info['task_status'] == TaskStatus.FAIL :
            reward += self.fail_reward

        return reward

    def _side_information(self):
        return self._info

    def _step(self):
        """
        At each turn _step is performed after the agent actions and before reward computation and new task selection
        """
        self.episode_steps += 1
        if self.episode_steps == 0:
            # first method call in episode is performed right after _reset() and before an agent starts to act.
            # We use this call to create first task for the new episode.
            self.current_task = self.task_manager.next(self, self.rnd)
            self._info['task_status'] = -1 # task_status is similar to the is_done variable.
        else:
            self._info['task_status']=self.current_task.update_status(self)
            if not self.current_task.running():
                self._tasks_history.append(self.current_task.as_info_dict())
                self.soft_respawn(self.current_task)
                self.current_task = self.task_manager.next(self, self.rnd)

        # task_id and task_status are intended to relate to each other
        # like state and is_done variables in a typical environment
        self._info['task_id']=self.current_task.task_id
        self._info['agent_loc'] = self._featurize_coords(self.agent.location)

    def soft_respawn(self, finished_task):
        """
        Override this function to respawn map objects after a task completion.

        Right now, the function respawns the passenger if the passenger has arrived
        to the destination and the current task is completed
        """
        if finished_task.status != TaskStatus.RUNNING:
            if (self.target.location == self.passenger.location) \
            and (self.passenger.is_pickedup is False):
                loc1, loc2 = self._get_placement_locs(self.agent, 2)
                new_loc = loc1 if loc1 != self.target.location else loc2
                self._move_item(self.passenger.id, new_loc)

    def hard_respawn(self):
        """
        Forcefully respawn all movable objects on the map.
        """
        if not self.agent.is_empty():
            self.agent.forced_dropoff()

        map_objs = [self.agent, self.passenger]
        locs = self._get_placement_locs(self.target, len(map_objs))
        for loc, obj in zip(locs, map_objs):
            self._move_item(self.agent.id, loc)

    def _finished(self):
        if self.single_task and len(self._tasks_history):
            return True
        return self.episode_steps >= self.max_episode_steps

    def distance(self, source_loc, target_loc):
        #this is slow! don't use this in training, only for testing
        visited, _ = creationutils.dijkstra(self, source_loc,
                                            creationutils.agent_movefunc)
        if target_loc in visited:
            return visited[target_loc]
        else:
            return float('inf')

    def min_steps_to_complete(self,):
        raise NotImplementedError()
        task = self.task()
        a, p, t = self.agent, self.passenger, self.target
        dist = lambda loc1, loc2: abs(loc1[0]-loc2[0]) + abs(loc1[1]-loc2[1])

        if task == Task.REACH_P:
            return dist(a.location, p.location)
        elif task == Task.REACH_D:
            return dist(a.location, t.location)
        elif task == Task.PICKUP:
            return 0 if p.is_pickedup else dist(a.location, p.location)+1
        elif task == task.DROPOFF:
            return 1 if p.is_pickedup else dist(a.location, p.location)+2
        elif task is None:
            return 0

    def get_tasks_history(self):
        return self._tasks_history

    def set_map_size(self, min_x, max_x, min_y, max_y):
        """
        Set new map size constraints for a next episode
        """
        self.future_map_size = (min_x, max_x, min_y, max_y)
        return self.future_map_size

    def update_map_size(self, *deltas):
        """
        Update map size constraints for a next episode
        Receives delta values for map size constraints
        """
        if len(deltas) != 4:
            raise ValueError('update_map_size expects four deltas for the min_x, max_x, min_y, max_y constraints')
        self.future_map_size = (val+delta for delta, val in zip(deltas, self.future_map_size))
        return self.future_map_size

    def _featurize_coords(self, location):
        """
        convert integer coordinates to float values from [-1,1]
        """
        x,y = location
        x_float = (x/(self.width-1))*2 - 1.
        y_float = (y/(self.height-1))*2 - 1.
        return x_float, y_float


class FixedTaxi(Taxi):

    def __init__(self, *args, **kwargs):
        self._preliminary_env = kwargs.pop('preliminary_env', False)
        self.repeat_episode = kwargs.pop('repeat_episode', False)  # if self.full_reset is True then
        self.fixed_passenger = kwargs.pop('fixed_passenger', True)
        self.initial_game_state = kwargs.pop('initial_state', None)

        if self.initial_game_state is not None:
            self.repeat_episode = True

        super(FixedTaxi, self).__init__(*args, **kwargs)

    def _reset(self):
        if self.repeat_episode:
            self._restore_old_map(self.initial_game_state)
        else:
            self._create_new_map()
            self.initial_game_state = self.collect_state_info()
            if not self._preliminary_env:
                self.display()

    def _create_new_map(self):
        self.map_size = self.future_map_size
        super(Taxi, self)._reset()
        #print('=============RESET====================')
        self.current_task = None
        init_state = self._choose_reset_config()
        if self._preliminary_env:
            user_input = False
        else:
            user_input = self._confirm('Whant to specify object locations?')

        if not user_input:
            loc_agent = choice(creationutils.empty_locations(self, bad_blocks=[maze_items.Block]))
        else:
            loc_agent = self._read_location('Agent')

        self.agent = self.agent_cls(location=loc_agent)
        self._add_agent(self.agent, "TaxiAgent")

        loc_destination, loc_passenger = self._get_placement_locs(self.agent, 2)

        # check relationship between locations of the passenger and the taxi locations
        if init_state.passenger_taxi == Relation.FAR:
            if user_input:
                loc_passenger = self._read_location('Passenger')
            self.passenger = Passenger(location=loc_passenger)
            self._add_item(self.passenger)
        else:
            self.passenger = Passenger(location=loc_agent)
            self._add_item(self.passenger)
            if init_state.passenger_taxi == Relation.INSIDE:
                self.agent.forced_pickup()
                assert self.passenger.is_pickedup, "Can't put a passenger into a taxi for init_state={}".format(init_state)

        if init_state.taxi_destination == Relation.NEAR:
            self.target = maze_items.Goal(location=loc_agent)
        else:
            if user_input:
                loc_destination = self._read_location('Target')
            self.target = maze_items.Goal(location=loc_destination)
        self._add_item(self.target)

        self.episode_steps = -1 # see the self._step() comment

        self._info = {}
        self._tasks_history = []

    def _confirm(self, question):
        for _ in range(10):
            answer = input(question + '[y/n]')
            if answer in ('y', 'n'):
                return answer == 'y'
            else:
                print('Input y or n!')
        else:
            raise ValueError("You just don't want to cooperate!")

    def _read_location(self,name):
        for _ in range(10):
            try:
                x,y = map(int, input("{} location (x y):".format(name)).split())
                if not self._in_bounds((x,y)):
                    raise ValueError('{} is out of bound! Map size is: ({},{})', (x,y), self.width, self.height)
                if self._tile_get_block((x,y), maze_items.Block) is not None:
                    raise ValueError("you can't place an object on the tile with a brick wall!")
                break
            except ValueError as v:
                print(v)
        else:
            raise ValueError("You just don't want to cooperate!")
        return x,y

    def _choose_reset_config(self, params=None):

        if self._preliminary_env:
            return super(FixedTaxi, self)._choose_reset_config()

        print('Current map:')
        self.display(show_coordinates=True)
        print('posible configs:')
        for i, config in enumerate(self.reset_configs):
            print('#{}: {}'.format(i, config))
        init_id = int(input('Please, input the number of the desired config:'))
        chosen_config = self.reset_configs[init_id]

        self.repeat_episode = self._confirm(
            'Do you want to use this map and configuration in every episode?'
        )
        self.fixed_passenger = self._confirm(
            'Do you want to spawn passenger'
            ' in the same location every time?'
        )
        return chosen_config

    def collect_state_info(self):
        state = {}
        state['width'] = self.width
        state['height'] = self.height
        block_locs = []
        water_locs = []
        for x in range(state['height']):
            for y in range(state['height']):
                for item in self._get_items(location=(x,y)):
                    if isinstance(item, maze_items.Block):
                       block_locs.append((x,y))
                    if isinstance(item, maze_items.Water):
                       water_locs.append((x,y))

        state['block_locs'] = block_locs
        state['water_locs'] = water_locs
        state['state_resume'] = dict(
            loc_taxi=self.agent.location,
            loc_passenger=self.passenger.location,
            loc_destination=self.target.location,
            passenger_in_taxi=self.passenger.is_pickedup,
            last_performed_act=self.agent.last_performed_act
        )
        return state

    def _restore_old_map(self, state_info):
        self.current_task = 0
        self.episode_steps = 0

        self.width = state_info['width']
        self.height = state_info['height']
        self._map = [[[] for x in range(self.height)]
                     for y in range(self.width)]
        self._approx_reward_map = [
            [-self.turn_penalty for x in range(self.height)] for y in range(self.width)
        ]

        for loc in state_info['block_locs']:
            self._add_item(maze_items.Block(location=loc))
        for loc in state_info['water_locs']:
            self._add_item(maze_items.Water(location=loc))

        state_resume = state_info['state_resume']
        self.target = maze_items.Goal(location=state_resume['loc_destination'])
        self._add_item(self.target)

        if not self.fixed_passenger:
            old_taxi_loc = state_resume['loc_taxi']
            old_pass_loc = state_resume['loc_passenger']

            loc1, loc2 = self._get_placement_locs(self.target, 2)
            if old_taxi_loc == old_pass_loc:
                #if agent and taxi should be in the same location in the beggining of a new episode
                state_resume['loc_taxi'] = state_resume['loc_passenger'] = loc1
            else: #otherwise taxi stays in the old location:
                state_resume['loc_passenger'] = loc1 if loc1 != old_taxi_loc else loc2

        self.agent = self.agent_cls(location=state_resume['loc_taxi'])
        self._add_agent(self.agent, "TaxiAgent")
        self.passenger = Passenger(location=state_resume['loc_passenger'])
        self._add_item(self.passenger)

        if state_resume['passenger_in_taxi']:
            self.agent.actions['pickup']()
            assert self.passenger.is_pickedup, "Can't put a passenger into a taxi for state={}".format(state_resume)

        self.episode_steps = -1 # see the self._step() comment

        self._info = {}
        self._tasks_history = []

    def soft_respawn(self, finished_task):
        """
        if passenger arrived to the destination and the current task is completed
        then respawns the passenger in a new location
        """
        if self.fixed_passenger:
            if finished_task != TaskStatus.RUNNING:
                if (self.target.location == self.passenger.location) \
                and (self.passenger.is_pickedup is False):
                    loc = self.initial_game_state['state_resume']['loc_passenger']
                    self._move_item(self.passenger.id, loc)
        else:
            super(FixedTaxi, self).soft_respawn(finished_task)

    def display(self, show_coordinates=False):
        ''' Displays the game map for visualization '''
        if show_coordinates: print('   ', end='')

        cprint(' ' * (self.width + 2) * 3, None, 'on_white')
        for y in reversed(range(self.height)):
            if show_coordinates:
                print('{0:>3}'.format(y), end="")
            cprint('   ', None, 'on_white', end="")
            for x in range(self.width):
                itemlst = sorted(filter(lambda x: x.visible, self._map[x][y]),
                                 key=lambda x: x.PRIO)
                disp = [u'   ', None, None, None]
                for item in itemlst:
                    config = item._get_display_symbol()
                    for i, v in list(enumerate(config))[1:]:
                        if v is not None:
                            disp[i] = v
                    s = config[0]
                    if s is None:
                        continue
                    d = list(disp[0])
                    for i, char in enumerate(s):
                        if char != ' ':
                            d[i] = char
                    disp[0] = "".join(d)
                text, color, bg, attrs = disp
                cprint(text, color, bg, attrs, end="")
            cprint('   ', None, 'on_white')

        if show_coordinates: print('   ', end='')

        cprint(' ' * (self.width + 2) * 3, None, 'on_white')
        if show_coordinates:
            print(("   "*2)+\
                   "".join('{0:>3}'.format(i) for i in range(self.width))+\
                   "   ")