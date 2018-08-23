from mazebase import games
import mazebase.items as maze_items
import copy
from mazebase.utils.mazeutils import choice, MazeException
from mazebase.utils import creationutils
from .taxi_featurizers import  LocalViewFeaturizer, GlobalViewFeaturizer, FewHotEncoder

from .taxi_game import Passenger, TaxiAgent, print_obs, user_action, rnd
from .taxi_tasks import TaskManager, TaskStatus

from collections import namedtuple
from enum import Enum, IntEnum
import numpy as np

Relation = Enum('Relation', ['INSIDE', 'NEAR', 'FAR'])
TaxiResetConfig = namedtuple('TaxiResetConfig', ['passenger_taxi', 'passenger_destination', 'taxi_destination'])


class CompactStateResume(object):
    __slots__ = [
        'loc_passenger',
        'loc_taxi',
        'loc_destination',
        'passenger_in_taxi',
        'last_performed_act'
    ]

    def __init__(self,
                 loc_passenger,
                 loc_taxi,
                 loc_destination,
                 passenger_in_taxi,
                 last_performed_act) -> None:

        super(CompactStateResume, self).__init__()
        self.loc_passenger = loc_passenger
        self.loc_taxi = loc_taxi
        self.loc_destination = loc_destination
        self.passenger_in_taxi = passenger_in_taxi
        self.last_performed_act = last_performed_act

class RestrainedMultiTaskTaxiAgent(TaxiAgent):
    """
    Right now my task generating module works incorrectly with the TaxiAgents.
    It doesn't take into account that an agent can end up in a state inappropriate
    for a next task. A quickest patch to this problem is to restrain possible taxi
    actions to prevent the agent from undesirable states during the completion of
    the "reach x" tasks.
    """
    def __init__(self, **kwargs):
        super(RestrainedMultiTaskTaxiAgent, self).__init__(**kwargs)
        self._update_actions()
        self.last_performed_act = None

    def _update_actions(self):
        for a, a_fun in self.actions.items():
            self.actions[a] = self.__only_if_allowed(a,a_fun)

    def __only_if_allowed(self, action_name, action_fun):
        def wrapper():
            task = self.game.task()
            #if no specific task is assigned or task allows that action:
            if (not task) or task.allowed_action(action_name):
                action_fun()
            self.last_performed_act = action_name

        return wrapper



class TaxiMultiTask(games.WithWaterAndBlocksMixin):
    ItemRelation = Relation
    InitState = TaxiResetConfig

    @staticmethod
    def get_reset_configs():
        configs = [
            # passenger stats inside the taxi
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


    def __init__(self, **kwargs):
        self.max_steps = kwargs.get('max_episode_steps', 300)
        self.reset_configs = self.get_reset_configs()

        finish_action = 'pass' if kwargs['finish_action'] else None
        if kwargs['finish_action']:
            print('finish_action flag is provided!')
        self.task_manager = TaskManager(
            ['pickup','find_p','convey_p'],
            extra_task_kwargs={"finish_action":finish_action}
        )

        self.completion_reward = kwargs.get('completion_reward', 1.1)
        self.fail_reward = kwargs.get('fail_reward',0.)
        self.agent_cls = RestrainedMultiTaskTaxiAgent
        self.current_config = None
        self.current_task = None
        self.episode_steps = 0
        em_seed = kwargs.get('random_seed')
        self.rnd = np.random.RandomState(em_seed)

        self.future_map_size = kwargs['map_size'] # map_size we'll use in the next episode
        #BaseMaseGame.__init__ calls self.reset(), so we need to create all feilds before the call
        super(TaxiMultiTask, self).__init__(**kwargs)
        # Here we directly modify BaseMazeGame.__all_possible_features property:
        features = super(TaxiMultiTask, self).all_possible_features()
        features.extend(Passenger.all_features())
        features.sort()

    def _get_reset_config(self, params=None):
        return choice(self.reset_configs)

    def _reset(self):
        self.map_size = self.future_map_size
        super(TaxiMultiTask, self)._reset()
        #print('=============RESET====================')
        self.current_task = None
        loc_agent = choice(creationutils.empty_locations(self, bad_blocks=[maze_items.Block]))
        self.agent = self.agent_cls(location=loc_agent)
        self._add_agent(self.agent, "TaxiAgent")

        loc_destination, loc_passenger = self._get_placement_locs(loc_agent, 2)
        init_state = self._get_reset_config()

        # check relationship between locations of the passenger and the taxi locations
        if init_state.passenger_taxi == Relation.FAR:
            self.passenger = Passenger(location=loc_passenger)
            self._add_item(self.passenger)
        else:
            self.passenger = Passenger(location=loc_agent)
            self._add_item(self.passenger)
            if init_state.passenger_taxi == Relation.INSIDE:
                self.agent.actions['pickup']()
                assert self.passenger.is_pickedup, "Can't put a passenger into a taxi for init_state={}".format(init_state)

        if init_state.taxi_destination == Relation.NEAR:
            self.target = maze_items.Goal(location=loc_agent)
        else:
            self.target = maze_items.Goal(location=loc_destination)
        self._add_item(self.target)

        self.episode_steps = -1 # see the self._step() comment

        self._info = {}

    def _get_placement_locs(self, agent_loc, n_required):
        for _ in range(10):
            visited, _ = creationutils.dijkstra(self, agent_loc,
                                                creationutils.agent_movefunc)
            empty_locs = set(creationutils.empty_locations(self, bad_blocks=[maze_items.Block, TaxiAgent]))
            reachable_locs = list(empty_locs & set(visited))
            n_lack = max(n_required - len(reachable_locs), 0)
            if n_lack == 0:
                break
            self._remove_adjacent_blocks(reachable_locs + [agent_loc, ], n_lack)
        else:
            raise MazeException('There is no enough space to place game items')

        placement_locs = rnd.sample(reachable_locs, n_required)
        return placement_locs

    def _remove_adjacent_blocks(self, reachable_locs, num_blocks):
        moves = [(0,-1), (0,1), (-1,0), (1,0)]
        blocks = []
        for x,y in reachable_locs:
            for move_x, move_y in moves:
                    loc_new= (x+move_x, y+move_y)
                    if self._in_bounds(loc_new):
                        block = self._tile_get_block(loc_new, maze_items.Block)
                        if block is not None:
                            blocks.append(block)

        blocks = rnd.sample(blocks, num_blocks)
        for b in blocks:
            self._remove_item(b.id)

    def task(self):
        return self.current_task

    def _get_reward(self, id):
        reward = super(TaxiMultiTask, self)._get_reward(id)

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
        here we update self.state_resume and update current task's status with it.
        """
        self.episode_steps += 1
        if self.episode_steps == 0:
            # first method call in episode is performed right after _reset() and before an agent starts to act.
            # We use this call to create first task for the new episode.
            self.current_task = self.task_manager.next(self.state_resume(), self.rnd)
            self._info['task_status'] = -1 # task_status is similar to the is_done variable.
        else:
            self._info['task_status']=self.current_task.update_status(self.state_resume())
            if not self.current_task.running():
                self._respawn_passenger_if_needed(self._info['task_status'])
                self.current_task = self.task_manager.next(self.state_resume(), self.rnd)

        # task_id and task_status are intended to relate to each other
        # like state and is_done variables in a typical environment
        self._info['task_id']=self.current_task.task_id

    def _respawn_passenger_if_needed(self, last_task_status):
        """
        if passenger arrived to the destination and the current task is completed
        then respawns the passenger in a new location
        """
        if last_task_status != TaskStatus.RUNNING:
            if (self.target.location == self.passenger.location) \
            and (self.passenger.is_pickedup is False):
                loc1, loc2 = self._get_placement_locs(self.agent.location, 2)
                new_loc = loc1 if loc1 != self.target.location else loc2
                self._move_item(self.passenger.id, new_loc)

    def state_resume(self):
        """
        Returns enough information to select new task or detect task completion
        """
        return CompactStateResume(
            loc_taxi=self.agent.location,
            loc_passenger=self.passenger.location,
            loc_destination=self.target.location,
            passenger_in_taxi=self.passenger.is_pickedup,
            last_performed_act=self.agent.last_performed_act
        )

    def _finished(self):
        return self.episode_steps >= self.max_steps

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


class FixedTaxiMultiTask(TaxiMultiTask):

    def __init__(self, **kwargs):
        self.repeat_episode = False  # if self.full_reset is True then
        self.initial_game_state = None
        super(FixedTaxiMultiTask, self).__init__(**kwargs)

    def _reset(self):
        if self.repeat_episode is True:
            self.restore(self.initial_game_state)
        else:
            super(FixedTaxiMultiTask, self)._reset()
            self.initial_game_state = self.collect_state_info(self)

    def collect_state_info(self, taxi_game):
        state = {}
        state['width'] = taxi_game.width
        state['height'] = taxi_game.height
        block_locs = []
        water_locs = []
        for x in range(state['height']):
            for y in range(state['height']):
                for item in taxi_game._get_items(location=(x,y)):
                    if isinstance(item, maze_items.Block):
                       block_locs.append((x,y))
                    if isinstance(item, maze_items.Water):
                       water_locs.append((x,y))

        state['block_locs'] = block_locs
        state['water_locs'] = water_locs
        state['config'] = taxi_game.current_config
        return state

    def restore(self, state_info):
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

        init_state, tasks = state_info['config']
        self.agent = self.agent_cls(location=init_state['loc_a'])
        self._add_agent(self.agent, "TaxiAgent")
        self.target = maze_items.Goal(location=init_state['loc_t'])
        self._add_item(self.target)
        self.passenger = Passenger(location=init_state['loc_p'])
        self._add_item(self.passenger)
        if init_state['is_picked_up']:
            self.agent.actions['pickup']()
        self.current_config = state_info['config']
