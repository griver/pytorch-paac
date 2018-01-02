from mazebase import games
import mazebase.items as maze_items
import copy
from mazebase.utils.mazeutils import choice, MazeException
from mazebase.utils import creationutils
from .taxi_featurizers import  LocalViewFeaturizer, GlobalViewFeaturizer, FewHotEncoder

from .taxi_game import Passenger, TaxiAgent, print_obs, user_action, rnd

from collections import namedtuple
from enum import Enum

Relation = Enum('Relation', ['INSIDE', 'NEAR', 'FAR'])
Task = Enum('Task', "PICKUP DROPOFF REACH_P REACH_D")
TaxiGameState = namedtuple('TaxiGameState', ['pRt', 'pRd', 'tRd'])


class RestrainedMultiTaskTaxiAgent(TaxiAgent):
    """
    Right now my the generating module works incorrectly with the TaxiAgents.
    It doesn't take into account that an agent can end up in a state inappropriate
    for a next task. A quickest patch to this problem is to restrain possible taxi
    actions to prevent the agent from undesirable states during the completion of
    the "reach x" tasks.
    """
    PICKUP_RESTRICTED_TASKS = [Task.REACH_P, Task.REACH_D]
    DROPOFF_RESTRICTED_TASKS = [Task.REACH_P, Task.REACH_D]

    def __init__(self, **kwargs):
        super(RestrainedMultiTaskTaxiAgent, self).__init__(**kwargs)
        self._update_actions()

    def _update_actions(self):
        self.actions['pickup'] = self._restrained_pickup
        self.actions['dropoff'] = self._restrained_dropoff

    def _restrained_pickup(self):
        if self.game.task() in self.PICKUP_RESTRICTED_TASKS:
            return
        x, y = self.location
        passenger = self.game._tile_get_block((x, y), Passenger)
        if passenger is not None and self.is_empty():
            passenger.pickup()
            self.passenger = passenger

    def _restrained_dropoff(self):
        if self.game.task() in self.DROPOFF_RESTRICTED_TASKS:
            return
        if not self.is_empty():
            assert self.passenger.location == self.location, \
            'Transported passenger must have the same location as the taxi!'
            self.passenger.dropoff()
            self.passenger = None


class TaxiGameNode(object):
    def __init__(self, game_state, is_start=False, is_end=False):
        self.state = game_state
        self.is_start = is_start
        self.is_end = is_end
        self._transitions = {}

    def add_transition(self, task, dst_node):
        self._transitions[task] = dst_node

    def add_transitions(self, *transition_pairs):
        for task, dst_node in transition_pairs:
            self.add_transition(task, dst_node)

    @property
    def transitions(self):
        return sorted(self._transitions.items(), key=lambda item: item[0].value)

    def __str__(self):
        return str('Node({0[0].name}|{0[1].name}|{0[2].name})'.format(self.state))


class MultiTaskEpisodeConfig(namedtuple('AbsEpisodeConfig', ['init_state', 'tasks'])):
    def __repr__(self, ):
        return 'TextEpisodeConfig(init_state={0},tasks={1})'.format(repr(self.init_state), repr(self.tasks))

    def __str__(self, ):
        init_state = 'INIT({0.name}|{1.name}|{2.name})'.format(*self.init_state)
        tasks = 'TASKS:' + str([t.name for t in self.tasks])
        return ' '.join([init_state, tasks])


def create_taxi_graph():
    nodes = []
    in_f_f = TaxiGameNode(TaxiGameState(pRt=Relation.INSIDE, pRd=Relation.FAR, tRd=Relation.FAR), is_start=True)
    nodes.append(in_f_f)
    n_f_f = TaxiGameNode(TaxiGameState(pRt=Relation.NEAR, pRd=Relation.FAR, tRd=Relation.FAR), is_start=True)
    nodes.append(n_f_f)
    f_f_f = TaxiGameNode(TaxiGameState(pRt=Relation.FAR, pRd=Relation.FAR, tRd=Relation.FAR), is_start=True)
    nodes.append(f_f_f)
    f_f_n = TaxiGameNode(TaxiGameState(pRt=Relation.FAR, pRd=Relation.FAR, tRd=Relation.NEAR))
    nodes.append(f_f_n)
    in_n_n = TaxiGameNode(TaxiGameState(pRt=Relation.INSIDE, pRd=Relation.NEAR, tRd=Relation.NEAR))
    nodes.append(in_n_n)
    n_n_n = TaxiGameNode(TaxiGameState(pRt=Relation.NEAR, pRd=Relation.NEAR, tRd=Relation.NEAR), is_end=True)
    nodes.append(n_n_n)

    in_f_f.add_transitions(
      (Task.DROPOFF, n_f_f),
      (Task.REACH_D, in_n_n)
    )
    n_f_f.add_transitions(
      (Task.PICKUP, in_f_f),
      (Task.REACH_D, f_f_n)
    )
    f_f_f.add_transitions(
      (Task.REACH_P, n_f_f),
      (Task.REACH_D, f_f_n)
    )
    f_f_n.add_transition(
      Task.REACH_P, n_f_f
    )
    in_n_n.add_transition(
      Task.DROPOFF, n_n_n
    )
    return nodes


def generate_task_seqs(graph, min_length, max_length):
    def dfs(node, path=()):
        if len(path) <= max_length:
            if min_length <= len(path):
                seqs.append(path)
            for task, dst in node.transitions:
                dfs(dst, path + (task,))

    start_nodes = [node for node in graph if node.is_start]
    start_state2seqs = {}
    for node in start_nodes:
        seqs = []
        dfs(node)
        start_state2seqs[node.state] = seqs

    return start_state2seqs


class TaxiMultiTask(games.WithWaterAndBlocksMixin):
    ItemRelation = Relation
    InitState = TaxiGameState
    Tasks = Task

    @staticmethod
    def create_episode_configs(min_tasks_number=2, max_tasks_number=3):
        graph = create_taxi_graph()
        start_states2seqs = generate_task_seqs(graph, min_tasks_number, max_tasks_number)
        episode_configs = []
        for state, seqs in start_states2seqs.items():
            for seq in seqs:
                episode_configs.append(MultiTaskEpisodeConfig(state, seq))
        return episode_configs

    def __init__(self, **kwargs):
        self.max_steps = kwargs.get('max_episode_steps', 300)
        self.episode_configs = kwargs.get('episode_configs',TaxiMultiTask.create_episode_configs())
        self.task_reward = kwargs.get('task_reward', 1.)
        self.agent_cls = RestrainedMultiTaskTaxiAgent
        self.current_config = None
        self.current_task = None
        self.episode_steps = 0
        #BaseMaseGame.__init__ calls self.reset(), so we need to create all feilds before the call
        super(TaxiMultiTask, self).__init__(**kwargs)
        # Here we directly modify BaseMazeGame.__all_possible_features property:
        features = super(TaxiMultiTask, self).all_possible_features()
        features.extend(Passenger.all_features())
        features.sort()

    def _get_new_config(self, params=None):
        return choice(self.episode_configs)

    def _reset(self):
        super(TaxiMultiTask, self)._reset()
        #print('=============RESET====================')
        self.current_task = None
        n_items = 2 #passenger, target
        loc_agent = choice(creationutils.empty_locations(self, bad_blocks=[maze_items.Block]))
        self.agent = self.agent_cls(location=loc_agent)
        self._add_agent(self.agent, "TaxiAgent")

        placement_locs = self._get_placement_locs(loc_agent, n_items)
        loc_target, loc_pass = rnd.sample(placement_locs, n_items)
        init_state, tasks = self._get_new_config()

        # check relationship between locations of the passenger and the taxi locations
        if init_state.pRt == Relation.FAR:
            self.passenger = Passenger(location=loc_pass)
            self._add_item(self.passenger)
        else:
            self.passenger = Passenger(location=loc_agent)
            self._add_item(self.passenger)
            if init_state.pRt == Relation.INSIDE:
                self.agent.actions['pickup']()
                assert self.passenger.is_pickedup, "Can't put a passenger into a taxi for init_state={}".format(init_state)

        self.target = maze_items.Goal(location=loc_target)
        self._add_item(self.target)
        self.current_task = 0
        self.episode_steps = 0
        detailed_state = dict(
            loc_a=self.agent.location,
            loc_p=self.passenger.location,
            loc_t=self.target.location,
            is_picked_up=self.passenger.is_pickedup
        )
        self.current_config = MultiTaskEpisodeConfig(init_state=detailed_state, tasks=tasks)

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
        return reachable_locs

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
        if self.current_task is None:
            return None
        return self.current_config.tasks[self.current_task]

    def _task_finished(self):
        task = self.task()
        a, p, t = self.agent, self.passenger, self.target

        if task == Task.REACH_P:
            assert (p.is_pickedup is False), "Can't assign the \"Reach Passenger\" task when the passenger is in taxi already!"
            return p.location == a.location
        elif task == Task.REACH_D:
            return a.location == t.location
        elif task == Task.PICKUP:
            return p.is_pickedup
        elif task == Task.DROPOFF:
            return (p.is_pickedup is False)
        else:
            print('Current task equals to {0}'.format(task))
            return False

    def _get_reward(self, id):
        reward = super(TaxiMultiTask, self)._get_reward(id)
        if self._task_finished():
            self.current_task += 1
            if self.current_task >= len(self.current_config.tasks):
                self.current_task = None
            return self.task_reward
        return reward

    def _side_information(self):
        task = self.task()
        if task is None:
            return [['None', 0]]
        return [[task.name, task.value]]

    def _step(self):
        self.episode_steps += 1

    def _finished(self):
        return (self.current_task == None) or (self.episode_steps >= self.max_steps)

    def min_steps_to_complete(self,):
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
