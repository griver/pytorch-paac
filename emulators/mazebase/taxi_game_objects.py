#from __future__ import unicode_literals
from mazebase import games
from mazebase.items import agents
import mazebase.items as maze_items
from mazebase.utils.mazeutils import choice, MazeException
from mazebase.utils import creationutils
from .taxi_featurizers import mb_featurizers, LocalViewFeaturizer, FewHotEncoder, FewHotEncoderPlus
#from six.moves import xrange

import itertools
import numpy as np
import random as rnd


class APickableItem(maze_items.MazeItem):
    """
    This class implements only a part of pickup mechanics:
    mainly different display style for a picked up object
    and the is_pickedup flag switch.

    Without subclasses and the agent that can pickup object it is useless.
    """
    def __init__(self,
                 map_symbol=u'z  ',
                 pickup_color='yellow',
                 **kwargs):

        super(APickableItem, self).__init__(**kwargs)
        self.is_pickedup = False
        self.pickup_color = pickup_color
        self.map_symbol = map_symbol
        self._features = [APickableItem.__name__]

    def pickup(self):
        self.is_pickedup = True

    def dropoff(self):
        self.is_pickedup = False

    def _get_display_symbol(self):
        if not self.is_pickedup:
            return (self.map_symbol, None, None, None)
        else:
            return (None, self.pickup_color, None, None)

    def featurize(self):
        return self._features


class Passenger(APickableItem):
    __MAX_PASSENGERS_IDS = 1
    __ID_TEMPLATE = 'psg%d'
    __PICKEDUP_SFX = '_in_taxi'

    def __init__(self, passenger_id=0, **kwargs):
        super(Passenger, self).__init__(map_symbol=u'p  ',
                                        pickup_color='yellow',
                                        **kwargs)

        assert passenger_id < self.__MAX_PASSENGERS_IDS,\
          "cannot create passenger with id >{0}".format(
            Passenger.__MAX_PASSENGERS_IDS
          )
        self.psg_id = Passenger.__ID_TEMPLATE % passenger_id
        self._features.append(Passenger.__name__)

    def featurize(self):
        curr_state = self.psg_id
        if self.is_pickedup:
            curr_state += Passenger.__PICKEDUP_SFX

        return self._features + [curr_state]

    @classmethod
    def all_features(cls):
        all_features = super(Passenger, cls).all_features()
        ids = list(range(Passenger.__MAX_PASSENGERS_IDS))
        all_features.extend(Passenger.__ID_TEMPLATE % id for id in ids)
        pickedup_id = Passenger.__ID_TEMPLATE+Passenger.__PICKEDUP_SFX
        all_features.extend(pickedup_id % id for id in ids)
        return all_features


class Cargo(APickableItem):
    __MAX_CARGO_IDS = 1
    __ID_TEMPLATE = 'crg%d'
    __PICKEDUP_SFX = '_in_taxi'

    def __init__(self, cargo_id=0, **kwargs):
        super(Cargo, self).__init__(map_symbol=u'  c',
                                    pickup_color='cyan',
                                    **kwargs)

        assert cargo_id < self.__MAX_CARGO_IDS,\
          "cannot create cargo with id >{0}".format(
            Cargo.__MAX_PASSENGERS_IDS
          )

        self.cargo_identifier = Cargo.__ID_TEMPLATE % cargo_id
        self._features.append(Cargo.__name__)

    def featurize(self):
        curr_state = self.cargo_identifier
        if self.is_pickedup:
            curr_state += Cargo.__PICKEDUP_SFX

        return self._features + [curr_state]

    @classmethod
    def all_features(cls):
        all_features = super(Cargo, cls).all_features()
        ids = list(range(Cargo.__MAX_CARGO_IDS))
        all_features.extend(Cargo.__ID_TEMPLATE % id for id in ids)
        pickedup_id = Cargo.__ID_TEMPLATE + Cargo.__PICKEDUP_SFX
        all_features.extend(pickedup_id % id for id in ids)
        return all_features


class Transport(agents.SingleTileMovable):

    def __init__(self, **kwargs):
        super(Transport, self).__init__(**kwargs)
        self._add_action('pickup', self._pickup)
        self._add_action('dropoff', self._dropoff)
        self.item = None
        self.__update_move_actions()
        self._features = ['SingleTileMovable', 'Agent']

    def is_empty(self):
        return self.item is None

    def _pickup(self):
        x,y = self.location
        if self.is_empty():
            pickable = self.game._tile_get_block((x,y), APickableItem)
            if pickable is not None:
                pickable.pickup()
                self.item = pickable

    def _dropoff(self):
        if not self.is_empty():
            # try:
            assert self.item.location == self.location, \
              'Transported item must have the same location as the taxi!'
            # except AssertionError as e:
            #     print('game state:')
            #     self.game.display()
            #     print('agent location:', self.location)
            #     print('passenger location:', self.game.passenger.location, 'picked_up:',self.game.passenger.is_pickedup)
            #     print('cargo location:', self.game.cargo.location, 'picked_up:', self.game.cargo.is_pickedup)
            #     print('carried item:', type(self.item), 'location:', self.item.location)
            #     raise e
            x, y = self.location
            if sum(1 for obj in self.game._map[x][y]
                   if isinstance(obj, APickableItem)) <= 1:
                self.item.dropoff()
                self.item = None

    def __update_move_actions(self):
        def move_with_item(move_action):
            def wrapper():
                move_action()
                if not self.is_empty():
                    nloc = self.location
                    self.game._move_item(self.item.id, location=nloc)

            return wrapper

        for id in ['up', 'down', 'left', 'right']:
            self.actions[id] = move_with_item(self.actions[id])

    def featurize(self):
        if self.item is None:
            return self._features
        return self._features + ['carries_item']


class TaxiAgent(Transport):
    pass


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
            self.actions[a] = self.__only_if_allowed(a, a_fun)

    def __only_if_allowed(self, action_name, action_fun):
        def wrapper():
            task = self.game.task()
            #if no specific task is assigned or task allows that action:
            if (not task) or task.allowed_action(action_name):
                action_fun()
            self.last_performed_act = action_name

        return wrapper

    def forced_dropoff(self):
        """
        we wrapped all actions so that the current task can prohibit them
        but sometimes we just have to dropoff or pickup an object anyway(i.e. during episode creation)!
        """
        if not self.is_empty():
            self.item.dropoff()
            self.item = None

    def forced_pickup(self):
        """
        we wrapped all actions so that the current task can prohibit them
        but sometimes we just have to dropoff or pickup an object anyway!
        """
        self._pickup()


class OldTaxi(games.WithWaterAndBlocksMixin):
    '''
    Agent picks up passengers all around the map
    and drops them of in the fixed target location
    '''
    def __init__(self, **kwargs):
        self.agent_cls = TaxiAgent
        self.max_steps = kwargs.get('max_episode_steps', 300)
        self.episode_steps = 0

        super(OldTaxi, self).__init__(**kwargs)
        # Here we directly modify BaseMazeGame.__all_possible_features property:
        features = super(OldTaxi, self).all_possible_features()
        features.extend(Passenger.all_features())
        features.sort()

    def _reset(self):
        super(OldTaxi, self)._reset()

        agent_loc = choice(creationutils.empty_locations(self,
                                                   bad_blocks=[maze_items.Block]))
        self.agent = self.agent_cls(location=agent_loc)
        self._add_agent(self.agent, "TaxiAgent")
        n_items = 2
        placement_locs = self._get_placement_locs(agent_loc, n_items)
        target_loc, psg_loc = rnd.sample(placement_locs, n_items)

        self.target = maze_items.Goal(location=target_loc)
        self._add_item(self.target)

        self.passenger = Passenger(location=psg_loc)
        self._add_item(self.passenger)
        self.episode_steps = 0


    def _get_placement_locs(self, agent_loc, n_required):
        for _ in range(10):
            visited, _ = creationutils.dijkstra(self, agent_loc,
                                                creationutils.agent_movefunc)
            empty_locs = set(creationutils.empty_locations(self,
                                                           bad_blocks=[maze_items.Block, TaxiAgent]))
            reachable_locs = list(empty_locs & set(visited))
            n_lack = max(n_required - len(reachable_locs), 0)
            if n_lack == 0:
                break
            self._remove_adjacent_blocks(reachable_locs + [agent_loc, ], n_lack)
        else:
            raise MazeException('There is no enough space to place game items')
        return reachable_locs

    def _remove_adjacent_blocks(self, reachable_locs, num_blocks):
        moves = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        blocks = []
        for x, y in reachable_locs:
            for move_x, move_y in moves:
                loc_new = (x + move_x, y + move_y)
                if self._in_bounds(loc_new):
                    block = self._tile_get_block(loc_new, maze_items.Block)
                    if block is not None:
                        blocks.append(block)

        blocks = rnd.sample(blocks, num_blocks)
        for b in blocks:
            self._remove_item(b.id)

    def task(self):
        if self._task_is_done():
            return None
        else:
            return 0

    def _side_information(self):
        if self._task_is_done():
            return [[None, 0]] #with accordance to the original mazebase info
        else:
            return [[None, 1]]

    def _task_is_done(self):
        psg_in_target_loc = self.passenger.location == self.target.location
        is_dropped_off = self.passenger.is_pickedup is False
        return psg_in_target_loc and is_dropped_off

    def _step(self):
        self.episode_steps += 1

    def _get_reward(self, id):
        reward = super(OldTaxi, self)._get_reward(id)
        reward -= 0.1
        if self.passenger.is_pickedup:
            reward += 0.1
        elif self.passenger.location == self.target.location: #and passenger is dorpped off
            reward = 1.
        return reward

    def _finished(self):
        return self._task_is_done() or (self.episode_steps >= self.max_steps)


def user_action(game, actions):
    act = None
    while act not in actions:
        if act is not None:
            print("{0} is not a valid action! Valid actions are: {1}".format(act, actions))

        act = input('Input your action:\n')

    game.act(act)


def print_obs(obs, cel_len=35):
    line_sep = '\n'+'-'*(cel_len+1)*3
    cell_str = '{0:^' + str(cel_len) + '}'
    obs = list(zip(*obs)) #matrix transpose:
    for row in reversed(obs): #rows go from lowest to highest
        for cell in row:
            print(cell_str.format(' '.join(cell)), end='|')
        print(line_sep)


def console_test_play():
    obs_encoder = FewHotEncoder()
    #game = games.SingleGoal(featurizer=featurizers.GridFeaturizer())
    featurizer = LocalViewFeaturizer(window_size=3, notify=True)
    #featurizer = featurizers.RelativeGridFeaturizer(bounds=2, notify=True)
    game = OldTaxi(featurizer=featurizer)
    print('{0}:'.format(type(game).__name__))
    actions = game.actions()
    for i, act in enumerate(actions):
      print(i, act)

    game.reset()
    game.display()
    while not game.is_over():
        print(actions)
        obs, info = game.observe()['observation']
        print('observations:')
        print_obs(obs)
        arr = obs_encoder.encode(obs)
        #featurizers.grid_one_hot(game, obs, np)
        user_action(game, actions)

        game.display()
        print('previous r:', game.reward(), 'total R:', game.reward_so_far())
        print(game.target.location, game.agent.location)
        #time.sleep(1.5)

