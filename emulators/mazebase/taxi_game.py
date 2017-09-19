#from __future__ import unicode_literals
from mazebase import games
from mazebase.items import agents
import mazebase.items as maze_items
from mazebase.utils.mazeutils import choice, MazeException
from mazebase.utils import creationutils
from .local_grid_featurizer import featurizers, LocalGridFeaturizer
from six.moves import xrange

import numpy as np
import random as rnd


class Passenger(maze_items.MazeItem):
  __MAX_PASSENGERS_IDS = 1
  __ID_TEMPLATE = 'psg%d'
  __PICKEDUP_SFX = '_in_taxi'

  def __init__(self, passenger_id=0, is_pickedup=False, **kwargs):
    super(Passenger, self).__init__(**kwargs)
    self.is_pickedup = is_pickedup

    assert passenger_id < self.__MAX_PASSENGERS_IDS,\
      "cannot create passenger with id >{0}".format(
        Passenger.__MAX_PASSENGERS_IDS
      )

    self.psg_id = Passenger.__ID_TEMPLATE % passenger_id


  def pickup(self):
    self.is_pickedup = True

  def dropoff(self):
    self.is_pickedup = False

  def _get_display_symbol(self):
    if not self.is_pickedup:
      return (u'p  ', None, None, None)
    else:
      return (None, 'yellow', None, None)

  def featurize(self):
    curr_state = self.psg_id
    if self.is_pickedup:
      curr_state += Passenger.__PICKEDUP_SFX

    return super(Passenger, self).featurize() + [curr_state]


  @classmethod
  def all_features(cls):
    all_features = super(Passenger, cls).all_features()
    ids = list(xrange(Passenger.__MAX_PASSENGERS_IDS))
    all_features.extend(Passenger.__ID_TEMPLATE % id for id in ids)
    pickedup_id = Passenger.__ID_TEMPLATE+Passenger.__PICKEDUP_SFX
    all_features.extend(pickedup_id % id for id in ids)
    return all_features


class Transport(agents.SingleTileMovable):

  def __init__(self, **kwargs):
    super(Transport, self).__init__(**kwargs)
    self._add_action('pickup', self.__pickup)
    self._add_action('dropoff', self.__dropoff)
    self.passenger = None
    self.__update_move_actions()

  def is_empty(self):
    return self.passenger is None

  def __pickup(self):
    x,y = self.location
    passenger = self.game._tile_get_block((x,y), Passenger)
    if passenger is not None and self.is_empty():
      passenger.pickup()
      self.passenger = passenger

  def __dropoff(self):
    if not self.is_empty():
      assert  self.passenger.location == self.location, \
        'Transported passenger must have the same location as the taxi!'
      self.passenger.dropoff()
      self.passenger = None

  def __update_move_actions(self):
    def move_with_passenger(move_action):
      def wrapper():
        move_action()
        if not self.is_empty():
          nloc = self.location
          self.game._move_item(self.passenger.id, location=nloc)

      return wrapper

    for id in ['up', 'down', 'left', 'right']:
      self.actions[id] = move_with_passenger(self.actions[id])


class TaxiAgent(Transport):
  pass


class TaxiGame(games.RewardOnEndMixin,
               games.WithWaterAndBlocksMixin,
               games.BaseVocabulary):
  '''
  Agent picks up passengers all around the map
  and drops them of in the fixed target location
  '''
  def __init__(self, **kwargs):
    super(TaxiGame, self).__init__(**kwargs)
    # Here we directly modify BaseMazeGame.__all_possible_features property:
    features = super(TaxiGame, self).all_possible_features()
    features.extend(Passenger.all_features())
    features.sort()

  def _reset(self):
    super(TaxiGame, self)._reset()

    loc = choice(creationutils.empty_locations(self,
                                               bad_blocks=[maze_items.Block]))
    self.agent = TaxiAgent(location=loc)
    self._add_agent(self.agent, "TaxiAgent")

    visited, _ = creationutils.dijkstra(self, loc,
                                        creationutils.agent_movefunc)

    empty_locs = set(creationutils.empty_locations(self))
    suitable_locs = list(empty_locs & set(visited))
    target_loc, psg_loc = rnd.sample(suitable_locs, 2)

    self.target = maze_items.Goal(location=target_loc)
    self._add_item(self.target)

    self.passenger = Passenger(location=psg_loc)
    self._add_item(self.passenger)

  def _side_information(self):
    return super(TaxiGame, self)._side_information() + \
           [[self.FEATURE.GOTO] + self.target.featurize()]

  def _finished(self):
    psg_in_target_loc = self.passenger.location == self.target.location
    is_dropped_off = self.passenger.is_pickedup is False
    return psg_in_target_loc and is_dropped_off


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


def main():
    #game = games.SingleGoal(featurizer=featurizers.GridFeaturizer())
    featurizer = LocalGridFeaturizer(window_size=3, notify=True)
    #featurizer = featurizers.RelativeGridFeaturizer(bounds=2, notify=True)
    game = TaxiGame(featurizer=featurizer)
    print('{0}:'.format(type(game).__name__))
    actions = game.actions()
    for i, act in enumerate(actions):
      print(i, act)

    game.reset()
    game.display()
    while not game.is_over():
        print(actions)
        obs, info = game.observe()['observation']
        print_obs(obs)
        featurizers.grid_one_hot(game, obs, np)
        user_action(game, actions)

        game.display()
        print('previous r:', game.reward(), 'total R:', game.reward_so_far())
        print(game.target.location, game.agent.location)
        #time.sleep(1.5)


#if __name__ == '__main__':
