import logging

import mazebase.games as games
import numpy as np
from mazebase.games import featurizers

from ..environment import BaseEnvironment
from .taxi_featurizers import LocalViewFeaturizer, GlobalViewFeaturizer

DEFAULT_LOCAL_VIEW_SIZE = (5,5)
DEFAULT_MAP_SIZE = (5,10,5,10)

MAZEBASE_GAME_CLASSES = frozenset([
    games.SingleGoal,
    games.Switches,
    games.MultiGoals,
    games.ConditionedGoals,
    games.PushBlock,
    games.PushBlockCardinal,
    games.Goto,
    games.GotoHidden,
    games.LightKey,
    games.BlockedDoor,
])

def camel_to_snake_case(s):
    'CamelCase --> snake_case'
    return ''.join([('_' if i and ch.isupper() else '') + ch.lower() for i, ch in enumerate(s)])

MAZEBASE_GAMES = {camel_to_snake_case(cls.__name__):cls for cls in MAZEBASE_GAME_CLASSES}

def get_from_args(args, name, default):
    return args.name if hasattr(args, name) else default


class MazebaseEmulator(BaseEnvironment):
    """
    Adapts games from mazebase to the BaseEnvironment interface.
    """
    @staticmethod
    def available_games():
        return MAZEBASE_GAMES

    def __init__(self, emulator_id, game, full_view=False, verbose=0,
                 view_size=DEFAULT_LOCAL_VIEW_SIZE, map_size=DEFAULT_MAP_SIZE, **unknown):
        if verbose >= 2:
            logging.debug('Emulator#{} received unknown args: {}'.format(emulator_id, unknown))

        available_games = self.available_games()
        assert game in available_games, '{0}: There is no such game in the mazebase framework'.format(game)
        game_cls = available_games[game]

        if full_view:
            featurizer = GlobalViewFeaturizer(notify=True)
        else:
            featurizer = LocalViewFeaturizer(window_size=view_size, notify=True)

        self.game = game_cls(map_size=map_size,
                             featurizer=featurizer,
                             max_episode_steps=300)

        state, _, _, _ = self._observe() #masebase resets games during __init__
        self.observation_shape = state.shape
        self.legal_actions = self.game.actions()
        assert 'pass' in self.legal_actions, 'There should be noop action among the available actions!'
        self.noop = 'pass'
        self.id = emulator_id
        if verbose > 2:
            logging.debug('Intializing mazebase.{0} emulator_id={1}'.format(game, self.id))
            logging.warning("The games from MazeBase don't use the random_seed argument")
        # Mazebase generates random samples
        # within itself in different modules across the package; therefore,
        # we can't fix a random generator without rewriting mazebase.
        # self.rand_gen = random.Random(args.random_seed)

    def reset(self):
        #There is no activity in the masebase games aside from agent's activity.
        #Therefore, performing a random start changes nothing in the games
        self.game.reset()
        state, _, _, info = self._observe()
        return state, info

    def _observe(self):
        # returns s, r, is_done, info
        game_data = self.game.observe()
        state, info = game_data['observation']
        state = featurizers.grid_one_hot(self.game, state, np=np)
        state = state.transpose((2,0,1)) #images go in the [C,H,W] shape for pytorch
        return state, game_data['reward'], self.game.is_over(), info

    def next(self, action):
        '''
        Performs action.
        Returns the next state, reward, and termination signal
        '''
        action = self.legal_actions[action]
        self.game.act(action) # no need for action repetition here
        state, reward, is_done, info, = self._observe()
        return state, reward, is_done, info

    def get_legal_actions(self):
        return self.legal_actions

    def get_noop(self):
        return self.noop

