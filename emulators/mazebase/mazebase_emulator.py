from environment import BaseEnvironment #TODO: Create a package
import mazebase.games as games
from mazebase.games import featurizers
from mazebase.games import curriculum
import numpy as np
from .taxi_featurizers import LocalViewFeaturizer, GlobalViewFeaturizer

import logging

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

    def __init__(self, actor_id, args):
        available_games = self.available_games()
        assert args.game in available_games, \
          '{0}: There is no such game in the mazebase framework'.format(args.game)
        game_cls = available_games[args.game]

        full_view = args.full_view if hasattr(args, 'full_view') else False
        # view_size is relevant only if full_view is False
        view_size = args.view_size if hasattr(args, 'view_size') else DEFAULT_LOCAL_VIEW_SIZE
        map_size = args.map_size if hasattr(args, 'map_size') else DEFAULT_MAP_SIZE
        if full_view:
            featurizer = GlobalViewFeaturizer(notify=True)
        else:
            featurizer = LocalViewFeaturizer(window_size=view_size, notify=True)

        self.game = game_cls(map_size=map_size, featurizer=featurizer)

        state, _, _, _ = self._observe() #masebase resets games during __init__
        self.observation_shape = state.shape
        self.legal_actions = self.game.actions()
        self.noop = np.array([a == 'pass' for a in self.legal_actions], dtype=np.float32)
        self.id = actor_id
        if args.verbose > 2:
            logging.debug('Intializing mazebase.{0} emulator_id={1}'.format(args.game, self.id))
            logging.warning("The games from MazeBase don't use the random_seed argument")
        # Mazebase generates random samples
        # within itself in different modules across the package; therefore,
        # we can't fix a random generator without rewriting mazebase.
        # self.rand_gen = random.Random(args.random_seed)

    def get_initial_state(self):
        #There is no activity in the masebase games aside from agent's activity.
        #Therefore, performing a random start changes nothing in the games
        self.game.reset()
        state, _, _, _ = self._observe()
        return state

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
        action = self.legal_actions[np.argmax(action)]
        self.game.act(action) # no need for action repetition here
        state, reward, is_done, info, = self._observe()
        return state, reward, is_done

    def get_legal_actions(self):
        return self.legal_actions

    def get_noop(self):
        return self.noop

    def on_new_frame(self, frame):
        super(MazebaseEmulator, self).on_new_frame(frame)

