from environment import BaseEnvironment #TODO: Create a package
import mazebase.games as games
from mazebase.games import featurizers
from mazebase.games import curriculum
import numpy as np
from .taxi_game import TaxiGame, TaxiAgent, LocalGridFeaturizer
import logging

IMG_SIZE = (7,7)

GAME_CLASSES = frozenset([
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
    TaxiGame
])

def camel_to_snake_case(s):
    'CamelCase --> snake_case'
    return ''.join([('_' if i and ch.isupper() else '') + ch.lower() for i, ch in enumerate(s)])

GAMES = {camel_to_snake_case(cls.__name__):cls for cls in GAME_CLASSES}

class MazebaseEmulator(BaseEnvironment):
    """
    Adapts games from mazebase to the BaseEnvironment interface.
    """
    def __init__(self, actor_id, args):
        assert args.game in GAMES, \
          '{0}: There is no such game in the mazebase framework'.format(args.game)
        game_cls = GAMES[args.game]
        self.game = game_cls(
            featurizer=LocalGridFeaturizer(window_size=IMG_SIZE, notify=True)
        )
        self.legal_actions = self.game.actions()
        self.noop = np.array([a == 'pass' for a in self.legal_actions], dtype=np.float32)
        self.id = actor_id
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
        state, reward, is_done, _, = self._observe()
        return state, reward, is_done

    def get_legal_actions(self):
        return self.legal_actions

    def get_noop(self):
        return self.noop

    def on_new_frame(self, frame):
        super(MazebaseEmulator, self).on_new_frame(frame)
