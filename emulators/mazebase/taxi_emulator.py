from .mazebase_emulator import *

from .taxi_game_objects import OldTaxi, FewHotEncoder, FewHotEncoderPlus
from .taxi_game import Taxi, FixedTaxi
from .taxi_plus_game import TaxiPlus

TAXI_CLASSES = frozenset([
    OldTaxi,
    Taxi,
    FixedTaxi,
    TaxiPlus
])

TAXI_GAMES = {camel_to_snake_case(cls.__name__):cls for cls in TAXI_CLASSES}

class TaxiEmulator(MazebaseEmulator):
    """
    Adapts games from mazebase to the BaseEnvironment interface.
    """
    @staticmethod
    def available_games():
        return TAXI_GAMES

    def __init__(self, emulator_id, game, full_view=False, verbose=0,
                 view_size=DEFAULT_LOCAL_VIEW_SIZE, map_size=DEFAULT_MAP_SIZE,
                 random_seed=17, finish_action=False, fail_reward=0.,
                 single_task_episodes=False, max_episode_steps=300, preliminary_env=False, tasks=None, **unknown):
        if verbose >= 2:
            logging.debug('Emulator#{} received unknown args: {}'.format(emulator_id, unknown))
        self.emulator_id = emulator_id
        available_games = self.available_games()
        assert game in available_games, '{0}: There is no such game in the mazebase framework'.format(game)
        game_cls = available_games[game]

        if full_view:
            featurizer = GlobalViewFeaturizer(notify=True)
        else:
            featurizer = LocalViewFeaturizer(window_size=view_size, notify=True)
        self._encoder = FewHotEncoderPlus() if game == 'taxi_plus' else FewHotEncoder()

        game_seed = (self.emulator_id * random_seed) % (2**32)
        self.game = game_cls(map_size=map_size,
                             featurizer=featurizer,
                             max_episode_steps=max_episode_steps,
                             random_seed=game_seed,
                             finish_action=finish_action,
                             fail_reward=fail_reward,
                             single_task_episodes=single_task_episodes,
                             preliminary_env=preliminary_env,
                             tasks=tasks)

        state, _, _, _ = self._observe() #masebase resets games during __init__
        self.observation_shape = state.shape
        self.legal_actions = self.game.actions()
        assert 'pass' in self.legal_actions, 'There should be noop action among the available actions!'
        self.noop = 'pass'
        self.id = emulator_id
        if verbose > 2:
            logging.debug('Intializing mazebase.{0} emulator_id={1}'.format(game, self.id))
        # Mazebase generates random samples
        # within itself in different modules across the package; therefore,
        # we can't fix a random generator without rewriting mazebase.
        # self.rand_gen = random.Random(args.random_seed)


    def reset(self):
        #There is no activity in the masebase games aside from agent's activity.
        #Therefore, performing a random start changes nothing in the games
        self.game.reset()
        state, _, _, info = self._observe()
        #state = self._merge_observation_with_task_info(state, info)
        return state, info

    def _observe(self):
        # returns s, r, is_done, info
        game_data = self.game.observe()
        state, info = game_data['observation'] #info looks like this: [str(task_name), int(task_id)]
        state = self._encoder.encode(state)
        state = state.transpose((2,0,1)) #images go in the [C,H,W] shape for pytorch
        return state, game_data['reward'], self.game.is_over(), info

    def next(self, action):
        '''
        Performs action.
        Returns the next state, reward, and termination signal
        '''
        action = self.legal_actions[action]
        self.game.act(action) # no need for action repetition here
        state, reward, is_done, info = self._observe()
        return state, reward, is_done, info

    def display(self):
        state, reward, is_done, info = self._observe()
        print('=============== EM#{} ============'.format(self.emulator_id))
        print('r_t={} task_t+1={} | info_t+1={} | state_t+1:'.format(
            reward, self.game.task(), info
        ))
        self.game.display()

    def set_map_size(self, min_x, max_x, min_y, max_y):
        return self.game.set_map_size(min_x, max_x, min_y, max_y)

    def update_map_size(self, *deltas):
        return self.game.update_map_size(*deltas)

    def get_tasks_history(self):
        return self.game.get_tasks_history()