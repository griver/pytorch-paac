from .mazebase_emulator import *

from .taxi_game import TaxiGame, FewHotEncoder
from .taxi_multi_task import TaxiMultiTask, FixedTaxiMultiTask


TAXI_CLASSES = frozenset([
    TaxiGame,
    TaxiMultiTask,
    FixedTaxiMultiTask
])

TAXI_GAMES = {camel_to_snake_case(cls.__name__):cls for cls in TAXI_CLASSES}


class TaxiEmulator(MazebaseEmulator):
    """
    Adapts games from mazebase to the BaseEnvironment interface.
    """
    @staticmethod
    def available_games():
        return TAXI_GAMES


    def __init__(self, *args, **kwargs):
        self._encoder = FewHotEncoder()
        super(TaxiEmulator, self).__init__(*args, **kwargs)


    def reset(self):
        #There is no activity in the masebase games aside from agent's activity.
        #Therefore, performing a random start changes nothing in the games
        self.game.reset()
        state, _, _, info = self._observe()
        state = self._merge_observation_with_task_info(state, info)
        return state, info

    def _observe(self):
        # returns s, r, is_done, info
        game_data = self.game.observe()
        state, info = game_data['observation']
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
        state = self._merge_observation_with_task_info(state, info)
        return state, reward, is_done, info

    def _merge_observation_with_task_info(self, state, info):
        return np.concatenate([state.reshape(-1), info[1:]])

    @staticmethod
    def split_observation_and_task_info(env_states, obs_shape):
        """
        Splits env_states into map observations and additional task information
        :param state: A previously merged state([flatten(observation), task_id])
        :return: A tuple (observation, task_id)
        """
        batch_size = env_states.shape[0]
        return env_states[:,:-1].reshape(batch_size,*obs_shape), env_states[:,-1]


