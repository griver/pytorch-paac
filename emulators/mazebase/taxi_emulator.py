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