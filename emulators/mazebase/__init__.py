from ..environment_creator import BaseEnvironmentCreator


class MazebaseGamesCreator(BaseEnvironmentCreator):
    @staticmethod
    def available_games(**kwargs):
        games = [
          "single_goal",
          "switches",
          "multi_goals",
          "conditioned_goals",
          "push_block",
          "push_block_cardinal",
          "goto",
          "goto_hidden",
          "light_key",
          "blocked_door",
        ]
        return games

    @staticmethod
    def get_environment_class():
        from .mazebase_emulator import MazebaseEmulator
        return MazebaseEmulator

    @staticmethod
    def add_required_args(argparser):
      argparser.add_argument('-g', default='taxi_multi_task',
          choices=['taxi_multi_task', 'taxi_game'], help='Name of game', dest='game')
      argparser.add_argument('-m', '--map_size', nargs=4, type=int, default=[5, 5, 5, 5],
          help='default=[5,5,5,5]. The size of environment of shape (min_x, max_x, min_y, max_y). At the'
          + ' beggining of a new episode size (x,y) of a new environment will be drawn uniformly from it')
      argparser.add_argument('-f', '--full_view', action='store_true',
          help='If the flag is provided then an agent will receive a full map view as an observation.')
      argparser.add_argument('-v', '--verbose', default=1, type=int,
          help="default=1. Verbose determines how much information to show during training",
          dest="verbose")


class TaxiGamesCreator(MazebaseGamesCreator):
    @staticmethod
    def available_games(**kwargs):
        games = [
          "taxi_game",
          "taxi_multi_task",
          "fixed_taxi_multi_task"
        ]
        return games

    @staticmethod
    def get_environment_class():
        from .taxi_emulator import TaxiEmulator
        return TaxiEmulator
