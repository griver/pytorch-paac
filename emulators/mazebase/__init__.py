from ..environment_creator import BaseEnvironmentCreator


class MazebaseGamesCreator(BaseEnvironmentCreator):
    @staticmethod
    def available_games(*args,**kwargs):
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

    @classmethod
    def add_required_args(cls, argparser):
        show_default = " [default: %(default)s]"
        games = MazebaseGamesCreator.available_games()
        argparser.add_argument('-g', default=games[0], choices=games,
                               help='Name of game', dest='game')
        argparser.add_argument('-m', '--map_size', nargs=4, type=int, default=[5, 5, 5, 5],
                               help='Size of the environment. Expected format is (min_x, max_x, min_y, max_y).' +
                                    ' At the beggining of a new episode (x,y) will be drawn uniformly of a new environment will be drawn uniformly from it'
                                    + show_default)
        argparser.add_argument('-f', '--full_view', action='store_true',
                               help='If the flag is provided then an agent will receive a full map view as an observation.' + show_default)
        argparser.add_argument('--gamma', default=0.99, type=float, help="Discount factor" + show_default, dest="gamma")


class TaxiGamesCreator(MazebaseGamesCreator):
    @staticmethod
    def available_games(*args, **kwargs):
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

    @classmethod
    def add_required_args(cls, argparser):
        show_default = " [default: %(default)s]"
        argparser.add_argument('-g', default='taxi_multi_task', choices=['taxi_multi_task', 'taxi_game'],
                               help='Name of game', dest='game')
        argparser.add_argument('-m', '--map_size', nargs=4, type=int, default=[5, 5, 5, 5],
                               help='Size of the environment. Expected format is (min_x, max_x, min_y, max_y).' +
                                    ' At the beggining of a new episode (x,y) will be drawn uniformly of a new environment will be drawn uniformly from it' + show_default)
        argparser.add_argument('-f', '--full_view', action='store_true',
                               help='If the flag is provided then an agent will receive a full map view as an observation.' + show_default)
        argparser.add_argument('--gamma', default=0.99, type=float, dest="gamma",
                               help="Discount factor" + show_default, )
        argparser.add_argument('--random_seed', default=14, type=int, dest='random_seed',
                               help='Random seed for environments.'+show_default)
        argparser.add_argument('-fa', '--finish_action', action='store_true',
                               help='If provided an agent have to perform special action at the end of each task to complete it.')
        argparser.add_argument('-fr', '--fail_reward', type=float, default=0., dest='fail_reward',
                               help='If agent fails current task it gets fail_reward')