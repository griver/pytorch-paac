from ..environment_creator import BaseEnvironmentCreator


class AtariGamesCreator(BaseEnvironmentCreator):

    @staticmethod
    def available_games(resource_folder, config_ext='.bin'):
        games = []
        if resource_folder is None:
            return games

        import os
        def remove_ext(filename):
            return filename[:-len(config_ext)]

        for root, dirnames, filenames in os.walk(resource_folder):
            fld_games = [remove_ext(fn) for fn in filenames if fn.lower().endswith(config_ext)]
            games.extend(fld_games)
        return sorted(games)

    @staticmethod
    def get_environment_class():
        from .atari_emulator import AtariEmulator
        return AtariEmulator

    @staticmethod
    def add_required_args(argparser):
        argparser.add_argument('-g', default='pong', help='Name of game', dest='game')
        argparser.add_argument('-rf', '--resource_folder', default='./resources/atari_roms',
            help='Directory with files required for the game initialization' +
            '(i.e. binaries for ALE and scripts for ViZDoom), default=./resources/atari_roms',
            dest="resource_folder")
        argparser.add_argument('-v', '--visualize', action='store_true',
            help="Show a game window", dest="visualize")
        argparser.add_argument('--single_life_episodes', action='store_true',
            help="If True, training episodes will be terminated when a life is lost (for games)",
            dest="single_life_episodes")
        argparser.add_argument('-rs', '--random_start', default=True, type=bool,
            help="Whether or not to start with 30 noops for each env. Default True",
            dest="random_start")
        argparser.add_argument('--seed', default=3, type=int, help='Sets the random seed.',
            dest='random_seed')
        argparser.add_argument('-hw', '--history_window', type=int, default=1,
                               help="Number of observations forming a state", dest='history_window')
