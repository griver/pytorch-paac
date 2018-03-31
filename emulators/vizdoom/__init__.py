from ..environment_creator import BaseEnvironmentCreator


class VizdoomGamesCreator(BaseEnvironmentCreator):

    @staticmethod
    def available_games(resource_folder, config_ext='.cfg'):
        games = []
        if resource_folder is None:
            return games

        import os
        def remove_ext(filename):
            return filename[:-len(config_ext)]

        for root, dirnames, filenames in os.walk(resource_folder):
            games.extend(
              [remove_ext(fn) for fn in filenames if fn.lower().endswith(config_ext)]
            )
        return sorted(games)

    @staticmethod
    def get_environment_class():
        from .vizdoom_emulator import VizdoomEmulator
        return VizdoomEmulator

    @staticmethod
    def add_required_args(argparser):
        show_default = " [default: %(default)s]"
        argparser.add_argument('-g', default='simpler_basic', help='Game to play.'+show_default, dest='game')
        argparser.add_argument('-rf', '--resource_folder', default='./resources/vizdoom_scenarios',
            help='Directory with files required for the game initialization'+
                 show_default,
            dest="resource_folder")
        argparser.add_argument('-v', '--visualize', action='store_true',
                               help="Show a game window." + show_default, dest="visualize")
        argparser.add_argument('-hw', '--history_window', type=int, default=1,
                               help="Number of observations forming a state"+show_default, dest='history_window')
        argparser.add_argument('--gray', action='store_true', help='Use GRAY8 format instead of BGR24.' + show_default +
                                '\n See: https://github.com/mwydmuch/ViZDoom/blob/master/doc/Types.md#screenformat')
