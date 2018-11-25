from ..environment_creator import BaseEnvironmentCreator



class WarehouseGameCreator(BaseEnvironmentCreator):

    @staticmethod
    def available_games(resource_folder, required_files=('.cfg','.json','.wad')):
        if resource_folder is None:
            return ()

        import os
        def remove_ext(filename, ext):
            return filename[:-len(ext)]

        games2files = {}

        for root, dirnames, filenames in os.walk(resource_folder):
            for fn in filenames:
                fn = fn.lower()
                for i,ext in enumerate(required_files):
                    if fn.endswith(ext):
                        game = remove_ext(fn, ext)
                        flags = games2files.setdefault(game, [False for r in required_files])
                        flags[i] = True
                        break

        games = [k for k, flags in games2files.items() if all(flags)]
        return sorted(games)

    @staticmethod
    def get_environment_class():
        from .warehouse_emulator import WarehouseEmulator
        return WarehouseEmulator

    @classmethod
    def add_required_args(cls, argparser):
        show_default = " [default: %(default)s]"
        argparser.add_argument('-g', default='warehouse', help='Game to play.'+show_default, dest='game')
        argparser.add_argument('-rf', '--resource_folder', default='./resources/vizdoom_scenarios',
            help='Directory with files required for the game initialization'+
                 show_default,
            dest="resource_folder")
        argparser.add_argument('--skill_level',  type=int, default=1,
                               help='Skill level determines the complexity of the game'+show_default)
        argparser.add_argument('-rs', default=29,  help='Random Seed.' + show_default)
        argparser.add_argument('-v', '--visualize', action='store_true',
                               help="Show a game window." + show_default, dest="visualize")
