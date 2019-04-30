from ..environment_creator import BaseEnvironmentCreator
import os

class StochGraphCreator(BaseEnvironmentCreator):

    @staticmethod
    def available_games(resource_folder, config_extensions=('.dot', '.json')):
        games = []
        if resource_folder is None:
            return games

        def is_config(filename):
            filename = filename.lower()
            for ext in config_extensions:
                if fn.endswith(ext):
                    return True
            return False

        remove_ext = lambda fn: os.path.splitext(fn)[0]

        for root, dirnames, filenames in os.walk(resource_folder):
            dirnames.sort()
            path = os.path.relpath(root, resource_folder).split(os.sep)
            #path = root.split(os.sep)
            #print('_'.join(path))
            for fn in sorted(filenames):
                if is_config(fn):
                    game = remove_ext(fn)
                    game = '-'.join(path + [game])
                    #print('{}'.format(game))
                    games.append(game)

        return games

    @staticmethod
    def get_environment_class():
        from .env_adapter import StochGraphEnv
        return StochGraphEnv

    @classmethod
    def add_required_args(cls, argparser):
        argparser.add_argument(
            '-g', default='torus-5x5-1',
            help='Name of the environment', dest='game'
        )
        argparser.add_argument(
            '-rf', '--resource-folder', type=str, dest='resource_folder',
            default='resources/stoch_graphs',
            help='Folder that contains saved stochastic graphs'
        )
        argparser.add_argument(
            '-p', '--period', type=int, default=1, dest='update_period',
            help="Update availability of graph edges every <update_period> steps"
        )
        argparser.add_argument(
            '--max-episode-steps', type=int, default=300,
            help='Maximum number of steps in each episode'
        )
        argparser.add_argument(
            '--seed', default=3, type=int, help='Sets the random seed.',
            dest='random_seed'
        )
        #argparser.add_argument('-hw', '--history_window', type=int, default=1,
        #                       help="Number of observations forming a state", dest='history_window')
