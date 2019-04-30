from ..environment_creator import BaseEnvironmentCreator
from os.path import join as pjoin
import glob

class TextWorldCreator(BaseEnvironmentCreator):

    @classmethod
    def add_required_args(cls, argparser):
        argparser.add_argument('-g', type=int, nargs="*", help="List of games to use for training.")
        argparser.add_argument('-rf', '--resource_folder', default='./resources/textworld/train',
            help='Directory with files required for the game initialization' +
            '(i.e. binaries for ALE and scripts for ViZDoom), default=./resources/textworld/train',
            dest="resource_folder")
        argparser.add_argument('-v', '--visualize', action='store_true',
            help="Show a game window", dest="visualize")
        argparser.add_argument('--seed', default=1234, type=int, help='Sets the random seed.',
            dest='random_seed')
        argparser.add_argument('-fp', '--fail_penalty', type=float, default=0.0,
                               help="Penalty in case the agent does not succeed in episode")



    @staticmethod
    def available_games(resource_folder, config_ext='ulx'):
        path_pattern = pjoin(resource_folder, "*." + config_ext)
        filenames = glob(path_pattern)
        return filenames

    @staticmethod
    def get_environment_class():
        from .textworld_env import TextworldAdapter
        return TextworldAdapter