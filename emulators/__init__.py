from .atari import AtariGamesCreator
from .mazebase import TaxiGamesCreator, MazebaseGamesCreator
from .vizdoom import VizdoomGamesCreator

game_creators = [VizdoomGamesCreator, AtariGamesCreator, TaxiGamesCreator, MazebaseGamesCreator]