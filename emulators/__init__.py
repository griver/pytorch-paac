from .atari import AtariGamesCreator
from .mazebase import TaxiGamesCreator, MazebaseGamesCreator
from .vizdoom import VizdoomGamesCreator
from .warehouse import WarehouseGameCreator
from .stoch_graphs import StochGraphCreator

game_creators = [VizdoomGamesCreator,
                 AtariGamesCreator,
                 TaxiGamesCreator,
                 MazebaseGamesCreator,
                 ]