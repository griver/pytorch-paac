from .mazebase.mazebase_emulator import MazebaseEmulator, MAZEBASE_GAMES
from .atari.atari_emulator import AtariEmulator
from .mazebase.taxi_emulator import TaxiEmulator, TAXI_GAMES

emulators = [MazebaseEmulator, TaxiEmulator, AtariEmulator]