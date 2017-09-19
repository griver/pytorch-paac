class EnvironmentCreator(object):

    def __init__(self, args):
        """
        Creates an object from which new environments can be created
        :param args:
        """
        from emulators import MAZEBASE_GAMES
        if args.game in MAZEBASE_GAMES:
            num_actions, create_env = self._init_mazebase(args)
        else:
            num_actions, create_env = self._init_atari(args)

        self.num_actions = num_actions
        self.create_environment = create_env


    def _init_atari(self, args):
        from emulators import AtariEmulator
        from ale_python_interface import ALEInterface
        filename = args.rom_path + "/" + args.game + ".bin"
        ale_int = ALEInterface()
        ale_int.loadROM(str.encode(filename))
        num_actions = len(ale_int.getMinimalActionSet())
        create_env = lambda i: AtariEmulator(i, args)
        return num_actions, create_env

    def _init_mazebase(self, args):
        from emulators import MazebaseEmulator
        create_env = lambda i: MazebaseEmulator(i, args)
        num_actions = len(create_env(-1).legal_actions())
        return num_actions, create_env