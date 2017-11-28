class EnvironmentCreator(object):

    def __init__(self, args):
        """
        Creates an object from which new environments can be created
        :param args:
        """
        from emulators import TaxiEmulator, MazebaseEmulator
        if args.game in MazebaseEmulator.available_games():
            num_actions, create_env = self._init_default(args, MazebaseEmulator)
        elif args.game in TaxiEmulator.available_games():
            num_actions, create_env = self._init_default(args, TaxiEmulator)
        else:
            num_actions, create_env = self._init_atari(args)

        self.num_actions = num_actions
        self.create_environment = create_env


    def _init_atari(self, args):
        """
        First checks if ALE can find rom with the demanded game,
        then returns a functions for creating emulator instances
        """
        from emulators import AtariEmulator
        from ale_python_interface import ALEInterface
        filename = args.rom_path + "/" + args.game + ".bin"
        ale_int = ALEInterface()
        ale_int.loadROM(str.encode(filename))
        num_actions = len(ale_int.getMinimalActionSet())
        create_env = lambda i: AtariEmulator(i, args)
        return num_actions, create_env

    def _init_default(self, args, emulator_cls):
        """
        A simple method for cases when there is no need
        in any preprocessing before creating the emulators
        """
        create_env = lambda i: emulator_cls(i, args)
        num_actions = len(create_env(-1).legal_actions)
        return num_actions, create_env