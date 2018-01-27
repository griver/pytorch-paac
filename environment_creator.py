import emulators as em
class EnvironmentCreator(object):

    def __init__(self, args):
        """
        Creates an object from which new environments can be created
        :param args:
        """
        if args.game in em.available_mazebase_games():
            field_dict = self._init_default(args, em.get_mazebase_emulator_cls())
        elif args.game in em.available_taxi_games():
            field_dict = self._init_default(args, em.get_taxi_emulator_cls())
        elif args.game in em.available_atari_games(resource_folder=args.resource_folder):
            field_dict = self._init_default(args, em.get_atari_emulator_cls())
        elif args.game in em.available_vizdoom_games(resource_folder=args.resource_folder):
            field_dict = self._init_default(args, em.get_vizdoom_emulator_cls())
        else:
            raise ValueError("Can't find {0} game".format(args.game))

        for name, value in field_dict.items():
            setattr(self, name, value)

    def _init_default(self, args, emulator_cls):
        """
        A simple method for cases when there is no need
        in any preprocessing before creating the emulators
        """
        create_env = lambda i: emulator_cls(i, args)
        test_env = create_env(-1)
        num_actions = len(test_env.legal_actions)
        obs_shape = test_env.observation_shape
        return dict(
            create_environment=create_env,
            num_actions=num_actions,
            obs_shape=obs_shape
        )