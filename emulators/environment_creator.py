
class BaseEnvironmentCreator(object):
    @staticmethod
    def add_required_args(argparser):
        '''Adds arguments required to create new environments'''
        raise NotImplementedError()

    @staticmethod
    def available_games(resource_folder):
        """:return A list of all available games it can create"""
        raise NotImplementedError()

    @staticmethod
    def get_environment_class():
      raise NotImplementedError()

    def __init__(self, args):
        resource_folder = getattr(args, 'resource_folder', None)
        if args.game in self.available_games(resource_folder):
            self.env_class = self.get_environment_class()
            self._default_args = args
            new_fields = self._init_default(args)
            for name, value in new_fields.items():
                setattr(self, name, value)
        else:
            raise ValueError(
                "{0} Can't find {0} game".format(self.__class__, args.game)
            )

    def _init_default(self, args):
        """
        A simple method for cases when there is no need
        in any preprocessing before creating the emulators
        """
        test_env = self.create_environment(-1)
        num_actions = len(test_env.legal_actions)
        obs_shape = test_env.observation_shape
        return dict(
          num_actions=num_actions,
          obs_shape=obs_shape
        )

    def create_environment(self, env_id, args=None):
        """
        Сreates a new environment that can be used for training
        or testing an agent
        :arg args - Args specific for this particular game instance.
        All args needed for environment creation that wasn't
        specified in  the argument will be taken from self.default_args()

        """
        if args:
            #TODO: merge the default and specific args
            raise NotImplementedError()
        return self.env_class(env_id, self.default_args())

    def default_args(self):
        return self._default_args