
class BaseEnvironmentCreator(object):
    @classmethod
    def add_required_args(cls, argparser):
        '''Adds arguments required to create new environments'''
        raise NotImplementedError()

    @staticmethod
    def available_games(resource_folder):
        """:return A list of all available games it can create"""
        raise NotImplementedError()

    @staticmethod
    def get_environment_class():
      raise NotImplementedError()

    def __init__(self, **default_emulator_args):
        resource_folder = default_emulator_args.get('resource_folder', None)
        game_name = default_emulator_args['game']
        if game_name in self.available_games(resource_folder):
            self.env_class = self.get_environment_class()
            self._default_args = default_emulator_args
            new_fields = self._init_default()
            for name, value in new_fields.items():
                setattr(self, name, value)
        else:
            raise ValueError(
                "{0} Can't find {0} game".format(self.__class__, game_name)
            )

    def _init_default(self):
        """
        A simple method for cases when there is no need
        in any preprocessing before creating the emulators
        """
        #
        test_env = self.create_environment(-1, visualize=False, verbose=2)
        num_actions = len(test_env.legal_actions)
        obs_shape = test_env.observation_shape
        return dict(
          num_actions=num_actions,
          obs_shape=obs_shape
        )

    def create_environment(self, env_id, **specific_args):
        """
        Сreates a new environment that can be used for training
        or testing an agent
        :arg specific_args - Args specific for this particular game instance.
        All args needed for environment creation that wasn't
        specified in  the argument will be taken from self.default_args()

        """
        if len(specific_args):
            for k,v in self._default_args.items():
                if k not in specific_args:
                    specific_args[k]=v
            return self.env_class(env_id, **specific_args)
        else:
            return self.env_class(env_id, **self._default_args)

    def default_args(self):
        return dict(self._default_args)