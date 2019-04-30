from ..environment import BaseEnvironment
import logging
import os
import json
import numpy as np
from .stoch_envs import PeriodicEnvironment
from .load_env import load as load_env
from networkx.drawing import nx_pydot

class StochGraphException(Exception):
    pass

class StochGraphEnv(BaseEnvironment):

    def __init__(
        self,
        emulator_id,
        game,
        resource_folder,
        update_period=1,
        max_episode_steps=500,
        #history_window=1,
        target_reward=1.,
        living_reward=0.0,
        reset_limit=100,
        verbose=0,
        visualize=False,
        random_seed=None,
        **unknown
    ):
        if verbose >= 2 and unknown:
            logging.debug('Emulator#{} received unknown main_args: {}'.format(emulator_id, unknown))
        if visualize:
            logging.warning("Sorry, we can't visualize this environment!")

        self.update_period = update_period
        def create_env(*args, **kwargs):
            return PeriodicEnvironment(self.update_period, *args, **kwargs)

        self.env = load_env(resource_folder, game, create_env)
        self.env_name = game
        self.emulator_id = emulator_id
        self.verbose=verbose
        #self.history_window=history_window
        self.living_reward = living_reward
        self.target_reward = target_reward
        self.max_episode_steps = max_episode_steps
        self.step = 0
        self.target = self._get_target_state()
        self.env.retry_update_limit = reset_limit

        self.done = False
        self._max_actions = self._compute_max_actions()
        self.legal_actions = list(range(self._max_actions))
        self.observation_shape = self._state2obs(self.target).shape

    def reset(self,):
        """"""
        # self.target = self._get_target_state()
        self.step = 0

        has_path = self.env.ensure_path_reset(self.target)
        assert has_path, "Couldn't find a path to the target!"
        assert self.target != self.env.get_current_state(), \
            'Env resets to a target state!'

        self.done = False
        return self._obs_and_info()

    def next(self, action):
        """
        :param action:
        :return:
        """
        if self.done:
            obs, info = self._obs_and_info()
            return obs, 0.0, self.done, info

        has_path = self.env.ensure_path_update_state(
            action, self.target
        )
        assert has_path, "Couldn't find a path to the target!"
        state = self.env.get_current_state()
        self.step += 1

        if state is self.target:
            self.done = True
            reward = self.target_reward
        else:
            self.done = self.step >= self.max_episode_steps
            reward = self.living_reward

        obs, info = self._obs_and_info()
        return obs, reward, self.done, info

    def _obs_and_info(self):
        node = self.env.get_current_state()
        obs = self._state2obs(node)

        n_acts = len(node.get_outcoming())
        act_mask = np.zeros(self._max_actions, dtype=np.float32)
        act_mask[:n_acts] = 1.
        info = {
            'n_acts': n_acts,
            'act_mask': act_mask
        }

        return obs, info

    @staticmethod
    def _state2obs(state):
        return np.array([int(c) for c in state.coords()], dtype=np.float32)

    def _get_target_state(self):
        if 'torus' in self.env_name:
            side_x = side_y = int(np.sqrt(self.env.get_vertices_number()))
            target_id = (side_x//2)*side_y + side_y//2
        else:
            target_id = self.env.get_vertices_number() // 2

        return self.env.get_vertex(target_id)

    def close(self):
        del self.env

    def _compute_max_actions(self, ):
        return max(len(v.get_outcoming()) for v in self.env.vertices())