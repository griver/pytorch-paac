import textworld
from textworld import EnvInfos
import gym
import re

from ..environment import BaseEnvironment

class TextworldAdapter(BaseEnvironment):

    def __init__(
            self,
            emulator_id,
            game_spec,
            max_steps=50,
            fail_penalty=0.,
            random_seed=1234,
    ):
        self.emulator_id = emulator_id
        self.seed = random_seed*(emulator_id+1)

        self.env = gym.make(game_spec)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.env.seed(self.seed)

        self.fail_penalty = fail_penalty
        self.max_steps = max_steps
        self.curr_step = 0
        self.score = 0.
        self.is_done = False

    def reset(self):
        self.curr_step = 0
        self.score = 0.
        self.is_done = False
        obs, info = self.env.reset()
        return obs, info

    def next(self, actions):
        if self.is_done:
            return None, 0., self.is_done, {
                'steps': self.curr_step,
                'score': self.score
            }

        obs, new_score, self.is_done, info = self.env.step(actions)
        self.curr_step += 1
        reward = new_score - self.score #actual reward
        self.score = new_score

        if self.is_done and (not info['has_won']):
                reward -= self.fail_penalty

        return obs, reward, self.is_done, info

    def close(self):
        self.env.close()