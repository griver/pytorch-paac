import itertools as it
import logging

import cv2
import numpy as np
import skimage.color
import skimage.transform
from scipy.misc import imresize
from vizdoom import Mode, ScreenResolution, ScreenFormat, DoomGame

from ..environment import BaseEnvironment, create_history_observation
from utils import join_path


def skimage_resize(image, size): #default from learning_pytorch.py
    #1.39 ms ± 260 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each), for the 240x320 image and (60,90) size 
    img = skimage.transform.resize(image, size)*255 #returns an array floats in the range [0,1]
    img = img.astype(np.uint8) #but we want an array of integers between 0 and 255
    if len(img.shape) == 2:
        return img[np.newaxis] #(H,W) -> (1,H,W)
    return img.transpose(2,0,1) #(H,W,C) -> (C,H,W)


def cv2_resize(image, size):
    #39.2 µs ± 292 ns per loop (mean ± std. dev. of 7 runs, 1000 loops each), basicaly 30 times faster than the skimage resize.
    img = cv2.resize(image, size[::-1]) #for some reason cv2 interprets a size argument as (n_cols, n_rows) instead of (n_rows, n_cols).
    if len(img.shape) == 2:
        return img[np.newaxis] #(H,W) -> (1,H,W)
    return img.transpose(2,0,1) #(H,W,C) -> (C,H,W)


class VizdoomEmulator(BaseEnvironment):
    # Values bellow are taken from ViZDoom/examples/python/learining_pytorch.py
    DEFAULT_ACTION_REPEAT = 6  # default is 12
    DEFAULT_SCREEN_SIZE = (60, 90)  # default is (30,45)
    DEFAULT_REWARD_COEF = 1 / 100
    SCREEN_RESOLUTION = ScreenResolution.RES_320X240
    MODE = Mode.PLAYER

    def __init__(self, emulator_id, game, resource_folder, gray=False, reward_coef=1/100,
                 action_repeat=6, history_window=1, screen_size=(60,90), verbose=0, visualize=False, **unknown):
        if verbose >= 2:
            logging.debug('Initializing Vizdoom.{}. emulator_id={}'.format(game, emulator_id))
            logging.debug('Emulator#{} received unknown args: {}'.format(emulator_id, unknown))
        doom_game = DoomGame()
        config_file_path = join_path(resource_folder, game+'.cfg')
        doom_game.load_config(config_file_path)
        doom_game.set_window_visible(visualize)
        doom_game.set_screen_resolution(self.SCREEN_RESOLUTION)
        doom_game.set_screen_format(ScreenFormat.GRAY8 if gray else ScreenFormat.BGR24)
        doom_game.set_mode(self.MODE)
        if self.MODE == Mode.SPECTATOR:
            doom_game.add_game_args("+freelook 1")

        # with a fixed seed all episodes in this environment will be identical
        #doom_game.set_seed(args.random_seed)
        # doom_game.add_available_game_variable(vizdoom.GameVariable.AMMO2)
        doom_game.init()
        self.game = doom_game
        self.legal_actions, self.noop = self._define_actions(self.game)
        self._preprocess = cv2_resize
        self.screen_size = screen_size
        self.reward_coef = reward_coef
        self.action_repeat = action_repeat
        self.history_window = history_window

        num_channels = doom_game.get_screen_channels()
        self.observation_shape = (self.history_window*num_channels,) + self.screen_size

        self.history = create_history_observation(self.history_window)
        # If episode is done WorkerProcess drops last returned state and
        #  returns the initial state of a new episode.
        # Therefore it doesn't really matter that terminal_screen is None
        self.terminal_obs = None

    def _define_actions(self, initialized_game):
        n_buttons = initialized_game.get_available_buttons_size()
        legal_actions = [
            list(a) for a in it.product([0, 1], repeat=n_buttons)
        ]
        # no-action/pass equals to the condition when no button is pressed
        noop = [0 for _ in range(n_buttons)]
        return legal_actions, noop

    def reset(self):
        """Starts a new episode and returns its initial state"""
        self.game.new_episode()
        # vars = self.game.get_state().game_variables
        obs = self.game.get_state().screen_buffer
        obs = self._preprocess(obs, self.screen_size)
        for _ in range(self.history_window):
            self.history.new_observation(obs)
        return self.history.get_state(), None


    def next(self, action):
        """
        Performs the given action.
        Returns the next state, reward, and terminal signal
        """
        reward = 0.
        if not self.game.is_episode_finished():
            action = self.legal_actions[action]
            reward = self.game.make_action(action, self.action_repeat)
            reward = reward*self.reward_coef
        is_done = self.game.is_episode_finished()
        if not is_done:
            next_screen = self.game.get_state().screen_buffer
            next_screen = self._preprocess(next_screen, self.screen_size)
            self.history.new_observation(next_screen)
            next_state = self.history.get_state()
        else:
            next_state = self.terminal_obs #is None right now
        return next_state, reward, is_done, None

    def get_legal_actions(self):
        return self.legal_actions

    def get_noop(self):
        return self.noop

    def close(self):
        self.game.close()