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

    def __init__(self, emulator_id, args):
        if getattr(args, 'verbose', 0) > 2:
            logging.debug('Initializing Vizdoom.{}. emulator_id={}'.fromat(args.game, emulator_id))
        game = DoomGame()
        config_file_path = join_path(args.resource_folder, args.game+'.cfg')
        game.load_config(config_file_path)
        visualize = getattr(args, 'visualize', False)
        game.set_window_visible(visualize)
        game.set_screen_resolution(self.SCREEN_RESOLUTION)#640X420
        game.set_mode(self.MODE)
        if self.MODE == Mode.SPECTATOR:
            game.add_game_args("+freelook 1")

        # with a fixed seed all episodes in this environment will be identical
        #game.set_seed(args.random_seed)
        # game.add_available_game_variable(vizdoom.GameVariable.AMMO2)
        game.init()
        self.game = game
        self.legal_actions, self.noop = self._define_actions(self.game)

        self._preprocess = cv2_resize
        self.screen_size = getattr(args, 'screen_size', self.DEFAULT_SCREEN_SIZE)
        self.reward_coef = getattr(args, 'reward_coef', self.DEFAULT_REWARD_COEF)
        self.action_repeat = getattr(args, 'action_repeat', self.DEFAULT_ACTION_REPEAT)

        self.history_window = args.history_window
        num_channels = game.get_screen_channels()
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
            action = self.legal_actions[np.argmax(action)]
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