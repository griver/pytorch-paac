import random

import numpy as np
from ale_python_interface import ALEInterface
import cv2

from ..environment import BaseEnvironment, create_history_observation
import logging

IMG_SIZE_X = 84
IMG_SIZE_Y = 84
ACTION_REPEAT = 4
MAX_START_WAIT = 30
FRAMES_IN_POOL = 2

ATARY_GAMES = None #check all roms

class FramePreprocessor(object):
    """
    Processes several consecutive atari frames.
    Call new_frame on every new atari frame(without frame-skipping).
    Call get_processed to return a singe observation or to add it the history window
    """
    def __init__(self, screen_shape, frame_pool_size):
        self.frame_pool = np.empty((frame_pool_size,) + screen_shape)
        self.frame_pool_index = 0
        self.frames_in_pool = self.frame_pool.shape[0]
        if screen_shape[-1] > 1: #for some reason cv2 squeezes the channel dimension for grayscale images
            self.reshape = lambda x: np.transpose(x, (2,0,1)) #from (H,W,C) to (C,H,W)
        else:
            self.reshape = lambda x: x[np.newaxis,:,:] #from (H,W) to (C=1, H, W)

    def new_frame(self, frame):
        self.frame_pool[self.frame_pool_index] = frame
        self.frame_pool_index = (self.frame_pool_index + 1) % self.frames_in_pool

    def get_processed(self):
        img = np.amax(self.frame_pool, axis=0) #max-pooling across last 'frame_pool_size' time-steps
        img = cv2.resize(img, (84, 84), interpolation=cv2.INTER_AREA)
        img = self.reshape(img.astype(np.uint8))
        return img


class AtariEmulator(BaseEnvironment):
    def __init__(self, emulator_id, game, resource_folder, random_seed=3,
                 random_start=True, single_life_episodes=False,
                 history_window=1, visualize=False, verbose=0, **unknown):
        if verbose >= 2:
            logging.debug('Emulator#{} received unknown args: {}'.format(emulator_id, unknown))
        self.emulator_id = emulator_id
        self.ale = ALEInterface()
        self.ale.setInt(b"random_seed", random_seed * (emulator_id + 1))
        # For fuller control on explicit action repeat (>= ALE 0.5.0)
        self.ale.setFloat(b"repeat_action_probability", 0.0)
        # Disable frame_skip and color_averaging
        # See: http://is.gd/tYzVpj
        self.ale.setInt(b"frame_skip", 1)
        self.ale.setBool(b"color_averaging", False)
        self.ale.setBool(b"display_screen", visualize)

        full_rom_path = resource_folder + "/" + game + ".bin"
        self.ale.loadROM(str.encode(full_rom_path))
        self.legal_actions = self.ale.getMinimalActionSet()
        self.screen_width, self.screen_height = self.ale.getScreenDims()
        self.lives = self.ale.lives()

        self.random_start = random_start
        self.single_life_episodes = single_life_episodes
        self.call_on_new_frame = visualize
        self.history_window = history_window
        self.observation_shape = (self.history_window, IMG_SIZE_X, IMG_SIZE_Y)
        self.rgb_screen = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
        self.gray_screen = np.zeros((self.screen_height, self.screen_width, 1), dtype=np.uint8)
        # Processed historcal frames that will be fed in to the network (i.e., four 84x84 images)
        self.history = create_history_observation(self.history_window)
        #ObservationPool(np.zeros(self.observation_shape, dtype=np.uint8))
        self.frame_preprocessor = FramePreprocessor(self.gray_screen.shape, FRAMES_IN_POOL)

    def get_legal_actions(self):
        return self.legal_actions

    def __get_screen_image(self):
        """
        Get the current frame luminance
        :return: the current frame
        """
        self.ale.getScreenGrayscale(self.gray_screen)
        if self.call_on_new_frame:
            self.ale.getScreenRGB(self.rgb_screen)
            self.on_new_frame(self.rgb_screen)
        return self.gray_screen

    def on_new_frame(self, frame):
        pass

    def __new_game(self):
        """ Restart game """
        self.ale.reset_game()
        self.lives = self.ale.lives()
        if self.random_start:
            wait = random.randint(0, MAX_START_WAIT)
            for _ in range(wait):
                self.ale.act(self.get_noop())


    def __action_repeat(self, a, times=ACTION_REPEAT):
        """ Repeat action and grab screen into frame pool """
        reward = 0
        for i in range(times - FRAMES_IN_POOL):
            reward += self.ale.act(self.legal_actions[a])
        # Only need to add the last FRAMES_IN_POOL frames to the frame pool
        for i in range(FRAMES_IN_POOL):
            reward += self.ale.act(self.legal_actions[a])
            self.frame_preprocessor.new_frame(self.__get_screen_image())
        return reward

    def reset(self):
        """ Get the initial state """
        self.__new_game()
        for step in range(self.history_window):
            _ = self.__action_repeat(0)
            self.history.new_observation(self.frame_preprocessor.get_processed())
        if self.__is_terminal():
            raise Exception('This should never happen.')
        return self.history.get_state(), None

    def next(self, action):
        """ Get the next state, reward, and game over signal """
        reward = self.__action_repeat(np.argmax(action))
        self.history.new_observation(self.frame_preprocessor.get_processed())
        terminal = self.__is_terminal()
        self.lives = self.ale.lives()
        return self.history.get_state(), reward, terminal, None

    def __is_terminal(self):
        if self.single_life_episodes:
            return self.__is_over() or (self.lives > self.ale.lives())
        else:
            return self.__is_over()

    def __is_over(self):
        return self.ale.game_over()

    def get_noop(self):
        return self.legal_actions[0]

    def close(self):
        del self.ale