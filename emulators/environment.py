import numpy as np
from collections import deque

class BaseEnvironment(object):
    def reset(self):
        """
        Sets the environment to its initial state.
        :return: the initial state
        """
        raise NotImplementedError()

    def next(self, action):
        """
        Appies the current action to the environment.
        :param action: one hot vector.
        :return: (observation, reward, is_terminal) tuple
        """
        raise NotImplementedError()

    def get_legal_actions(self):
        """
        Get the list of indices of legal actions
        :return: a numpy array of the indices of legal actions
        """
        raise NotImplementedError()

    def get_noop(self):
        """
        Gets the no-op action, to be used with self.next
        :return: the action
        """
        raise NotImplementedError()


    def close(self):
        """
        If any necessary cleanup is needed, do it in this method
        :return:
        """
        pass


def create_history_observation(history_window):
    if history_window == 1:
        return DummyObservationHistory()
    else:
        return ObservationHistory(history_window)


class ObservationHistory(object):
    def __init__(self, history_window):
        self.history = deque(maxlen=history_window)

    def new_observation(self, observation):
        """Receives an observation of shape (num_channels, height, width)"""
        self.history.append(observation)

    def get_state(self):
        return np.concatenate(self.history, axis=0)


class DummyObservationHistory(ObservationHistory):
    def __init__(self):
        self.observation = None

    def new_observation(self, observation):
        self.observation = observation

    def get_state(self):
        return self.observation