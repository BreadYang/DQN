"""Suggested Preprocessors."""

import numpy as np
from PIL import Image
from deeprl_hw2 import utils
from deeprl_hw2.core import Preprocessor, Sample


class HistoryPreprocessor(Preprocessor):
    """Keeps the last k states.

    Useful for domains where you need velocities, but the state
    contains only positions.

    When the environment starts, this will just fill the initial
    sequence values with zeros k times.

    Parameters
    ----------
    history_length: int
      Number of previous states to prepend to state being processed.

    """

    def __init__(self, config):
        self.history_length = config.history_length
        self.config = config
        self.state = np.zeros((config.history_length, config.screen_height, config.screen_width), dtype=np.float32)
        self.zero_screen = np.zeros((config.screen_height, config.screen_width), dtype=np.float32)
        self.index = 0

    def insert_screen(self, screen):
        """You only want history when you're deciding the current action to take. Give one screen, return one state"""
        self.state[self.index] = np.copy(screen)
        # print "\t insert screen at: %d" % (self.index)
        self.index = (self.index+1) % self.history_length
        return self.index

    def get_current_state(self):
        state = np.zeros((self.config.history_length, self.config.screen_height, self.config.screen_width), dtype=np.float32)
        for i in range(self.history_length):
            currentind = (self.index+1+i) % self.history_length
            state[i] = np.copy(self.state[currentind])
        return state

    def reset(self):
        self.state = np.zeros((self.config.history_length, self.config.screen_height, self.config.screen_width), dtype=np.float32)
        self.index = 0

    def get_config(self):
        return {'history_length': self.history_length}


class AtariPreprocessor(Preprocessor):
    """Converts images to greyscale and downscales.
    Based on the preprocessing step described in:

    @article{mnih15_human_level_contr_throug_deep_reinf_learn,
    author =	 {Volodymyr Mnih and Koray Kavukcuoglu and David
                  Silver and Andrei A. Rusu and Joel Veness and Marc
                  G. Bellemare and Alex Graves and Martin Riedmiller
                  and Andreas K. Fidjeland and Georg Ostrovski and
                  Stig Petersen and Charles Beattie and Amir Sadik and
                  Ioannis Antonoglou and Helen King and Dharshan
                  Kumaran and Daan Wierstra and Shane Legg and Demis
                  Hassabis},
    title =	 {Human-Level Control Through Deep Reinforcement
                  Learning},
    journal =	 {Nature},
    volume =	 518,
    number =	 7540,
    pages =	 {529-533},
    year =	 2015,
    doi =        {10.1038/nature14236},
    url =	 {http://dx.doi.org/10.1038/nature14236},
    }

    You may also want to max over frames to remove flickering. Some
    games require this (based on animations and the limited sprite
    drawing capabilities of the original Atari).

    Parameters
    ----------
    new_size: 2 element tuple
      The size that each image in the state should be scaled to. e.g
      (84, 84) will make each image in the output have shape (84, 84).
    """

    def __init__(self, config):
        self.history_length = config.history_length
        self.config = config
        self.newsize = (config.screen_width, config.screen_height)

    def process_state_for_memory(self, state): ## return copied
        """Scale, convert to greyscale and store as uint8.

        We don't want to save floating point numbers in the replay
        memory. We get the same resolution as uint8, but use a quarter
        to an eigth of the bytes (depending on float32 or float64)

        We recommend using the Python Image Library (PIL) to do the
        image conversions.
        """
        img = Image.fromarray(state).resize(self.newsize).convert('L')
        return np.array(img, dtype=np.uint8)

    def process_state_for_network(self, state): ## return copied
        """Scale, convert to greyscale and store as float32.
        Basically same as process state for memory, but this time
        outputs float32 images.
        """
        img = Image.fromarray(state).resize(self.newsize).convert('L')
        return np.array(img, dtype=np.float32)/255

    def process_batch(self, samples):
        """The batches from replay memory will be uint8, convert to float32.
        Same as process_state_for_network but works on a batch of
        samples from the replay memory. Meaning you need to convert
        both state and next state values.
        """
        processedlist = list()
        for S in samples:
            state = np.array(S.state, dtype=np.float32)/255
            next_state = np.array(S.next_state, dtype=np.float32)/255
            tup = Sample(state, S.action, S.reward, next_state, S.is_terminal)
            processedlist.append(tup)
        return processedlist

    def process_reward(self, reward):
        """Clip reward between -1 and 1."""
        return super(AtariPreprocessor, self).process_reward(reward)


# class PreprocessorSequence(Preprocessor):
#     """You may find it useful to stack multiple prepcrocesosrs (such as the History and the AtariPreprocessor).
#
#     You can easily do this by just having a class that calls each preprocessor in succession.
#
#     For example, if you call the process_state_for_network and you
#     have a sequence of AtariPreproccessor followed by
#     HistoryPreprocessor. This this class could implement a
#     process_state_for_network that does something like the following:
#
#     state = atari.process_state_for_network(state)
#     return history.process_state_for_network(state)
#     """
#     def __init__(self, preprocessors):
#
#         pass
