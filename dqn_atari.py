#!/usr/bin/env python
"""Run Atari Environment with DQN."""
import argparse
import os
import random
import gym
import datetime
import numpy as np
import tensorflow as tf
from keras.layers import (Input, Activation, Convolution2D, Dense, Flatten, Input, Permute, Conv2D, Lambda)
from keras.models import Model
from keras.optimizers import Adam
import keras.backend as K

import deeprl_hw2 as tfrl
from deeprl_hw2.dqn import DQNAgent
from deeprl_hw2.doubleDQN import DoubleDQNAgent
from deeprl_hw2.policy import GreedyEpsilonPolicy, LinearDecayGreedyEpsilonPolicy
from deeprl_hw2.objectives import mean_huber_loss
from config import get_config
import sys


def create_model(window, input_shape, num_actions, model_name):
    """Create the Q-network model.

    Use Keras to construct a keras.models.Model instance (you can also
    use the SequentialModel class).

    We highly recommend that you use tf.name_scope as discussed in
    class when creating the model and the layers. This will make it
    far easier to understand your network architecture if you are
    logging with tensorboard.

    Parameters
    ----------
    window: int
      Each input to the network is a sequence of frames. This value
      defines how many frames are in the sequence.
    input_shape: tuple(int, int)
      The expected input image size.
    num_actions: int
      Number of possible actions. Defined by the gym environment.
    model_name: str
      Useful when debugging. Makes the model show up nicer in tensorboard.

    Returns
    -------
    keras.models.Model
      The Q-model.
    """

    input_shape = (window, input_shape[0], input_shape[1])
    input = Input(shape=(input_shape), name='input', dtype='float')
    if model_name == 'linear':
        with tf.name_scope('flatten'):
            flatten = Flatten()(input)
        with tf.name_scope('Q'):
            output = Dense(num_actions, activation='linear', name='Q')(flatten)

    elif model_name in ['q_network', 'dq_network', 'duel_network']:
        # Q layers
        with tf.name_scope('hidden'):
            hidden1 = Conv2D(16, 8, strides=4, activation='relu', input_shape=input_shape,
                             data_format='channels_first', name='h1')(input)
            hidden2 = Conv2D(32, 4, strides=2, activation='relu', name='h2')(hidden1)
            flatten = Flatten()(hidden2)
            hidden4 = Dense(256, activation='relu', name='h4')(flatten)
        # average
        if model_name == 'duel_network':
            y = Dense(num_actions+1, activation='linear')(hidden4)
            with tf.name_scope('Q'):
                output = Lambda(lambda a: K.expand_dims(a[:, 0], axis=-1) + a[:, 1:] - K.mean(a[:, 1:], keepdims=True),
                                output_shape=(num_actions,))(y)
        elif model_name in ['q_network', 'dq_network']:
            with tf.name_scope('Q'):
                output = Dense(num_actions, activation='linear', name='Q')(hidden4)
    else:
        sys.exit("Network not defined")
    model = Model(input=input, output=output)
    return model


def get_output_folder(parent_dir, env_name):
    """Return save folder.

    Assumes folders in the parent_dir have suffix -run{run
    number}. Finds the highest run number and sets the output folder
    to that number + 1. This is just convenient so that if you run the
    same script multiple times tensorboard can plot all of the results
    on the same plots with different names.

    Parameters
    ----------
    parent_dir: str
      Path of the directory containing all experiment runs.

    Returns
    -------
    parent_dir/run_dir
      Path to this run's save directory.
    """
    os.makedirs(parent_dir, exist_ok=True)
    experiment_id = 0
    for folder_name in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, folder_name)):
            continue
        try:
            folder_name = int(folder_name.split('-run')[-1])
            if folder_name > experiment_id:
                experiment_id = folder_name
        except:
            pass
    experiment_id += 1

    parent_dir = os.path.join(parent_dir, env_name)
    parent_dir = parent_dir + '-run{}'.format(experiment_id)
    return parent_dir


def main():  # noqa: D103
    config = get_config(True)
    env = gym.make(config.env_name)
    q = create_model(4, (84, 84), env.action_space.n, model_name=config.modelname)
    q_target = create_model(4, (84, 84), env.action_space.n, model_name=config.modelname)
    huber_loss = tfrl.objectives.mean_huber_loss
    adam = Adam(lr = config.learning_rate)
    q.compile(adam, huber_loss, metrics=['accuracy'])
    q_target.compile(adam, huber_loss, metrics=['accuracy'])
    policy = LinearDecayGreedyEpsilonPolicy(0.9, 0.05, config.iteration_num/50)   # Deprecated
    with open(config.losslog, "w") as log:
        log.write("Iteraton,Loss,Accuarcy\n")
    with open(config.rewardlog, "w") as log:
        log.write("Iteraton,reward\n")
    #####
    #Agent = DoubleDQNAgent(q, q_target, policy, config.gamma, config.num_burn_in, config.train_freq, config.batch_size, config)

    Agent = DQNAgent(q, q_target, policy, config.gamma, config.num_burn_in, config.train_freq, config.batch_size, config)

    mse_loss, mae_metric, q, q_target = Agent.fit(env, config.iteration_num, 0)
    TimeStamp = datetime.datetime.strftime(datetime.datetime.now(), "%y-%m-%d_%H-%M")

    q.save_weights(str(config.modelname) + '_' + TimeStamp+'_final_weights.h5')


    # here is where you should start up a session,
    # create your DQN agent, create your model, etc.
    # then you can run your fit method.



if __name__ == '__main__':
    main()