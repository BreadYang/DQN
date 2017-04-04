"""Main DQN agent."""
#!/usr/bin/env python
"""Run Atari Environment with DQN."""
import argparse
import os
import random
import attr
import numpy as np
import datetime
import tensorflow as tf
from keras.layers import (Activation, Convolution2D, Dense, Flatten, Input,
                          Permute)
from keras.models import Model
from keras.optimizers import Adam
from deeprl_hw2 import utils
from deeprl_hw2.policy import UniformRandomPolicy, LinearDecayGreedyEpsilonPolicy, GreedyEpsilonPolicy
from deeprl_hw2.core import Sample, ReplayMemory
from deeprl_hw2.preprocessors import HistoryPreprocessor, AtariPreprocessor

import deeprl_hw2 as tfrl
from config import get_config

class DQNAgent:
    """Class implementing DQN.

    This is a basic outline of the functions/parameters you will need
    in order to implement the DQNAgnet. This is just to get you
    started. You may need to tweak the parameters, add new ones, etc.

    Feel free to change the functions and funciton parameters that the
    class provides.

    We have provided docstrings to go along with our suggested API.

    Parameters
    ----------
    q_network: keras.models.Model
      Your Q-network model.
    preprocessor: deeprl_hw2.core.Preprocessor
      The preprocessor class. See the associated classes for more
      details.
    memory: deeprl_hw2.core.Memory
      Your replay memory.
    gamma: float
      Discount factor.
    target_update_freq: float
      Frequency to update the target network. You can either provide a
      number representing a soft target update (see utils.py) or a
      hard target update (see utils.py and Atari paper.)
    num_burn_in: int
      Before you begin updating the Q-network your replay memory has
      to be filled up with some number of samples. This number says
      how many.
    train_freq: int
      How often you actually update your Q-Network. Sometimes
      stability is improved if you collect a couple samples for your
      replay memory, for every Q-network update that you run.
    batch_size: int
      How many samples in each minibatch.
    """
    def __init__(self,
                 q_network,
                 target_netwrok,
                 policy,
                 gamma,
                 num_burn_in,
                 train_freq,
                 batch_size, config):
        self.q = q_network
        self.q_target = target_netwrok
        self.memory = ReplayMemory(config)
        self.policy = policy
        self.gamma = gamma
        self.num_burn_in = num_burn_in
        self.train_freq = train_freq
        self.batch_size = batch_size
        self.currentIter = 0
        self.currentEps = 0
        self.currentReward = 0
        self.config = config
        #####
        self.historyPre = HistoryPreprocessor(config)
        self.AtariPre = AtariPreprocessor(config)
        pass

    def compile(self, optimizer, loss_func):
        """Setup all of the TF graph variables/ops.

        This is inspired by the compile method on the
        keras.models.Model class.

        This is a good place to create the target network, setup your
        loss function and any placeholders you might need.
        
        You should use the mean_huber_loss function as your
        loss_function. You can also experiment with MSE and other
        losses.

        The optimizer can be whatever class you want. We used the
        keras.optimizers.Optimizer class. Specifically the Adam
        optimizer.
        """
        pass

    def calc_q_values(self, state, network):
        """Given a state (or batch of states) calculate the Q-values.

        Basically run your network on these states.

        Return
        ------
        Q-values for the state(s)
        """
        state_pre = np.zeros((1, 4, 84, 84), dtype=np.float32)
        state_pre[0] = state
        q_values = network.predict(state_pre, batch_size=1)[0]
        return q_values

    def select_action(self, state, network, **kwargs):
        """Select the action based on the current state.

        You will probably want to vary your behavior here based on
        which stage of training your in. For example, if you're still
        collecting random samples you might want to use a
        UniformRandomPolicy.

        If you're testing, you might want to use a GreedyEpsilonPolicy
        with a low epsilon.

        If you're training, you might want to use the
        LinearDecayGreedyEpsilonPolicy.

        This would also be a good place to call
        process_state_for_network in your preprocessor.

        Returns
        --------
        selected action
        """
        state_pre = np.zeros((1, 4, 84, 84), dtype=np.float32)
        state_pre[0] = state
        q_values = network.predict(state_pre, batch_size=1)[0]
        return self.policy.select_action(q_values)

    def fit(self, env, num_iterations, max_episode_length=None):
        """Fit your model to the provided environment.

        Its a good idea to print out things like loss, average reward,
        Q-values, etc to see if your agent is actually improving.

        You should probably also periodically save your network
        weights and any other useful info.

        This is where you should sample actions from your network,
        collect experience samples and add them to your replay memory,
        and update your network parameters.

        Parameters
        ----------
        env: gym.Env
          This is your Atari environment. You should wrap the
          environment using the wrap_atari_env function in the
          utils.py
        num_iterations: int
          How many samples/updates to perform.
        max_episode_length: int
          How long a single episode should last before the agent
          resets. Can help exploration.
        """
        cnt = np.long(0)
        episode_rwd = 0
        _screen_raw = self.process_env_reset(env)  # Save to history
        mse_loss, mae_metric = 0, 0
        self.policy = UniformRandomPolicy(env.action_space.n)
        evaluation_interval_cnt = 0
        while cnt < num_iterations:
            cnt += 1
            evaluation_interval_cnt += 1
            current_state = self.historyPre.get_current_state()
            action = self.select_action(current_state, self.q)  # Get action
            _screen_next_raw, reward, isterminal, _ = env.step(action)   # take action, observe new
            episode_rwd += reward
            _screen_raw = self.process_one_screen(_screen_raw, action, reward, _screen_next_raw, isterminal, True)  # Save to history, Memory
            # print "\t state: %d, Step: %d, reward: %d, terminal: %d, Observe: %d" \
            #       % (np.matrix(_screen).sum(), action, reward, isterminal, np.matrix(_screen_next).sum())
            # env.render()

            if isterminal:     # reset
                if evaluation_interval_cnt >= self.config.evaluation_interval:
                    Aver_reward = self.evaluate(env, self.config.eval_batch_num)
                    # print ("----------Evaluate, Average reward", Aver_reward)
                    evaluation_interval_cnt = 0
                    with open(self.config.rewardlog, "a") as log:
                        log.write(",".join([str(int(cnt/self.config.evaluation_interval)), str(Aver_reward)]) + "\n")
                _screen_raw = self.process_env_reset(env)
                # print ("Episode End, iter: ", cnt, "last batch loss: ", mse_loss, 'last mae Metric: ', mae_metric, "Episode reward: ", episode_rwd)
                episode_rwd = 0

            if cnt >= self.num_burn_in and cnt % self.train_freq == 0:          # update
                samples = self.AtariPre.process_batch(self.memory.sample(self.batch_size))
                x = np.zeros((self.batch_size, self.config.history_length, self.config.screen_height, self.config.screen_width), dtype=np.float32)
                y = np.zeros((self.batch_size, int(action_size(env))), dtype=np.float32)
                for _index in range(len(samples)):
                    sample = samples[_index]
                    x[_index] = np.copy(sample.state)
                    if sample.is_terminal:
                        y[_index] = self.calc_q_values(sample.state, self.q)
                        y[_index][sample.action] = sample.reward
                    else:
                        y[_index] = self.calc_q_values(sample.state, self.q)
                        q_next = max(self.calc_q_values(sample.next_state, self.q_target))   # Use max to update
                        y[_index][sample.action] = sample.reward + self.gamma*q_next

                mse_loss, mae_metric = self.q.train_on_batch(x, y)
                with open(self.config.losslog, "a") as log:
                    log.write(",".join([str(cnt/4), str(mse_loss), str(mae_metric)]) + "\n")
                # print(cnt, mse_loss, mae_metric)

            if cnt % self.config.target_q_update_step == 0:  # Set q == q^
                self.q_target.set_weights(self.q.get_weights())
            if cnt == self.config.memory_size:    # change Policy
                self.policy = LinearDecayGreedyEpsilonPolicy(1, 0.05, self.config.decayNum)

            if cnt % (num_iterations/3) == 0:  # Save model
                TimeStamp = datetime.datetime.strftime(datetime.datetime.now(), "%y-%m-%d_%H-%M")
                self.q.save_weights(str(self.config.modelname) + '_' + TimeStamp+'_weights.h5')
        return mse_loss, mae_metric, self.q, self.q_target

    def process_one_screen(self, screen_raw, action, reward, screen_next_raw, isterminal, Is_train):
        screen_32_next = self.AtariPre.process_state_for_network(screen_next_raw)
        screen_8 = self.AtariPre.process_state_for_memory(screen_raw)
        self.historyPre.insert_screen(screen_32_next)
        if Is_train:
            self.memory.append(screen_8, action, reward, isterminal)
        return screen_next_raw

    def process_env_reset(self, env):
        self.historyPre.reset()
        screen_raw = env.reset()
        screen_32 = self.AtariPre.process_state_for_network(screen_raw)
        self.historyPre.insert_screen(screen_32)
        return screen_raw

    def evaluate(self, env, num_episodes):
        """Test your agent with a provided environment.

        You shouldn't update your network parameters here. Also if you
        have any layers that vary in behavior between train/test time
        (such as dropout or batch norm), you should set them to test.

        Basically run your policy on the environment and collect stats
        like cumulative reward, average episode length, etc.

        You can also call the render function here if you want to
        visually inspect your policy.
        """
        eval_policy = GreedyEpsilonPolicy(self.config.epsilon)
        cumu_reward = 0
        epscnt = 0
        while epscnt < num_episodes:
            isterminal = False
            _screen_raw = self.process_env_reset(env)  # Save to history
            while not isterminal:
                current_state = self.historyPre.get_current_state()
                action = self.select_action_test(current_state, eval_policy)  # Get action
                _screen_next_raw, reward, isterminal, _ = env.step(action)   # take action, observe new
                cumu_reward += reward
                _screen_raw = self.process_one_screen(_screen_raw, action, reward, _screen_next_raw, isterminal, True)  # Save to history, Memory
            epscnt += 1
        return cumu_reward/num_episodes

    def select_action_test(self, state, policy, **kwargs):
        """Select the action based on the current state.

        You will probably want to vary your behavior here based on
        which stage of training your in. For example, if you're still
        collecting random samples you might want to use a
        UniformRandomPolicy.

        If you're testing, you might want to use a GreedyEpsilonPolicy
        with a low epsilon.

        If you're training, you might want to use the
        LinearDecayGreedyEpsilonPolicy.

        This would also be a good place to call
        process_state_for_network in your preprocessor.

        Returns
        --------
        selected action
        """
        state_pre = np.zeros((1, 4, 84, 84), dtype=np.float32)
        state_pre[0] = state
        q_values = self.q.predict(state_pre, batch_size=1)[0]
        return policy.select_action(q_values)

def action_size(env):
    return env.action_space.n


