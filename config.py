__author__ = 'bread'
import datetime

class AgentConfig(object):
    #scale = 200
    scale = 10000
    iteration_num = 300 * scale
    memory_size = 100 * scale
    num_burn_in = 10 * scale
    batch_size = 32
    random_start = 30
    epsilon = 0.05
    gamma = 0.99
    target_q_update_step = 1 * scale
    learning_rate = 0.0001
    evaluation_interval = 10 * scale
    history_length = 4
    train_freq = 5
    eval_batch_num = 10
    decayNum = iteration_num / 2
    # learn_start = 5. * scale
    #
    # min_delta = -1
    # max_delta = 1
    #
    # double_q = False
    # dueling = False
    #
    # _test_step = 5 * scale
    # _save_step = _test_step * 10
    modelname = 'q_network'

class EnvironmentConfig(object):
    TimeStamp = datetime.datetime.strftime(datetime.datetime.now(), "%y-%m-%d_%H-%M")
    env_name = 'SpaceInvaders-v0'
    losslog = TimeStamp+"_V3"+'losslog_space_invader.csv'
    rewardlog = TimeStamp+"_V3"+'rewardlog_space_invader.csv'
    # env_name = 'Enduro-v0'
    history_length = 4
    screen_width = 84
    screen_height = 84
    max_reward = 1.
    min_reward = -1.


class DQNConfig(AgentConfig, EnvironmentConfig):
    model = ''
    pass


def get_config(FLAGS):
    config = DQNConfig
    return config
