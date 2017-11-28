import argparse
import logging
import os
import sys

import torch

import utils
import utils.eval_multi_task as evaluate

from emulators import TaxiEmulator
from multi_task_paac import MultiTaskPAAC
from networks import MultiTaskFFNetwork, MultiTaskLSTMNetwork, preprocess_taxi_input
from train import bool_arg, args_to_str, setup_kill_signal_handler

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
ARGS_FILE = 'args_multi_task.json'
VIEW_SIZE = (5,5)
MAP_SIZE = (5,5,5,5)

class MultiTaskEnvironmentCreator(object):
    def __init__(self, args):
        self.create_environment = lambda i: TaxiEmulator(i, args)
        test_env = self.create_environment(-1)
        self.num_actions = len(test_env.legal_actions)
        self.obs_shape = test_env.observation_shape
        self.preprocess_states = TaxiEmulator.split_observation_and_task_info


def view_size(args):
    min_x, max_x, min_y, max_y = args.map_size
    if args.full_view:
        return (max_x, max_y)
    else:
        x, y = VIEW_SIZE
        return (min(x, min_x), min(y, min_y))


def get_network_and_environment_creator(args, random_seed=None):
    if (not hasattr(args, 'random_seed')) or (random_seed is not None):
        args.random_seed = 3
    args.view_size = view_size(args)

    env_creator = MultiTaskEnvironmentCreator(args)
    args.obs_shape = env_creator.obs_shape
    args.num_actions = env_creator.num_actions

    device = args.device

    if args.arch == 'lstm':
        Network = MultiTaskLSTMNetwork
    elif args.arch == 'ff':
        Network = MultiTaskFFNetwork

    def network_creator():
        if device == 'gpu':
            network = Network(
              args.num_actions, args.obs_shape, torch.cuda,
              preprocess=preprocess_taxi_input
            )
            network = network.cuda()
            logging.debug("Moved network's computations on a GPU")
        else:
            network = Network(
              args.num_actions, args.obs_shape, torch,
              preprocess=preprocess_taxi_input
            )
        return network

    return network_creator, env_creator


def main(args):
    network_creator, env_creator = get_network_and_environment_creator(args)
    logging.info(args_to_str(args))
    logging.info('Initializing PAAC...')

    learner = MultiTaskPAAC(network_creator, env_creator, args)


    learner.set_eval_function(
        eval_func=evaluate.stats_eval,
        args = [learner.network, env_creator],
        kwargs = dict(test_count=50, is_recurrent=learner.use_lstm) # default termination_threshold=0.5
    )

    setup_kill_signal_handler(learner)

    logging.info('Starting training')
    learner.train()
    logging.info('Finished training')



def get_arg_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument('-g', default='taxi_multi_task', choices=['taxi_multi_task', 'taxi_game'], help='Name of game', dest='game')
  parser.add_argument('-d', '--device', default='gpu', type=str, choices=['gpu', 'cpu'],
                      help="Device to be used ('cpu' or 'gpu'). Use CUDA_VISIBLE_DEVICES to specify a particular gpu",
                      dest="device")
  parser.add_argument('-m', '--map_size', nargs=4, type=int, default=[5,5,5,5],
                      help='The size of environment of shape (min_x, max_x, min_y, max_y). At the beggining of a new episode size (x,y) of a new environment will be drawn uniformly from it')
  parser.add_argument('-f', '--full_view', action='store_true', help='If the flag is provided then an agent will receive a full map view as an observation.')
  parser.add_argument('-v', '--verbose', default=1, type=int,
                      help="0: no visualization of emulator; 1: all emulators, for all actors, are visualized; 2: only 1 emulator (for one of the actors) is visualized",
                      dest="verbose")
  parser.add_argument('--e', default=0.1, type=float, help="Epsilon for the Rmsprop and Adam optimizers", dest="e")
  parser.add_argument('--alpha', default=0.99, type=float,
                      help="Discount factor for the history/coming gradient, for the Rmsprop optimizer", dest="alpha")
  parser.add_argument('-lr', '--initial_lr', default=0.0224, type=float,
                      help="Initial value for the learning rate. Default = 0.0224", dest="initial_lr")
  parser.add_argument('-lra', '--lr_annealing_steps', default=80000000, type=int,
                      help="Nr. of global steps during which the learning rate will be linearly annealed towards zero",
                      dest="lr_annealing_steps")
  parser.add_argument('--entropy', default=0.02, type=float,
                      help="Strength of the entropy regularization term (needed for actor-critic)",
                      dest="entropy_regularisation_strength")
  parser.add_argument('--clip_norm', default=3.0, type=float,
                      help="If clip_norm_type is local/global, grads will be clipped at the specified maximum (avaerage) L2-norm",
                      dest="clip_norm")
  parser.add_argument('--clip_norm_type', default="global",
                      help="Whether to clip grads by their norm or not. Values: ignore (no clipping), local (layer-wise norm), global (global norm)",
                      dest="clip_norm_type")
  parser.add_argument('--gamma', default=0.99, type=float, help="Discount factor", dest="gamma")
  parser.add_argument('--max_global_steps', default=80000000, type=int, help="Max. number of training steps",
                      dest="max_global_steps")
  parser.add_argument('--max_local_steps', default=5, type=int,
                      help="Number of steps to gain experience from before every update.", dest="max_local_steps")
  parser.add_argument('-ec', '--emulator_counts', default=32, type=int,
                      help="The amount of emulators per agent. Default is 32.", dest="emulator_counts")
  parser.add_argument('-ew', '--emulator_workers', default=8, type=int,
                      help="The amount of emulator workers per agent. Default is 8.", dest="emulator_workers")
  parser.add_argument('-df', '--debugging_folder', default='logs/', type=str,
                      help="Folder where to save the debugging information.", dest="debugging_folder")
  parser.add_argument('--arch', choices=['ff', 'lstm'], default='ff',
                      help="Which network architecture to use: a feedforward network or an lstm network", dest="arch")
  parser.add_argument('--loss_scale', default=5., dest='loss_scaling', type=float,
                      help='Scales loss according to a given value')
  parser.add_argument('--critic_coef', default=0.25, dest='critic_coef', type=float,
                      help='Weight of the critic loss in the total loss')
  parser.add_argument('-tmc --termination_model_coef', default=0., dest='termination_model_coef', type=float,
                      help='Weight of the termination model loss in the total loss')
  parser.add_argument('--eval_every', default=102400, type=int, dest='eval_every',
                      help='Model evaluation frequency.')
  return parser


if __name__ == '__main__':
    #args_line = '-lr 0.0075 --e 0.033 -d cpu -ew 4 -ec 32 -f -df logs_taxi --max_global_steps 200000 --max_local_steps 10 --arch lstm'
    args = get_arg_parser().parse_args()

    utils.save_args(args, args.debugging_folder, file_name=ARGS_FILE)
    logging.debug('Saved args in the {0} folder'.format(args.debugging_folder))
    main(args)

