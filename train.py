import argparse
import logging
import os
import signal
import sys

import torch

import environment_creator
import utils
from networks.paac_nets import FFNetwork, LSTMNetwork
from paac import PAACLearner

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
FF_HISTORY_WINDOW=4
LSTM_HISTORY_WINDOW=1
ARGS_FILE='args.json'


def bool_arg(string):
    value = string.lower()
    if value == 'true':
        return True
    elif value == 'false':
        return False
    else:
        raise argparse.ArgumentTypeError("Expected True or False, but got {}".format(string))


def args_to_str(args):
    lines = ['','ARGUMENTS:']
    newline = os.linesep
    args = vars(args)
    for key in sorted(args.keys()):
        lines.append('    "{0}": {1}'.format(key, args[key]))
    return newline.join(lines)

def main(args):
    network_creator, env_creator = get_network_and_environment_creator(args)
    logging.info(args_to_str(args))
    logging.info('Initializing PAAC...')

    learner = PAACLearner(network_creator, env_creator, args)
    setup_kill_signal_handler(learner)

    logging.info('Starting training')
    learner.train()
    logging.info('Finished training')


def setup_kill_signal_handler(learner):
    main_process_pid = os.getpid()

    def signal_handler(signal, frame):
        if os.getpid() == main_process_pid:
            logging.info('Signal ' + str(signal) + ' detected, cleaning up.')
            learner.cleanup()
            logging.info('Cleanup completed, shutting down...')
            sys.exit(0)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)


def get_network_and_environment_creator(args, random_seed=None):
    if args.arch == 'lstm':
        args.history_window = LSTM_HISTORY_WINDOW
        Network = LSTMNetwork
    elif args.arch == 'ff':
        args.history_window = FF_HISTORY_WINDOW
        Network = FFNetwork

    if (not hasattr(args, 'random_seed')) or (random_seed is not None):
        args.random_seed = 3

    env_creator = environment_creator.EnvironmentCreator(args)
    args.num_actions = env_creator.num_actions
    num_actions = args.num_actions
    device = args.device

    def network_creator():
        if device == 'gpu':
            network = Network(num_actions, torch.cuda.FloatTensor, args.history_window)
            network = network.cuda()
            logging.debug("Moved network's computations on a GPU")
        else:
            network = Network(num_actions, torch.FloatTensor, args.history_window)
        return network

    return network_creator, env_creator


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', default='pong', help='Name of game', dest='game')
    parser.add_argument('-d', '--device', default='gpu', type=str, choices=['gpu', 'cpu'], help="Device to be used ('cpu' or 'gpu'). Use CUDA_VISIBLE_DEVICES to specify a particular gpu", dest="device")
    parser.add_argument('--rom_path', default='./atari_roms', help='Directory where the game roms are located (needed for ALE environment)', dest="rom_path")
    parser.add_argument('-v', '--visualize', default=False, type=bool_arg, help="0: no visualization of emulator; 1: all emulators, for all actors, are visualized; 2: only 1 emulator (for one of the actors) is visualized", dest="visualize")
    parser.add_argument('--e', default=0.1, type=float, help="Epsilon for the Rmsprop and Adam optimizers", dest="e")
    parser.add_argument('--alpha', default=0.99, type=float, help="Discount factor for the history/coming gradient, for the Rmsprop optimizer", dest="alpha")
    parser.add_argument('-lr', '--initial_lr', default=0.0224, type=float, help="Initial value for the learning rate. Default = 0.0224", dest="initial_lr")
    parser.add_argument('-lra', '--lr_annealing_steps', default=80000000, type=int, help="Nr. of global steps during which the learning rate will be linearly annealed towards zero", dest="lr_annealing_steps")
    parser.add_argument('--entropy', default=0.02, type=float, help="Strength of the entropy regularization term (needed for actor-critic)", dest="entropy_regularisation_strength")
    parser.add_argument('--clip_norm', default=3.0, type=float, help="If clip_norm_type is local/global, grads will be clipped at the specified maximum (avaerage) L2-norm", dest="clip_norm")
    parser.add_argument('--clip_norm_type', default="global", help="Whether to clip grads by their norm or not. Values: ignore (no clipping), local (layer-wise norm), global (global norm)", dest="clip_norm_type")
    parser.add_argument('--gamma', default=0.99, type=float, help="Discount factor", dest="gamma")
    parser.add_argument('--max_global_steps', default=80000000, type=int, help="Max. number of training steps", dest="max_global_steps")
    parser.add_argument('--max_local_steps', default=5, type=int, help="Number of steps to gain experience from before every update.", dest="max_local_steps")
    parser.add_argument('--single_life_episodes', default=False, type=bool_arg, help="If True, training episodes will be terminated when a life is lost (for games)", dest="single_life_episodes")
    parser.add_argument('-ec', '--emulator_counts', default=32, type=int, help="The amount of emulators per agent. Default is 32.", dest="emulator_counts")
    parser.add_argument('-ew', '--emulator_workers', default=8, type=int, help="The amount of emulator workers per agent. Default is 8.", dest="emulator_workers")
    parser.add_argument('-df', '--debugging_folder', default='logs/', type=str, help="Folder where to save the debugging information.", dest="debugging_folder")
    parser.add_argument('-rs', '--random_start', default=True, type=bool_arg, help="Whether or not to start with 30 noops for each env. Default True", dest="random_start")
    parser.add_argument('--arch', choices=['ff', 'lstm'], default='ff', help="Which network architecture to use: a feedforward network or an lstm network", dest="arch")
    parser.add_argument('--loss_scale', default=5., dest='loss_scaling', type=float, help='Scales loss according to a given value')
    parser.add_argument('--critic_coef', default=0.25, dest='critic_coef', type=float, help='Weight of the critic loss in the total loss')
    return parser


if __name__ == '__main__':
    args = get_arg_parser().parse_args()

    utils.save_args(args, args.debugging_folder, file_name=ARGS_FILE)
    logging.debug('Saved args in the {0} folder'.format(args.debugging_folder))

    main(args)
