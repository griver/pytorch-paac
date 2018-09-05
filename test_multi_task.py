import argparse
import itertools
import time

import numpy as np
import torch
import torch.nn.functional as F

import utils
from utils import eval_taxi as eval
from utils import report_taxi_evaluations as report_eval
import train_multi_task as train
import logging


#eval_mode = dict(
#    stats=train.eval_network,
#    visual=eval.visual_eval,
#    interactive=eval.interactive_eval,
#    custom=report_eval.custom_task_eval,
#    fixed=report_eval.fixed_episode_eval,
#)


def handle_commandline(command_line=None):
    eval_modes = ['stats', 'visual']#list(eval_mode.keys())
    default_mode = 'stats'
    devices = ['cuda', 'cpu'] if torch.cuda.is_available() else ['cpu']
    available_games = train.TaxiGamesCreator.available_games()
    parser = get_argparser(eval_modes, default_mode, devices, available_games)
    args = parser.parse_args(command_line.split()) if command_line else parser.parse_args()

    logging.info("Loading training config from {}".format(args.folder))
    train_args = utils.load_args(folder=args.folder, file_name=train.ARGS_FILE)
    args = fix_args_for_test(args, train_args)

    return args


def get_argparser(eval_modes, default_mode, devices, available_games):
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=str, help="Folder with a trained model.")
    parser.add_argument('-tc', '--test_count', default=1, type=int, dest="test_count",
                        help="Number of episodes to test the model")
    parser.add_argument('-g', '--greedy', action='store_true',
                        help='Determines whether to use a stochastic or deterministic policy')
    parser.add_argument('-d', '--device', default=devices[0], type=str, choices=devices, dest="device", help="Device to be used ('cpu' or 'cuda'). Use CUDA_VISIBLE_DEVICES to specify a particular GPU")
    parser.add_argument('-v', '--visualize', action='store_true')
    #parser.add_argument('-m', '--mode', type=str, default=default_mode, choices=eval_modes, help='A evaluation type')

    parser.add_argument('-tt', '--termination_threshold', default=None, type=float_or_none, dest='termination_threshold', help='Real value between [0.,1.] or None.', )

    parser.add_argument('--map_size', nargs=4, type=int, default=None, dest='test_map_size',
                        help='The size of environment of shape (min_x, max_x, min_y, max_y)'
                             'At the beggining of a new episode the size of the new environment '
                             'will be drawn uniformly from the specified range. If map_size is not' 
                             'given  then parameters from training config are used.')
    parser.add_argument('-tg', '--test_game', type=str, default=None, choices=available_games,
                        help='if test_game is not given, then the game from the training config is used')
    parser.add_argument('--single_task_episodes', action='store_true',
                           help="if provided each episode equals one subtask")
    return parser


def float_or_none(arg_str):
    if arg_str.lower() == "none":
        return None
    else:
        return float(arg_str)


def fix_args_for_test(args, train_args):
    if args.test_map_size is not None:
        args.map_size = args.test_map_size
        delattr(args, 'test_map_size')

    if args.test_game is not None:
        args.game = args.test_game
        delattr(args, 'test_game')

    for k, v in train_args.items():
        if getattr(args, k, None) == None:
            setattr(args, k, v)
            
    args.max_global_steps = 0
    args.initial_lr = 0.
    args.random_seed = np.random.randint(1000)
    return args


def load_trained_weights(network, checkpoint_path, use_cpu):
    if use_cpu:
        #it avoids loading cuda tensors in case a gpu is unavailable
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    else:
        checkpoint = torch.load(checkpoint_path)
    network.load_state_dict(checkpoint['network_state_dict'])
    return checkpoint['last_step']


if __name__=='__main__':
    args = handle_commandline()

    checkpoint_path = utils.join_path(
        args.folder,
        train.MultiTaskActorCritic.CHECKPOINT_SUBDIR,
        train.MultiTaskActorCritic.CHECKPOINT_LAST
    )
    env_creator = train.TaxiGamesCreator(**vars(args))
    network = train.create_network(args, env_creator.num_actions, env_creator.obs_shape)
    steps_trained = load_trained_weights(network, checkpoint_path, args.device == 'cpu')

    print(train.args_to_str(args))

    print('Model was trained for {} steps'.format(steps_trained))
    #evaluate = eval_mode[args.mode]

    if args.visualize:
        num_steps, rewards, extra_stats = eval.visual_eval(
            network, env_creator,
            args.test_count, args.greedy,
            args.termination_threshold
        )

        print('Perfromed {0} tests for {1}.'.format(args.test_count, args.game))
        print('Mean number of steps: {0:.3f}'.format(np.mean(num_steps)))
        print('Mean R: {0:.2f}'.format(np.mean(rewards)), end=' | ')
        print('Max R: {0:.2f}'.format(np.max(rewards)), end=' | ')
        print('Min R: {0:.2f}'.format(np.min(rewards)), end=' | ')
        print('Std of R: {0:.2f}'.format(np.std(rewards)))

        if extra_stats is not None:
            if hasattr(extra_stats, 'pretty_print'):
                extra_stats.pretty_print()
            else:
                print(extra_stats)
    else:
        train.eval_network(
            network, env_creator,
            args.test_count, args.greedy,
            args.termination_threshold, args.verbose)