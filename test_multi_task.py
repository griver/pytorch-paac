import argparse
import itertools
import time

import numpy as np
import torch
import torch.nn.functional as F

import utils
from utils import eval_multi_task as eval
import train_multi_task as train


def print_dict(d, name=None):
    title = ' '.join(['=='*10, '{}','=='*10])
    if name is not None:
        title.format(name)

    print(title)
    for k in sorted(d.keys()):
        print('  ', k,':', d[k])
    print('='*len(title))


def float_or_none(arg_str):
    if arg_str.lower() == "none":
        return None
    else:
        return float(arg_str)



def fix_args_for_test(args, train_args):
    if args.test_map_size is not None:
        args.map_size = args.test_map_size

    for k, v in train_args.items():
        if not hasattr(args, k):
            setattr(args, k, v)

    args.max_global_steps = 0
    args.debugging_folder = '/tmp/logs'
    args.verbose = 1
    return args

eval_mode = dict(
    stats=eval.stats_eval,
    visual=eval.visual_eval,
    interactive=eval.interactive_eval
)



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--folder', type=str, help="Folder where to save the debugging information.", dest="folder", required=True)
    parser.add_argument('-tc', '--test_count', default='1', type=int, help="The amount of tests to run on the given network", dest="test_count")
    parser.add_argument('-g', '--greedy', action='store_true', help='Determines whether to use a stochastic or deterministic policy')
    parser.add_argument('-m', '--mode', type=str, default='stats',
                        choices=['stats', 'visual', 'interactive'],
                        help='In ineractive mode a user can set tasks to the learned model')
    parser.add_argument('-d', '--device', default='gpu', type=str, choices=['gpu', 'cpu'],
        help="Device to be used ('cpu' or 'gpu'). Use CUDA_VISIBLE_DEVICES to specify a particular gpu", dest="device")
    parser.add_argument('-tt', '--termination_threshold', default=None, type=float_or_none,
                        help='A real value between [0.,1.] or None.', dest='termination_threshold')
    parser.add_argument('--map_size', nargs=4, type=int, default=None, dest='test_map_size',
                        help='The size of environment of shape (min_x, max_x, min_y, max_y). ' +
                             'At the beggining of a new episode size (x,y) of a new environment ' +
                             'will be drawn uniformly from it. If map_size is not given for the ' +
                             ' script then a value from the training config is used.')

    args = parser.parse_args()
    train_args = utils.load_args(folder=args.folder, file_name=train.ARGS_FILE)
    #train_args = utils.load_args(folder='pretrained/multi_task_lstm', file_name=train.ARGS_FILE)
    args = fix_args_for_test(args, train_args)

    checkpoint_path = utils.join_path(
        args.folder, train.MultiTaskPAAC.CHECKPOINT_SUBDIR, train.MultiTaskPAAC.CHECKPOINT_LAST
    )
    checkpoint = torch.load(checkpoint_path)
    net_creator, env_creator = train.get_network_and_environment_creator(args)
    network = net_creator()
    network.load_state_dict(checkpoint['network_state_dict'])
    network.eval()

    use_lstm = ('ff' not in args.arch)
    print_dict(vars(args), 'ARGS')
    print('Model was trained for {} steps'.format(checkpoint['last_step']))
    evaluate = eval_mode[args.mode]

    num_steps, rewards, prediction_stats = evaluate(
            network, env_creator, args.test_count,
            greedy=args.greedy, is_recurrent=use_lstm,
            termination_threshold=args.termination_threshold
        )

    print('Perfromed {0} tests for {1}.'.format(args.test_count, args.game))
    print('Mean number of steps: {0:.3f}'.format(np.mean(num_steps)))
    print('Mean R: {0:.2f}'.format(np.mean(rewards)), end=' | ')
    print('Max R: {0:.2f}'.format(np.max(rewards)), end=' | ')
    print('Min R: {0:.2f}'.format(np.min(rewards)), end=' | ')
    print('Std of R: {0:.2f}'.format(np.std(rewards)))

    if prediction_stats:
        print('Test Termination Predictor:')
        print('Number of samples:', prediction_stats.size)
        print('Accuracy: {0:.2f}%'.format(prediction_stats.accuracy))
        print('Precision: {0:.2f}%'.format( prediction_stats.precision))
        print('Recall: {0:.2f}%'.format(prediction_stats.recall))
        print('targets_ratio: {0:.2f}%'.format(prediction_stats.targets_ratio))
        print('predictions_ratio: {0:.2f}%'.format(prediction_stats.predictions_ratio))
