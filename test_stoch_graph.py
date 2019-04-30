import argparse
import time

import numpy as np
import torch

from networks import old_preprocess_images
import utils
from algos import ParallelActorCritic
from train_stoch_graph import create_network, eval_network, \
    evaluate, args_to_str, StochGraphCreator


def fix_args_for_test(args, train_args):
    for k, v in train_args.items():
        if getattr(args, k, None) == None: #this includes cases where main_args.k is None
            setattr(args, k, v)

    args.max_global_steps = 0
    rnd = np.random.RandomState(int(time.time()))
    args.random_seed = rnd.randint(1000)

    return args


if __name__=='__main__':
    devices = ['cuda', 'cpu'] if torch.cuda.is_available() else ['cpu']
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('folder', type=str, help="Folder with a trained model.")
    parser.add_argument('-tc', '--test_count', default=1, type=int, help="Number of episodes to test the model", dest="test_count")
    parser.add_argument('-g', '--greedy', action='store_true', help='Determines whether to use a stochastic or deterministic policy')
    parser.add_argument('-d', '--device', default=devices[0], type=str, choices=devices,
        help="Device to be used ('cpu' or 'cuda'). Use CUDA_VISIBLE_DEVICES to specify a particular GPU", dest="device")
    parser.add_argument('--use_best', action='store_true',
                        help='Whether to load a last saved model or a model with the best score')

    args = parser.parse_args()
    train_args = utils.load_args(args.folder, 'train_args.json')
    args = fix_args_for_test(args, train_args)


    env_creator = StochGraphCreator(**vars(args))
    network = create_network(args, env_creator.num_actions, env_creator.obs_shape)
    steps_trained = ParallelActorCritic.update_from_checkpoint(args.folder, network, use_cpu=(args.device=='cpu'), use_best=args.use_best)

    print(args_to_str(args), '=='*30, sep='\n')
    print('Model was trained for {} steps'.format(steps_trained))

    eval_network(network, env_creator, args.test_count, greedy=args.greedy)

