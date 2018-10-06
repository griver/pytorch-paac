import argparse
import time

import numpy as np
import torch

from networks import old_preprocess_images
import utils
from paac import ParallelActorCritic
from train import get_environment_creator, create_network, eval_network, evaluate, args_to_str


def fix_args_for_test(args, train_args):
    for k, v in train_args.items():
        if getattr(args, k, None) == None: #this includes cases where args.k is None
            setattr(args, k, v)

    args.max_global_steps = 0
    rnd = np.random.RandomState(int(time.time()))
    args.random_seed = rnd.randint(1000)

    if args.framework == 'vizdoom':
        args.reward_coef = 1.
        args.step_delay = 0.33
    elif args.framework == 'atari':
        args.random_start = True
        args.single_life_episodes = False
        args.step_delay = 0
    return args


if __name__=='__main__':
    devices = ['cuda', 'cpu'] if torch.cuda.is_available() else ['cpu']
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('folder', type=str, help="Folder with a trained model.")
    parser.add_argument('-tc', '--test_count', default=1, type=int, help="Number of episodes to test the model", dest="test_count")
    parser.add_argument('-g', '--greedy', action='store_true', help='Determines whether to use a stochastic or deterministic policy')
    parser.add_argument('-d', '--device', default=devices[0], type=str, choices=devices,
        help="Device to be used ('cpu' or 'cuda'). Use CUDA_VISIBLE_DEVICES to specify a particular GPU", dest="device")
    parser.add_argument('-v', '--visualize', action='store_true')
    parser.add_argument('--old_preprocessing', action='store_true',
                        help="""Previous image preprocessing squashed values in a [0, 255] int range to a [0.,1.] float range.
                                The new one returns an image with values in a [-1.,1.] float range.""")

    args = parser.parse_args()
    train_args = utils.load_args(folder=args.folder)
    args = fix_args_for_test(args, train_args)


    env_creator = get_environment_creator(args)
    network = create_network(args, env_creator.num_actions, env_creator.obs_shape)
    steps_trained = ParallelActorCritic.update_from_checkpoint(args.folder, network, use_cpu=(args.device=='cpu'))

    if args.old_preprocessing:
        network._preprocess = old_preprocess_images

    print(args_to_str(args), '=='*30, sep='\n')
    print('Model was trained for {} steps'.format(steps_trained))
    if not args.visualize:
        #eval_network prints stats by itself
        eval_network(network, env_creator, args.test_count, greedy=args.greedy)
    else:
        num_steps, rewards = evaluate.visual_eval(
            network, env_creator, args.greedy,
            args.test_count, verbose=1, delay=args.step_delay
        )
        print('Perfromed {0} tests'.format(args.test_count))
        print('Mean number of steps: {0:.3f}'.format(np.mean(num_steps)))
        print('Mean R: {0:.2f}'.format(np.mean(rewards)), end=' | ')
        print('Max R: {0:.2f}'.format(np.max(rewards)), end=' | ')
        print('Min R: {0:.2f}'.format(np.min(rewards)), end=' | ')
        print('Std of R: {0:.2f}'.format(np.std(rewards)))

