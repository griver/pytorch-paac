import argparse
import utils
import logging
import sys
import numpy as np
import multiprocessing

import torch
import utils.evaluate as evaluate
from networks import stoch_env_nets
from emulators.stoch_graphs import StochGraphCreator
from algos import ParallelActorCritic, ProximalPolicyOptimization
from train import args_to_str, set_exit_handler, concurrent_emulator_handler

from utils.lr_scheduler import LinearAnnealingLR
from batch_play import SharedMemWorker, SharedMemBatchEnv, SequentialBatchEnv

from collections import namedtuple
TrainingStats = namedtuple("TrainingStats", ['mean_r', 'mean_steps', 'std_steps'])


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
ARGS_FILE='train_args.json'


def main(args):
    utils.save_args(args, args.save_folder, file_name=ARGS_FILE)
    logging.info('Saved main_args in the {} folder'.format(args.save_folder))
    logging.info(args_to_str(args))

    env_creator = StochGraphCreator(**vars(args))
    network = create_network(args, env_creator.num_actions, env_creator.obs_shape)

    if args.algo == 'a2c':
        OptimizerCls = torch.optim.RMSprop  # RMSprop defualts: momentum=0., centered=False, weight_decay=0
        AlgorithmCls = ParallelActorCritic
        algo_specific_args = dict()
    elif args.algo == 'ppo':
        OptimizerCls = torch.optim.Adam
        AlgorithmCls = ProximalPolicyOptimization
        algo_specific_args = dict(
            ppo_epochs=args.ppo_epochs,  # default=5
            ppo_batch_num=args.ppo_batch_num,  # defaults= 4
            ppo_clip=args.ppo_clip,  # default=0.1
        )
    else:
        raise ValueError('Only ppo and a2c are implemented right now!')

    opt = OptimizerCls(network.parameters(), lr=args.initial_lr, eps=args.e)
    global_step = AlgorithmCls.update_from_checkpoint(
        args.save_folder, network, opt,
        use_cpu=args.device == 'cpu'
    )
    lr_scheduler = LinearAnnealingLR(opt, args.lr_annealing_steps)

    batch_env = SharedMemBatchEnv(SharedMemWorker, env_creator, args.num_workers, args.num_envs)

    set_exit_handler(concurrent_emulator_handler(batch_env))
    try:
        batch_env.start_workers()
        learner = AlgorithmCls(
            network, opt,
            lr_scheduler,
            batch_env,
            global_step=global_step,
            save_folder=args.save_folder,
            max_global_steps=args.max_global_steps,
            rollout_steps=args.rollout_steps,
            gamma=args.gamma,
            critic_coef=args.critic_coef,
            entropy_coef=args.entropy_coef,
            clip_norm=args.clip_norm,
            use_gae=args.use_gae,
            **algo_specific_args)
        # evaluation results are saved as summaries of the training process:
        learner.evaluate = lambda network:eval_network(network, env_creator, 40)
        learner.train()
    finally:
        batch_env.close()


def create_network(args, num_actions, obs_shape):
    NetworkCls = stoch_env_nets[args.arch]
    device = torch.device(args.device)
    network = NetworkCls(num_actions, obs_shape, device)
    # IMPROVE: if i send device then i don't need to cast it myself!!
    network.to(device)
    return network


def eval_network(network, env_creator, num_episodes,
                 greedy=False, verbose=True):

    emulator = SequentialBatchEnv(
        env_creator, num_episodes, False,
    )
    try:
        num_steps, rewards = evaluate.stats_eval(network, emulator, greedy=greedy)
        #_ = evaluate.visual_eval(network, env_creator,verbose=1, delay=0.1, visualize=True)
    finally:
        emulator.close()
        set_exit_handler()

    mean_steps, std_steps = np.mean(num_steps), np.std(num_steps)
    mean_r = np.mean(rewards)

    stats = TrainingStats(mean_r, mean_steps, std_steps)
    if verbose:
        lines = ['Perfromed {0} tests:'.format(len(num_steps)),
                 'Mean number of steps: {0:.3f} | Std: {1:.3f}'.format(mean_steps, std_steps),
                 'Mean R: {0:.3f}'.format(mean_r)]
        logging.info(utils.red('\n'.join(lines)))

    return stats


def count_actions():
    args = handle_commandline()
    args.game = 'erdos_renyi-n400-1'
    print(args_to_str(args))
    print()

    topology, size, start_id = args.game.split('-')
    start_id = int(start_id)
    max_acts = 0
    acts_per_env = []
    for i in range(10):
        game = '-'.join([topology, size, str(start_id + i)])
        args.game = game
        print(args.game)

        env_creator = StochGraphCreator(**vars(args))
        num_acts, obs_shape = env_creator.num_actions, env_creator.obs_shape
        max_acts = max(num_acts, max_acts)
        acts_per_env.append(num_acts)

        print('num_actions:', num_acts, 'obs_shape:', obs_shape)
        print()

    print('maximum number of acts is:', max_acts,
          'mean:', np.mean(acts_per_env))


def handle_commandline():
    devices = ['cuda', 'cpu'] if torch.cuda.is_available() else ['cpu']
    default_device = devices[0]
    net_choices = list(stoch_env_nets.keys())
    default_workers = min(8, multiprocessing.cpu_count())
    show_default = " [default: %(default)s]"

    parser = argparse.ArgumentParser()
    # environment main_args:
    StochGraphCreator.add_required_args(parser)
    # algorithm and model main_args:
    parser.add_argument('-d', '--device', default=default_device, type=str, choices=devices,
                        help="Device to be used ('cpu' or 'cuda'). " +
                             "Use CUDA_VISIBLE_DEVICES to specify a particular GPU" + show_default,
                        dest="device")

    parser.add_argument('--algo', choices=['a2c', 'ppo'], default='a2c', dest='algo',
                        help="Algorithm to train." + show_default)

    parser.add_argument('--arch', choices=net_choices, default=net_choices[0],
                        help="Which network architecture to train" + show_default,
                        dest="arch")

    parser.add_argument('--e', default=1e-5, type=float,
                        help="Epsilon for the Rmsprop and Adam optimizers." + show_default, dest="e")

    parser.add_argument('-lr', '--initial-lr', default=7e-4, type=float,
                        help="Initial value for the learning rate." + show_default, dest="initial_lr", )

    parser.add_argument('-lra', '--lr-annealing-steps', default=80000000, type=int,
                        help="Nr. of global steps during which the learning rate will be linearly" +
                             "annealed towards zero." + show_default,
                        dest="lr_annealing_steps")

    parser.add_argument('--entropy', default=0.01, type=float,
                        help="Strength of the entropy regularization term (needed for actor-critic). " + show_default,
                        dest="entropy_coef")

    parser.add_argument('--clip-norm', default=1.0, type=float,
                        help="Grads will be clipped at the specified maximum (average) L2-norm. " + show_default,
                        dest="clip_norm")

    parser.add_argument('--gamma', default=0.99, type=float, help="Discount factor." + show_default, dest="gamma")

    parser.add_argument('--max-global-steps', default=80000000, type=int,
                        help="Number of training steps." + show_default,
                        dest="max_global_steps")

    parser.add_argument('-r', '--rollout-steps', default=10, type=int,
                        help="Number of steps to gain experience from before every update. " + show_default,
                        dest="rollout_steps")

    parser.add_argument('-n', '--num-envs', default=32, type=int,
                        help="Number of environments to run simultaneously. " + show_default, dest="num_envs")

    parser.add_argument('-w', '--workers', default=default_workers, type=int,
                        help="Number of parallel worker processes to run the environments. " + show_default,
                        dest="num_workers")

    parser.add_argument('-sf', '--save-folder', default='logs/', type=str,
                        help="Folder where to save training progress.", dest="save_folder")

    parser.add_argument('--critic-coef', default=0.5, dest='critic_coef', type=float,
                        help='Weight of the critic loss in the total loss' + show_default)

    parser.add_argument('--gae', action='store_true', dest='use_gae',
                        help='Whether to use Generalized Advantage Estimator '
                             'or N-step Return for the advantage function estimation')

    parser.add_argument('-pe', '--ppo-epochs', default=4, type=int,
                        help="Number of training epochs in PPO." + show_default)

    parser.add_argument('-pc', '--ppo-clip', default=0.2, type=float,
                        help="Clipping parameter for actor loss in PPO." + show_default)

    parser.add_argument('-pb', '--ppo-batch', default=4, type=int, dest='ppo_batch_num',
                        help='Number of bathes for one ppo_epoch. In case of a recurrent network'
                             'batches consist from entire trajectories, otherwise from one-step transitions.'
                             + show_default)

    args = parser.parse_args()

    if not hasattr(args, 'random_seed'):
        args.random_seed = 3

    #we definitely don't want learning rate to become zero before the training ends:
    if args.lr_annealing_steps < args.max_global_steps:
        new_value = args.max_global_steps
        args.lr_annealing_steps = new_value
        logging.warning('lr_annealing_steps was changed to {}'.format(new_value))

    return args


if __name__ == '__main__':
    args = handle_commandline()
    torch.set_num_threads(1)
    main(args)
