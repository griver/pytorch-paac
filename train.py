import argparse
import logging
import os
import signal
import sys
import torch
from emulators import VizdoomGamesCreator, AtariGamesCreator

import utils
from utils.lr_scheduler import LinearAnnealingLR
import utils.evaluate as evaluate
from networks import vizdoom_nets, atari_nets
from algos import ParallelActorCritic, ProximalPolicyOptimization
from batch_play import ConcurrentBatchEmulator, SequentialBatchEmulator, WorkerProcess
import multiprocessing
import numpy as np
from collections import namedtuple

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
FF_HISTORY_WINDOW=4
LSTM_HISTORY_WINDOW=1
ARGS_FILE='args.json'


def args_to_str(args):
    lines = ['','ARGUMENTS:']
    newline = os.linesep
    args = vars(args)
    for key in sorted(args.keys()):
        lines.append('    "{0}": {1}'.format(key, args[key]))
    return newline.join(lines)


exit_handler = None
def set_exit_handler(new_handler_func=None):
    #for some reason a creation of ViZDoom game(which starts a new subprocess) drops all previously set singal handlers.
    #therefore we reset handler_func right after the new game creation, which apparently happens only in the eval_network and main
    global exit_handler
    if new_handler_func is not None:
        exit_handler = new_handler_func

    if exit_handler:
        print('set up exit handler!')
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, exit_handler)


def concurrent_emulator_handler(batch_env):
    logging.debug('setup signal handler!!')
    main_process_pid = os.getpid()
    def signal_handler(signal, frame):
        if os.getpid() == main_process_pid:
            logging.info('Signal ' + str(signal) + ' detected, cleaning up.')
            batch_env.close()
            logging.info('Cleanup completed, shutting down...')
            sys.exit(0)

    return signal_handler


TrainingStats = namedtuple("TrainingStats", ['mean_r', 'max_r', 'min_r', 'std_r', 'mean_steps'])
def eval_network(network, env_creator, num_episodes, greedy=False, verbose=True):
    emulator = SequentialBatchEmulator(
        env_creator, num_episodes, False,
        specific_emulator_args={'single_life_episodes':False}
    )
    try:
        num_steps, rewards = evaluate.stats_eval(network, emulator, greedy=greedy)
        #_ = evaluate.visual_eval(network, env_creator,verbose=1, delay=0.1, visualize=True)
    finally:
        emulator.close()
        set_exit_handler()

    mean_steps = np.mean(num_steps)
    min_r, max_r = np.min(rewards), np.max(rewards)
    mean_r, std_r = np.mean(rewards), np.std(rewards)

    stats = TrainingStats(mean_r, max_r, min_r, std_r, mean_steps)
    if verbose:
        lines = ['Perfromed {0} tests:'.format(len(num_steps)),
                 'Mean number of steps: {0:.3f}'.format(mean_steps),
                 'Mean R: {0:.2f} | Std of R: {1:.3f}'.format(mean_r, std_r)]
        logging.info(utils.red('\n'.join(lines)))

    return stats


def main(args):
    utils.save_args(args, args.debugging_folder, file_name=ARGS_FILE)
    logging.info('Saved args in the {0} folder'.format(args.debugging_folder))
    logging.info(args_to_str(args))

    env_creator = get_environment_creator(args)
    network = create_network(args, env_creator.num_actions, env_creator.obs_shape)

    if args.algo == 'a2c':
        Optimizer = torch.optim.RMSprop #RMSprop defualts: momentum=0., centered=False, weight_decay=0
        Algorithm = ParallelActorCritic
        kwargs = dict(
            save_folder=args.debugging_folder,
            max_global_steps=args.max_global_steps,
            rollout_steps=args.rollout_steps,
            gamma=args.gamma,
            critic_coef=args.critic_coef,
            entropy_coef=args.entropy_coef,
            clip_norm=args.clip_norm,
            use_gae = args.use_gae
        )
    elif args.algo == 'ppo':
        Optimizer = torch.optim.Adam
        Algorithm = ProximalPolicyOptimization
        kwargs = dict(
            save_folder=args.debugging_folder,
            max_global_steps=args.max_global_steps,
            rollout_steps=args.rollout_steps,
            gamma=args.gamma,
            critic_coef=args.critic_coef,
            entropy_coef=args.entropy_coef,
            clip_norm=args.clip_norm,
            ppo_epochs=args.ppo_epochs, # default=5
            ppo_batch_size=args.ppo_batch_size, # defaults= 4
            ppo_clip=args.ppo_clip, # default=0.1
            use_gae = args.use_gae
        )

    opt = Optimizer(network.parameters(), lr=args.initial_lr, eps=args.e)
    step = Algorithm.update_from_checkpoint(
        args.debugging_folder, network, opt,
        use_cpu=args.device == 'cpu'
    )
    lr_scheduler = LinearAnnealingLR(opt, args.lr_annealing_steps)

    batch_env = ConcurrentBatchEmulator(WorkerProcess, env_creator, args.num_workers, args.num_envs)

    set_exit_handler(concurrent_emulator_handler(batch_env))
    try:
        batch_env.start_workers()
        learner = Algorithm(network, opt, lr_scheduler, batch_env, global_step=step, **kwargs)
        # evaluation results are saved as summaries of the training process:
        learner.evaluate = lambda network: eval_network(network, env_creator, 10)
        learner.train()
    finally:
        batch_env.close()


def get_environment_creator(args):
    if args.framework == 'vizdoom':
        env_creator = VizdoomGamesCreator(**vars(args))
    elif args.framework == 'atari':
        env_creator = AtariGamesCreator(**vars(args))
    return env_creator


def create_network(args, num_actions, obs_shape):
    if args.framework == 'vizdoom':
        NetworkCls = vizdoom_nets[args.arch]
    elif args.framework == 'atari':
        NetworkCls = atari_nets[args.arch]

    device = torch.device(args.device)
    #network explicitly stores device in order to facilitate casting of incoming tensors
    network = NetworkCls(num_actions, obs_shape, device)
    network = network.to(device)
    return network


def add_algo_args(parser, framework):
    devices =['cuda', 'cpu'] if torch.cuda.is_available() else ['cpu']
    default_device = devices[0]
    nets = vizdoom_nets if framework == 'vizdoom' else atari_nets
    net_choices = list(nets.keys())
    default_workers = min(8, multiprocessing.cpu_count())
    show_default = " [default: %(default)s]"

    parser.add_argument('-d', '--device', default=default_device, type=str, choices=devices,
                        help="Device to be used ('cpu' or 'cuda'). " +
                         "Use CUDA_VISIBLE_DEVICES to specify a particular GPU" + show_default,
                        dest="device")
    parser.add_argument('--algo', choices=['a2c','ppo'], default='a2c', dest='algo',
                        help="Algorithm to train."+show_default)
    parser.add_argument('--arch', choices=net_choices, help="Which network architecture to train"+show_default,
                        dest="arch", required=True)
    parser.add_argument('--e', default=1e-5, type=float,
                        help="Epsilon for the Rmsprop and Adam optimizers."+show_default, dest="e")
    parser.add_argument('-lr', '--initial_lr', default=7e-4, type=float,
                        help="Initial value for the learning rate."+show_default, dest="initial_lr",)
    parser.add_argument('-lra', '--lr_annealing_steps', default=80000000, type=int,
                        help="Nr. of global steps during which the learning rate will be linearly" +
                             "annealed towards zero." + show_default,
                        dest="lr_annealing_steps")
    parser.add_argument('--entropy', default=0.01, type=float,
                        help="Strength of the entropy regularization term (needed for actor-critic). "+show_default,
                        dest="entropy_coef")
    parser.add_argument('--clip_norm', default=1.0, type=float,
                        help="Grads will be clipped at the specified maximum (average) L2-norm. "+show_default,
                        dest="clip_norm")
    parser.add_argument('--gamma', default=0.99, type=float, help="Discount factor."+show_default, dest="gamma")
    parser.add_argument('--max_global_steps', default=80000000, type=int,
                        help="Number of training steps."+show_default,
                        dest="max_global_steps")
    parser.add_argument('-r', '--rollout_steps', default=10, type=int,
                        help="Number of steps to gain experience from before every update. "+show_default,
                        dest="rollout_steps")
    parser.add_argument('-n', '--num_envs', default=32, type=int,
                        help="Number of environments to run simultaneously. "+show_default, dest="num_envs")
    parser.add_argument('-w', '--workers', default=default_workers, type=int,
                        help="Number of parallel worker processes to run the environments. "+show_default,
                        dest="num_workers")
    parser.add_argument('-df', '--debugging_folder', default='logs/', type=str,
                        help="Folder where to save training progress.", dest="debugging_folder")
    parser.add_argument('--critic_coef', default=0.5, dest='critic_coef', type=float,
                        help='Weight of the critic loss in the total loss'+show_default)

    parser.add_argument('--gae', action='store_true', dest='use_gae',
                        help='Whether to use Generalized Advantage Estimator '
                             'or N-step Return for the advantage function estimation')

    parser.add_argument('-pe', '--ppo_epochs', default=4, type=int,
                        help="Number of training epochs in PPO."+show_default)
    parser.add_argument('-pc', '--ppo_clip', default=0.2,  type=float,
                        help="Clipping parameter for actor loss in PPO."+show_default)
    parser.add_argument('-pb', '--ppo_batch', default=4, type=int, dest='ppo_batch_size',
                        help='Batch size for PPO updates. If recurrent network is used '
                             'this parameters specify the number of trajectories to be sampled '
                             'from num_envs collected trajectories.'+show_default)


def get_arg_parser():
    parser = argparse.ArgumentParser()
    framework_parser = parser.add_subparsers(
        help='An RL friendly framework for agent-environment interaction',
        dest='framework')

    vz_parser = framework_parser.add_parser('vizdoom',  help="Arguments for the Vizdoom emulator")
    VizdoomGamesCreator.add_required_args(vz_parser)

    atari_parser = framework_parser.add_parser('atari', help="Arguments for the atari games")
    AtariGamesCreator.add_required_args(atari_parser)

    for framework, subparser in [('vizdoom', vz_parser), ('atari', atari_parser)]:
        paac_group = subparser.add_argument_group(
            title='Learning arguments', description='Arguments specific to the algorithm')
        add_algo_args(paac_group, framework)

    return parser


if __name__ == '__main__':
    args = get_arg_parser().parse_args()
    # if the specified architecture is a feedforward network then we use history window:
    args.history_window = FF_HISTORY_WINDOW if args.arch.endswith('ff') else LSTM_HISTORY_WINDOW
    args.random_seed = 3
    torch.set_num_threads(1) # sometimes pytorch works faster with this setting(from ~1300fps to 1500fps on ALE)
    main(args)
