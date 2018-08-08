import argparse
import logging
import os
import signal
import sys
import torch
from emulators import VizdoomGamesCreator, AtariGamesCreator

import utils
import utils.evaluate as evaluate
from networks import vizdoom_nets, atari_nets
from paac import ParallelActorCritic
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
    emulator = SequentialBatchEmulator(env_creator, num_episodes, False)
    try:
        num_steps, rewards = evaluate.stats_eval(network, emulator, greedy=greedy)
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
    env_creator = get_environment_creator(args)
    network = create_network(args, env_creator.num_actions, env_creator.obs_shape)

    utils.save_args(args, args.debugging_folder, file_name=ARGS_FILE)
    logging.info('Saved args in the {0} folder'.format(args.debugging_folder))
    logging.info(args_to_str(args))

    #batch_env = SequentialBatchEmulator(env_creator, args.num_envs, init_env_id=1)
    batch_env = ConcurrentBatchEmulator(WorkerProcess, env_creator, args.num_workers, args.num_envs)
    set_exit_handler(concurrent_emulator_handler(batch_env))
    try:
        batch_env.start_workers()
        learner = ParallelActorCritic(network, batch_env, args)
        # evaluation results are saved as summaries of the training process:
        learner.evaluate = lambda network: eval_network(network, env_creator, 10)
        learner.train()
    finally:
        batch_env.close()


def get_environment_creator(args):
    if args.framework == 'vizdoom':
        env_creator = VizdoomGamesCreator(args)
    elif args.framework == 'atari':
        env_creator = AtariGamesCreator(args)
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


def add_paac_args(parser, framework):
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
    parser.add_argument('--e', default=0.02, type=float,
                        help="Epsilon for the Rmsprop and Adam optimizers."+show_default, dest="e")
    parser.add_argument('-lr', '--initial_lr', default=0.007, type=float,
                        help="Initial value for the learning rate."+show_default, dest="initial_lr",)
    parser.add_argument('-lra', '--lr_annealing_steps', default=80000000, type=int,
                        help="Nr. of global steps during which the learning rate will be linearly" +
                             "annealed towards zero." + show_default,
                        dest="lr_annealing_steps")
    parser.add_argument('--entropy', default=0.02, type=float,
                        help="Strength of the entropy regularization term (needed for actor-critic). "+show_default,
                        dest="entropy_regularisation_strength")
    parser.add_argument('--clip_norm', default=3.0, type=float,
                        help="If clip_norm_type is local/global, grads will be"+
                             "clipped at the specified maximum (average) L2-norm. "+show_default,
                        dest="clip_norm")
    parser.add_argument('--clip_norm_type', default="global",
                        help="""Whether to clip grads by their norm or not. Values: ignore (no clipping),
                         local (layer-wise norm), global (global norm)"""+show_default,
                        dest="clip_norm_type")
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
    parser.add_argument('--arch', choices=net_choices, help="Which network architecture to train"+show_default,
                        dest="arch", required=True)
    parser.add_argument('--loss_scale', default=5., dest='loss_scaling', type=float,
                        help='Scales loss according to a given value'+show_default )
    parser.add_argument('--critic_coef', default=0.5, dest='critic_coef', type=float,
                        help='Weight of the critic loss in the total loss'+show_default)


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
            title='PAAC arguments', description='Arguments specific to the algorithm')
        add_paac_args(paac_group, framework)

    return parser


if __name__ == '__main__':
    args = get_arg_parser().parse_args()
    # if the specified architecture is a feedforward network then we use history window:
    args.history_window = FF_HISTORY_WINDOW if args.arch.endswith('ff') else LSTM_HISTORY_WINDOW
    args.random_seed = 3
    torch.set_num_threads(1) # sometimes pytorch works faster with this setting(from ~1300fps to 1500fps on ALE)
    main(args)
