import argparse
import logging
import os
import signal
import sys
import torch
from emulators import VizdoomGamesCreator, AtariGamesCreator

import utils
import utils.evaluate as evaluate
from networks.paac_nets import AtariFF, AtariLSTM, VizdoomLSTM
from paac import PAACLearner
from batch_play import ConcurrentBatchEmulator, SequentialBatchEmulator, WorkerProcess


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
FF_HISTORY_WINDOW=4
LSTM_HISTORY_WINDOW=1
ARGS_FILE='args.json'
vz_nets = {'lstm': VizdoomLSTM}
atari_nets = {'lstm': AtariLSTM, 'ff':AtariFF}


def args_to_str(args):
    lines = ['','ARGUMENTS:']
    newline = os.linesep
    args = vars(args)
    for key in sorted(args.keys()):
        lines.append('    "{0}": {1}'.format(key, args[key]))
    return newline.join(lines)


exit_handler = None
def set_exit_handler(new_handler_func=None):
    #for some reason a creation of Vizdoom game(which starts a new subprocess) drops all previously set singal handlers.
    #therefore we reset handler_func right new games creation, which apparently happens only in the eval_network and main
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


def eval_network(network, env_creator, num_episodes, is_recurrent, greedy=False):
    emulator = SequentialBatchEmulator(env_creator, num_episodes, False)
    try:
        stats = evaluate.stats_eval(network, emulator, is_recurrent=is_recurrent, greedy=greedy)
    finally:
        emulator.close()
        set_exit_handler()

    return stats


def main(args):
    network_creator, env_creator = get_network_and_environment_creator(args)

    utils.save_args(args, args.debugging_folder, file_name=ARGS_FILE)
    logging.info('Saved args in the {0} folder'.format(args.debugging_folder))
    logging.info(args_to_str(args))

    batch_env = ConcurrentBatchEmulator(WorkerProcess, env_creator, args.num_workers, args.num_envs)
    set_exit_handler(concurrent_emulator_handler(batch_env))
    try:
        batch_env.start_workers()
        learner = PAACLearner(network_creator, batch_env, args)
        learner.set_eval_function(eval_network,
                                  learner.network, env_creator, 10, learner.use_rnn) # args to eval_network
        learner.train()
    finally:
        batch_env.close()


def get_network_and_environment_creator(args, random_seed=None):
    if args.arch == 'lstm':
        args.history_window = LSTM_HISTORY_WINDOW
    elif args.arch == 'ff':
        args.history_window = FF_HISTORY_WINDOW

    if (not hasattr(args, 'random_seed')) or (random_seed is not None):
        args.random_seed = 3
    if args.framework == 'vizdoom':
        env_creator = VizdoomGamesCreator(args)
        Network = vz_nets[args.arch]
    elif args.framework == 'atari':
        env_creator = AtariGamesCreator(args)
        Network = atari_nets[args.arch]

    device = args.device
    num_actions = env_creator.num_actions
    obs_shape = env_creator.obs_shape
    def network_creator():
        if device == 'gpu':
            network = Network(num_actions, obs_shape, torch.cuda)
            network = network.cuda()
            logging.debug("Moved network's computations on a GPU")
        else:
            network = Network(num_actions, obs_shape, torch)
        return network

    return network_creator, env_creator


def add_paac_args(parser):
    parser.add_argument('-d', '--device', default='gpu', type=str, choices=['gpu', 'cpu'],
                        help="Device to be used ('cpu' or 'gpu'). Use CUDA_VISIBLE_DEVICES to specify a particular gpu",
                        dest="device")
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
    parser.add_argument('-n', '--num_envs', default=32, type=int,
                        help="The amount of emulators per agent. Default is 32.", dest="num_envs")
    parser.add_argument('-w', '--workers', default=8, type=int,
                        help="The amount of emulator workers per agent. Default is 8.", dest="num_workers")
    parser.add_argument('-df', '--debugging_folder', default='logs/', type=str,
                        help="Folder where to save the debugging information.", dest="debugging_folder")
    parser.add_argument('--arch', choices=['ff', 'lstm'], default='ff',
                        help="Which network architecture to use: a feedforward network or an lstm network", dest="arch")
    parser.add_argument('--loss_scale', default=5., dest='loss_scaling', type=float,
                        help='Scales loss according to a given value')
    parser.add_argument('--critic_coef', default=0.25, dest='critic_coef', type=float,
                        help='Weight of the critic loss in the total loss')


def get_arg_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    framework_parser = parser.add_subparsers(
        help='An RL friendly framework for agent-environment interaction',
        dest='framework')

    vz_parser = framework_parser.add_parser('vizdoom',  help="Arguments for the Vizdoom emulator")
    VizdoomGamesCreator.add_required_args(vz_parser)

    atari_parser = framework_parser.add_parser('atari', help="Arguments for the atari games")
    AtariGamesCreator.add_required_args(atari_parser)

    for p in [vz_parser, atari_parser]:
        paac_group = p.add_argument_group(
            title='PAAC arguments', description='Arguments specific to the algorithm')
        add_paac_args(paac_group)

    return parser


if __name__ == '__main__':
    args = get_arg_parser().parse_args()
    main(args)
