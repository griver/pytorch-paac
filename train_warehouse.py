import argparse
import logging
import os
import signal
import sys
import torch
from emulators.warehouse import WarehouseGameCreator, warehouse_tasks as tasks

import utils
import utils.evaluate as evaluate
from networks import vizdoom_nets
from multi_task import MultiTaskPAAC, network_dict
from batch_play import ConcurrentBatchEmulator, SequentialBatchEmulator, WorkerProcess
import multiprocessing

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
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


def main(args):
    network_creator, env_creator = get_network_and_environment_creator(args)
    logging.info(args_to_str(args))

    #batch_env = ConcurrentBatchEmulator(WorkerProcess, env_creator, args.num_workers, args.num_envs)
    batch_env = SequentialBatchEmulator(env_creator, args.num_envs, init_env_id=1)
    set_exit_handler(concurrent_emulator_handler(batch_env))
    try:
        #batch_env.start_workers()
        learner = MultiTaskPAAC(network_creator, batch_env, args)
        #learner.set_eval_function(eval_network, learner.network, env_creator, 10, learner.use_rnn)
        learner.train()
    finally:
        batch_env.close()


def get_network_and_environment_creator(args, random_seed=None):
    task_manager =  tasks.TaskManager(
        [tasks.Drop, tasks.PickUp, tasks.Visit, tasks.CarryItem],
        priorities=[1.5, 2., 1., 1.]
    )
    env_creator = WarehouseGameCreator(task_manager=task_manager, **vars(args))

    num_actions = env_creator.num_actions
    obs_shape = env_creator.obs_shape
    Network = vizdoom_nets[args.arch]

    def network_creator():
        if args.device == 'gpu':
            network = Network(num_actions, obs_shape, torch.cuda)
            network = network.cuda()
            logging.debug("Moved network's computations on a GPU")
        else:
            network = Network(num_actions, obs_shape, torch)
        return network

    return network_creator, env_creator


def add_multi_task_learner_args(parser):
    devices =['gpu', 'cpu'] if torch.cuda.is_available() else ['cpu']
    default_device = devices[0]
    nets = vizdoom_nets
    net_choices = list(nets.keys())
    default_workers = min(8, multiprocessing.cpu_count())
    show_default = " [default: %(default)s]"

    parser.add_argument('--arch', choices=net_choices, help="Which network architecture to train" + show_default,
                        dest="arch", required=True)
    parser.add_argument('-d', '--device', default=default_device, type=str, choices=devices,
                        help="Device to be used ('cpu' or 'gpu'). " +
                         "Use CUDA_VISIBLE_DEVICES to specify a particular gpu" + show_default,
                        dest="device")
    parser.add_argument('--e', default=0.025, type=float,
                        help="Epsilon for the Rmsprop and Adam optimizers."+show_default, dest="e")
    parser.add_argument('-lr', '--initial_lr', default=5e-3, type=float,
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
    parser.add_argument('--max_local_steps', default=10, type=int,
                        help="Number of steps to gain experience from before every update. "+show_default,
                        dest="max_local_steps")
    parser.add_argument('-n', '--num_envs', default=32, type=int,
                        help="Number of environments to run simultaneously. "+show_default, dest="num_envs")
    parser.add_argument('-w', '--workers', default=default_workers, type=int,
                        help="Number of parallel worker processes to run the environments. "+show_default,
                        dest="num_workers")
    parser.add_argument('-df', '--debugging_folder', default='logs/', type=str,
                        help="Folder where to save training progress.", dest="debugging_folder")
    parser.add_argument('--loss_scale', default=5., dest='loss_scaling', type=float,
                        help='Scales loss according to a given value'+show_default )
    parser.add_argument('--critic_coef', default=0.25, dest='critic_coef', type=float,
                        help='Weight of the critic loss in the total loss'+show_default)
    parser.add_argument('-tmc --termination_model_coef', default=1., dest='termination_model_coef', type=float,
                        help='Weight of the termination model loss in the total loss.'+show_default)
    parser.add_argument('--eval_every', default=None, type=int, dest='eval_every',
                        help='Model evaluation frequency.'+show_default)
    parser.add_argument('-tw', '--term_weights', default=[0.4, 1.6], nargs=2, type=float,
                        help='Class weights for the termination classifier loss.'+show_default)


def get_arg_parser():
    parser = argparse.ArgumentParser()
    WarehouseGameCreator.add_required_args(parser)
    add_multi_task_learner_args(parser)

    return parser


if __name__ == '__main__':
    args = get_arg_parser().parse_args()
    utils.save_args(args, args.debugging_folder, file_name=ARGS_FILE)
    logging.info('Saved args in the {0} folder'.format(args.debugging_folder))
    main(args)
