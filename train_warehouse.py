import argparse
import logging
import sys
import torch
import numpy as np
from emulators.warehouse import WarehouseGameCreator, warehouse_tasks as tasks

import utils
from utils import eval_warehouse as evaluate
from networks import warehouse_nets
from collections import namedtuple
from multi_task import MultiTaskActorCritic
from batch_play import ConcurrentBatchEmulator, SequentialBatchEmulator, WorkerProcess
import multiprocessing
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
ARGS_FILE='args_multi_task.json'

from train import args_to_str, concurrent_emulator_handler, set_exit_handler

TrainingStats = namedtuple("TrainingStats",
                           ['mean_r', 'std_r', 'tasks_stats',
                           'term_acc', 'term_rec', 'term_prec',
                           't_ratio', 'p_ratio'])

def eval_network(network, env_creator, num_episodes, greedy=False, verbose=True, **emulator_args):
    emulator = SequentialBatchEmulator(env_creator, num_episodes, auto_reset=False,
                                       specific_emulator_args=emulator_args)
    try:
        stats = evaluate.stats_eval(network, emulator, greedy=greedy)
    finally:
        emulator.close()
        set_exit_handler()

    num_steps, rewards, prediction_stats, task_stats = stats
    mean_r, std_r = np.mean(rewards), np.std(rewards)
    acc = prediction_stats.accuracy
    rec = prediction_stats.recall
    prec = prediction_stats.precision
    targets_ratio = prediction_stats.targets_ratio
    preds_ratio = prediction_stats.predictions_ratio

    stats = TrainingStats(
        mean_r=mean_r, std_r=std_r, tasks_stats=task_stats,
        term_acc=acc, term_prec=prec, term_rec=rec,
        t_ratio=targets_ratio, p_ratio=preds_ratio
    )
    if verbose:
        lines = [
            'Perfromed {0} tests:'.format(len(num_steps)),
            'Mean R: {0:.2f} | Std of R: {1:.3f}'.format(mean_r, std_r),
            'Mean number of steps: {0:.3f}'.format(np.mean(num_steps)),
            'Tasks Statistics:',
            task_stats.pretty_repr(),
            'Termination Predictor:',
            'Acc: {:.2f}% | Precision: {:.2f}% | Recall: {:.2f}'.format(acc, prec, rec),
            'Class 1 ratio. Targets: {0:.2f}% Preds: {1:.2f}%'.format(targets_ratio, preds_ratio)]
        logging.info(utils.red('\n'.join(lines)))

    return stats


def main(args):
    network_creator, env_creator = get_network_and_environment_creator(args)
    logging.info(args_to_str(args))

    batch_env = ConcurrentBatchEmulator(WorkerProcess, env_creator, args.num_workers, args.num_envs)
    #batch_env = SequentialBatchEmulator(env_creator, args.num_envs, init_env_id=1)
    set_exit_handler(concurrent_emulator_handler(batch_env))
    try:
        batch_env.start_workers()
        learner = MultiTaskActorCritic(network_creator, batch_env, args)
        learner.set_eval_function(eval_network, learner.network, env_creator, 16, verbose=True)
        learner.train()
    finally:
        batch_env.close()


def get_network_and_environment_creator(args, random_seed=None):
    #task_manager =  tasks.TaskManager(
    #    [tasks.Drop, tasks.PickUpPassenger, tasks.Visit, tasks.CarryItem],
    #    priorities=[1., 1., .35, 1.]
    #)
    task_manager = tasks.TaskManager([tasks.Visit])
    env_creator = WarehouseGameCreator(task_manager=task_manager, **vars(args))

    num_actions = env_creator.num_actions
    obs_shape = env_creator.obs_shape
    Network = warehouse_nets[args.arch]

    def network_creator():
        if args.device == 'gpu':
            network = Network(num_actions, obs_shape, torch.cuda,
                              num_tasks=4+1, num_properties=4+8+1)#num_tasks+NoTask
            network = network.cuda()
            logging.debug("Moved network's computations on a GPU")
        else:
            network = Network(num_actions, obs_shape, torch,
                              num_tasks=4+1, num_properties=4+8+1) #4items+8textures+NoProperty
        return network

    return network_creator, env_creator


def add_multi_task_learner_args(parser):
    devices =['gpu', 'cpu'] if torch.cuda.is_available() else ['cpu']
    default_device = devices[0]
    nets = warehouse_nets
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
                        dest="entropy_coef")
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
