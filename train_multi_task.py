import argparse
import logging
import sys
import torch
import numpy as np

from collections import namedtuple
from emulators import TaxiGamesCreator
import utils
import utils.eval_taxi as evaluate
from utils.lr_scheduler import LinearAnnealingLR
from algos_multi_task import MultiTaskA2C, MultiTaskPPO
from networks import taxi_nets, preprocess_taxi_input

from train import args_to_str, concurrent_emulator_handler, set_exit_handler
from batch_play import ConcurrentBatchEmulator, SequentialBatchEmulator, WorkerProcess
import multiprocessing


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
ARGS_FILE = 'args_multi_task.json'


TrainingStats = namedtuple("TrainingStats",
                           ['mean_r','max_r','min_r','std_r',
                            'mean_steps','term_acc','term_rec',
                            'term_prec','t_ratio', 'p_ratio'])


ExtraTrainingStats = namedtuple("ExtraTrainingStats",
                           ['mean_r','max_r','min_r','std_r',
                            'mean_steps','term_acc','term_rec',
                            'term_prec','task_stats'])


def eval_network(network, env_creator, num_episodes,
                 greedy=False, term_threshold=0.5, verbose=True):
    emulator = SequentialBatchEmulator(
        env_creator, num_episodes, False,
        specific_emulator_args={'single_life_episodes':False}
    )
    try:
        num_steps, rewards, extras = evaluate.stats_eval(
            network, emulator,
            greedy=greedy,
            termination_threshold=term_threshold
        )
    finally:
        emulator.close()
        set_exit_handler()
    done_preds = extras['done_pred_stats']
    task_stats = extras['task_stats']

    mean_steps = np.mean(num_steps)
    min_r, max_r = np.min(rewards), np.max(rewards)
    mean_r, std_r = np.mean(rewards), np.std(rewards)
    acc = done_preds.accuracy
    rec = done_preds.recall
    prec = done_preds.precision
    targets_ratio = done_preds.targets_ratio
    preds_ratio = done_preds.predictions_ratio

    stats = ExtraTrainingStats(
        mean_r=mean_r, min_r=min_r, max_r=max_r, std_r=std_r,
        term_acc=acc, term_prec=prec, term_rec=rec,
        mean_steps=mean_steps, task_stats=task_stats.logging_form()
    )

    if verbose:
        lines = [
            'Perfromed {0} tests:'.format(len(num_steps)),
            'Mean number of steps: {0:.3f}'.format(mean_steps),
            'Mean R: {0:.2f} | Std of R: {1:.3f}'.format(mean_r, std_r),
            'Termination Predictor:',
            'Acc: {:.2f}% | Precision: {:.2f} | Recall: {:.2f}'.format(acc, prec, rec),
            'Class 1 ratio. Targets: {:.2f}% Preds: {:.2f}%'.format(targets_ratio, preds_ratio),
            'Task Statistics:',
            str(task_stats),
        ]
        logging.info(utils.red('\n'.join(lines)))

    return stats


def main(args):
    utils.save_args(args, args.debugging_folder, file_name=ARGS_FILE)
    logging.info('Saved main_args in the {0} folder'.format(args.debugging_folder))
    logging.info(args_to_str(args))

    env_creator = TaxiGamesCreator(**vars(args))
    #RMSprop defualts: momentum=0., centered=False, weight_decay=0
    network = create_network(args, env_creator.num_actions, env_creator.obs_shape)

    if args.algo == 'a2c':
        OptimizerCls = torch.optim.RMSprop
        AlgorithmCls = MultiTaskA2C
        algo_specific_args = dict()

    elif args.algo == 'ppo':
        OptimizerCls = torch.optim.Adam
        AlgorithmCls = MultiTaskPPO
        algo_specific_args = dict(
            ppo_epochs=args.ppo_epochs,  # default=5
            ppo_batch_num=args.ppo_batch_num,  # defaults= 4
            ppo_clip=args.ppo_clip,  # default=0.1
            kl_threshold=0.01
        )
    else:
        raise ValueError('Only ppo and a2c are implemented right now!')

    opt = OptimizerCls(network.parameters(), lr=args.initial_lr, eps=args.e)
    global_step = AlgorithmCls.update_from_checkpoint(
        args.debugging_folder,network, opt,
        use_cpu=args.device=='cpu'
    )
    lr_scheduler = LinearAnnealingLR(opt, args.lr_annealing_steps)

    batch_env = ConcurrentBatchEmulator(WorkerProcess, env_creator, args.num_workers, args.num_envs)

    set_exit_handler(concurrent_emulator_handler(batch_env))
    try:
        batch_env.start_workers()
        learner = AlgorithmCls(
            network, opt,
            lr_scheduler,
            batch_env,
            global_step=global_step,
            save_folder=args.debugging_folder,
            max_global_steps=args.max_global_steps,
            rollout_steps=args.rollout_steps,
            gamma=args.gamma,
            critic_coef=args.critic_coef,
            entropy_coef=args.entropy_coef,
            clip_norm=args.clip_norm,
            use_gae=args.use_gae,
            term_weights=args.term_weights,
            termination_model_coef=args.termination_model_coef,
            **algo_specific_args
        )
        learner.evaluate = lambda net:eval_network(net, env_creator, 50)
        learner.train()
    finally:
        batch_env.close()


def create_network(args, num_actions, obs_shape):
    NetworkCls = taxi_nets[args.arch]
    device = torch.device(args.device)
    use_conv = getattr(args, 'use_conv', True)
    #network explicitly stores device in order to facilitate casting of incoming tensors
    network = NetworkCls(num_actions, obs_shape, device,
                         preprocess=preprocess_taxi_input,
                         use_location=args.use_location,
                         use_conv=use_conv,
                         num_tasks=(6 if args.game != 'taxi_plus' else 12))
    network = network.to(device)
    return network


def handle_command_line(parser, args_line=None):
    if args_line:
        args = parser.parse_args(args_line.split())
    else:
        args = parser.parse_args()

    if not hasattr(args, 'random_seed'):
        args.random_seed = 3

    #we definitely don't want learning rate to become zero before the training ends:
    if args.lr_annealing_steps < args.max_global_steps:
        new_value = args.max_global_steps
        args.lr_annealing_steps = new_value
        logging.warning('lr_annealing_steps was changed to {}'.format(new_value))

    def view_size(args):
        #changes view_size if full_view is specified
        min_side, max_side = args.map_size
        view = min([max_side, args.view_size])
        if view < args.view_size:
            logging.warning('View size has been changed to match the specified maximum map size')
        return view, view

    args.view_size = view_size(args)
    args.map_size = list(args.map_size) + list(args.map_size)

    return args


def get_arg_parser():
    devices = ['cuda', 'cpu'] if torch.cuda.is_available() else ['cpu']
    default_device = devices[0]
    net_choices = list(taxi_nets.keys())
    default_workers = min(8, multiprocessing.cpu_count())
    show_default = " [default: %(default)s]"

    parser = argparse.ArgumentParser()
    #environment main_args:
    TaxiGamesCreator.add_required_args(parser)
    #algorithm and model main_args:
    parser.add_argument('--algo', choices=['a2c', 'ppo'], default='a2c', dest='algo',
                        help="Algorithm to train." + show_default)
    parser.add_argument('--arch', choices=net_choices,  dest="arch", required=True,
                        help="Which network architecture to train"+show_default)
    parser.add_argument('--e', default=1e-5, type=float, help="Epsilon for the Rmsprop optimizer"+ show_default, dest="e")
    parser.add_argument('-lr', '--initial_lr', default=7e-4, type=float,  dest="initial_lr",
                        help="Initial value for the learning rate."+show_default)
    parser.add_argument('-lra', '--lr_annealing_steps', default=80000000, type=int, dest="lr_annealing_steps",
                        help="Number of global steps during which the learning rate will be linearly annealed towards zero"+show_default)
    parser.add_argument('--critic_coef', default=0.5, dest='critic_coef', type=float,
                        help='Weight of the critic loss in the total loss'+show_default)
    parser.add_argument('--entropy', default=0.01, type=float, dest="entropy_coef",
                      help="default=0.02. Strength of the entropy regularization term"+show_default)
    parser.add_argument('--clip_norm', default=0.7, type=float, dest="clip_norm",
                        help="Grads will be clipped at the specified maximum (average) L2-norm"+show_default)
    parser.add_argument('-l', '--use_loc', action='store_true', dest='use_location',
                        help="If provided an agent will use it's coordinates as part of the observation")
    parser.add_argument('--no_conv', action='store_false', dest='use_conv',
                        help='Whether to use a convolutional module')

    parser.add_argument('--gae', action='store_true', dest='use_gae',
                        help='Whether to use Generalized Advantage Estimator '
                             'or N-step Return for the advantage function estimation')

    parser.add_argument('-pe', '--ppo_epochs', default=4, type=int,
                        help="Number of training epochs in PPO."+show_default)
    parser.add_argument('-pc', '--ppo_clip', default=0.1,  type=float,
                        help="Clipping parameter for actor loss in PPO."+show_default)
    parser.add_argument('-pb', '--ppo_batch', default=4, type=int, dest='ppo_batch_num',
                        help='Number of bathes for one ppo_epoch. In case of a recurrent network'
                             'batches consist from entire trajectories, otherwise from one-step transitions.'
                             +show_default)

    #termination predictor main_args:
    parser.add_argument('-tmc', '--termination_model_coef', default=1., dest='termination_model_coef', type=float,
                        help='Weight of the termination model loss in the total loss'+show_default)
    parser.add_argument('-tw', '--term_weights', default=[0.4, 1.6], nargs=2, type=float,
                        help='Class weights for the termination classifier loss.'+show_default)
    #main_args that common for any model that learns on parallel environments:
    parser.add_argument('-d', '--device', default=default_device, type=str, choices=devices, dest="device",
                        help="Device to be used ('cpu' or 'cuda'). Use CUDA_VISIBLE_DEVICES to specify a particular GPU" + show_default)
    parser.add_argument('--max_global_steps', default=80000000, type=int, dest="max_global_steps",
                        help="Max. number of interaction steps"+show_default)
    parser.add_argument('-r', '--rollout_steps', default=10, type=int, dest="rollout_steps",
                        help="Number of steps to gain experience from before every update."+show_default)
    parser.add_argument('-n', '--num_envs', default=32, type=int, dest="num_envs",
                        help="Number of environments to run simultaneously. " + show_default)
    parser.add_argument('-w', '--workers', default=default_workers, type=int, dest="num_workers",
                        help="Number of parallel worker processes to handle the environments. " + show_default)
    #save and print:
    parser.add_argument('-df', '--debugging_folder', default='test_log/', type=str, dest="debugging_folder",
                        help="Folder where to save summaries and trained models"+show_default)
    parser.add_argument('-v', '--verbose', default=1, type=int, dest="verbose",
                        help="determines how much information to show during training" + show_default)

    return parser


if __name__ == '__main__':
    args = handle_command_line(get_arg_parser())
    torch.set_num_threads(1)  # sometimes pytorch works faster with this setting
    main(args)
