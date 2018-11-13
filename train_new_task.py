import train_multi_task as mt_train
import logging
import utils
from emulators import TaxiGamesCreator
import torch
from algos_multi_task import MultiTaskActorCritic
from utils.lr_scheduler import LinearAnnealingLR
from batch_play import ConcurrentBatchEmulator, WorkerProcess


def get_arg_parser():

    parser = mt_train.get_arg_parser()
    parser.add_argument('-lf', '--load_folder', default='pretrained/algos_multi_task/mt_factor_5to10_ttm/',
                        type=str, help='path to a pretrained model')
    parser.add_argument('-tl','--train_layers', nargs='+', type=str,required=True,
                        help="List of layers for retraining"
                             "( i.e: task_lstm.embedding fc_terminal fc_value),"
                             " the remaining layers will be frozen!")
    parser.add_argument('--no_finetuning', action='store_false', dest='finetuning',
                        help='If no_finetuning is specified then all unfrozen layers'
                             ' will be reinitialized with random values!')
    #retrain_modules = kwargs.pop('retrain_modules', set('task_lstm.embedding', 'fc_terminal','fc_value')
    return parser

def handle_command_line(parser, args_line=None):
    args = mt_train.handle_command_line(parser, args_line)

    if not hasattr(args, 'finetuning'):
        args.finetuning = True

    return args

def split_network_parameters(net, retrain_layers, finetuning=True):
    """
    Return a 2-tuple containing a list of parameters
    for training and a list of parameters that will be frozen
    """
    # We assume that parameter names have the following format:
    #higher_module.lower_module.parameter_name
    get_module_name = lambda name: name.rpartition('.')[0]

    train, freeze = [], []

    freeze_color = utils.cyan
    train_color = utils.green if finetuning else utils.red
    train_type = 'finetuning' if finetuning else 'train'
    log_str = ['Model parameters({}, {}):'.format(freeze_color('freeze'), train_color(train_type)),]

    for name, param in net.named_parameters():
        if get_module_name(name) in retrain_layers:
            train.append((name, param))
            log_str.append(train_color(name))
        else:
            freeze.append((name, param))
            log_str.append(freeze_color(name))

    for name, p in freeze:
        #print('Freeze parameter:', name)
        p.requires_grad = False

    logging.info("\n".join(log_str))
    return train, freeze


def main(args):
    utils.save_args(args, args.debugging_folder, file_name=mt_train.ARGS_FILE)
    logging.info('Saved args in the {0} folder'.format(args.debugging_folder))
    logging.info(mt_train.args_to_str(args))

    env_creator = TaxiGamesCreator(**vars(args))
    #RMSprop defualts: momentum=0., centered=False, weight_decay=0
    network = mt_train.create_network(args, env_creator.num_actions, env_creator.obs_shape)

    training_steps_passed = MultiTaskActorCritic.update_from_checkpoint(
        args.load_folder, network, use_cpu=args.device == 'cpu',
        ignore_layers=tuple() if args.finetuning else args.train_layers
    )

    train_params, freeze_params = split_network_parameters(network, args.train_layers, args.finetuning)
    opt = torch.optim.RMSprop([p for n, p in train_params], lr=args.initial_lr, eps=args.e)
    lr_scheduler = LinearAnnealingLR(opt, args.lr_annealing_steps)

    batch_env = ConcurrentBatchEmulator(WorkerProcess, env_creator, args.num_workers, args.num_envs)

    mt_train.set_exit_handler(
        mt_train.concurrent_emulator_handler(batch_env)
    )

    try:
        batch_env.start_workers()
        learner = MultiTaskActorCritic(
            network, opt,
            lr_scheduler,
            batch_env,
            save_folder=args.debugging_folder,
            global_step=0,
            max_global_steps=args.max_global_steps,
            rollout_steps=args.rollout_steps,
            gamma=args.gamma,
            critic_coef=args.critic_coef,
            entropy_coef=args.entropy_coef,
            clip_norm=args.clip_norm,
            term_weights=args.term_weights,
            termination_model_coef=args.termination_model_coef
        )
        learner.evaluate = lambda net: mt_train.eval_network(
            net, env_creator, 50
        )
        learner.train()
    finally:
        batch_env.close()


if __name__ == '__main__':
    args = handle_command_line(get_arg_parser())
    torch.set_num_threads(1)
    main(args)

