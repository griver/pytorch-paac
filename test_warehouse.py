import argparse
import time
import numpy as np
import torch
import utils
from train_warehouse import MultiTaskPAAC, get_network_and_environment_creator, args_to_str, eval_network, evaluate


def handle_commandline():
    devices = ['gpu', 'cpu'] if torch.cuda.is_available() else ['cpu']
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('folder', type=str, help="Folder with a trained model.")
    parser.add_argument('-tc', '--test_count', default=1, type=int, help="Number of episodes to test the model", dest="test_count")
    parser.add_argument('-g', '--greedy', action='store_true', help='Determines whether to use a stochastic or deterministic policy')
    parser.add_argument('-d', '--device', default=devices[0], type=str, choices=devices,
        help="Device to be used ('cpu' or 'gpu'). Use CUDA_VISIBLE_DEVICES to specify a particular gpu", dest="device")
    parser.add_argument('-v', '--visualize', action='store_true')
    parser.add_argument('--last', default=False, action='store_true',
                        help="If is specified then the last saved model is used otherwise the best is model")
    return parser.parse_args()


def load_trained_network(net_creator, checkpoint_path, use_cpu):
    if use_cpu:
        #it avoids loading cuda tensors in case a gpu is unavailable
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    else:
        checkpoint = torch.load(checkpoint_path)
    network = net_creator()
    network.load_state_dict(checkpoint['network_state_dict'])
    return network, checkpoint['last_step']


def fix_args_for_test(args, train_args):
    for k, v in train_args.items():
        if getattr(args, k, None) == None: #this includes cases where args.k is None
            setattr(args, k, v)

    args.max_global_steps = 0
    rng = np.random.RandomState(int(time.time()))
    args.step_delay = 0.20
    args.random_seed = rng.randint(1000)
    return args


if __name__ == '__main__':
    args = handle_commandline()
    train_args = utils.load_args(folder=args.folder,file_name='args_multi_task.json')
    args = fix_args_for_test(args, train_args)

    checkpoint_path = utils.join_path(
        args.folder, MultiTaskPAAC.CHECKPOINT_SUBDIR,
        MultiTaskPAAC.CHECKPOINT_LAST if args.last else MultiTaskPAAC.CHECKPOINT_BEST
    )
    net_creator, env_creator = get_network_and_environment_creator(args)
    network, steps_trained = load_trained_network(net_creator, checkpoint_path, args.device=='cpu')

    print([t.__name__ for t in env_creator.default_args()['task_manager']._task_types])
    use_rnn = hasattr(network, 'get_inital_state')
    print(args_to_str(args))
    print('Model was trained for {} steps'.format(steps_trained))
    if args.visualize:
        evaluate.visual_eval(network, env_creator, args.test_count,
                             greedy=args.greedy, delay=args.step_delay,
                             verbose=2)
    else:
        eval_network(network, env_creator, args.test_count, greedy=args.greedy)