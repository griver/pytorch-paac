import argparse
import itertools
import time

import numpy as np
import torch
import torch.nn.functional as F

import utils
from paac import PAACLearner
from train import get_network_and_environment_creator


def get_save_frame(name):
    import imageio
    writer = imageio.get_writer(name + '.gif', fps=30)

    def save_frame(frame):
        writer.append_data(frame)

    return save_frame


def add_gif_processor(envs):

    for i, environment in enumerate(envs):
        path = utils.join_path(args.gif_folder, args.gif_name + str(i))
        environment.on_new_frame = get_save_frame()


def print_dict(d, name=None):
    title = ' '.join(['=='*10, '{}','=='*10])
    if name is not None:
        title.format(name)

    print(title)
    for k in sorted(d.keys()):
        print('  ', k,':', d[k])
    print('='*len(title))


def play(network, envs, args, is_recurrent=False):
    terminated = np.full(len(envs), False, dtype=np.bool)
    rewards = np.zeros(args.test_count, dtype=np.float32)
    num_steps = np.full(len(envs), float('inf'))
    action_codes = np.eye(args.num_actions)
    noop = np.array([a == envs[0].get_noop() for a in envs[0].legal_actions])

    if is_recurrent:
      rnn_init = network.get_initial_state(len(envs))
      extra_states = rnn_init
    else:
      extra_states = None

    states = [env.get_initial_state() for env in envs]
    states = np.array(states)

    for i, env in enumerate(envs):
        for _ in range(np.random.randint(args.noops+1)):
            states[i], _, _ = env.next(noop)

    print('Use stochasitc policy' if not args.greedy else 'Use deterministic policy')

    for t in itertools.count():
        acts, extra_states = choose_action(network, states, extra_states, greedy=args.greedy)
        acts_one_hot = action_codes[acts.data.cpu().view(-1).numpy(),:]

        for env_id, env in enumerate(envs):
            if not terminated[env_id]:
                s, r, is_done= env.next(acts_one_hot[env_id])
                states[env_id] = s
                rewards[env_id] += r
                terminated[env_id] = is_done
                num_steps[env_id] = t+1
        if all(terminated): break



    return num_steps, rewards


def choose_action(network, state, extra_input=None, greedy=False):
    if extra_input is not None:
        _, a_logits, extra_output = network((state, extra_input))
    else:
      _, a_logits = network(state)
      extra_output = None

    a_probs = F.softmax(a_logits, dim=1)
    if not greedy:
      acts = a_probs.multinomial()
    else:
      _, acts = a_probs.max(1, keepdim=True)
    return acts, extra_output

def fix_args_for_test(args, train_args):
    for k, v in train_args.items():
        if not hasattr(args, k):
            setattr(args, k, v)

    args.max_global_steps = 0
    args.debugging_folder = '/tmp/logs'
    args.random_start = False
    args.single_life_episodes = False
    if args.gif_name:
      args.visualize = 1
    rng = np.random.RandomState(int(time.time()))
    args.random_seed = rng.randint(1000)

    return args

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--folder', type=str, help="Folder where to save the debugging information.", dest="folder", required=True)
    parser.add_argument('-rf', '--resource_folder', default='./resources/atari_roms',
        help='Directory with files required for the game initialization(i.e. binaries for ALE and scripts for ViZDoom)', dest="resource_folder")
    parser.add_argument('-tc', '--test_count', default='1', type=int, help="The amount of tests to run on the given network", dest="test_count")
    parser.add_argument('-np', '--noops', default=30, type=int, help="Maximum amount of no-ops to use", dest="noops")
    parser.add_argument('-gn', '--gif_name', default=None, type=str, help="If provided, a gif will be produced and stored with this name", dest="gif_name")
    parser.add_argument('-gf', '--gif_folder', default='gifs/', type=str, help="The folder where to save gifs.", dest="gif_folder")
    parser.add_argument('-g', '--greedy', action='store_true', help='Determines whether to use a stochastic or deterministic policy')
    parser.add_argument('-d', '--device', default='gpu', type=str, choices=['gpu', 'cpu'],
        help="Device to be used ('cpu' or 'gpu'). Use CUDA_VISIBLE_DEVICES to specify a particular gpu", dest="device")

    args = parser.parse_args()
    train_args = utils.load_args(folder=args.folder)
    args = fix_args_for_test(args, train_args)

    checkpoint_path = utils.join_path(
        args.folder, PAACLearner.CHECKPOINT_SUBDIR, PAACLearner.CHECKPOINT_LAST
    )
    checkpoint = torch.load(checkpoint_path)
    net_creator, env_creator = get_network_and_environment_creator(args)
    network = net_creator()
    network.load_state_dict(checkpoint['network_state_dict'])
    network.eval()

    envs = [env_creator.create_environment(i) for i in range(args.test_count)]
    use_lstm = (args.arch == 'lstm')
    if args.gif_name is not None:
        gif_path = utils.join_path(args.gif_folder, args.gif_name)
        gif_path += '#{0}'
        utils.ensure_dir(gif_path)
        for i, env in enumerate(envs):
            env.on_new_frame = get_save_frame(gif_path.format(i))

    print_dict(vars(args), 'ARGS')
    print('Model was trained for {} steps'.format(checkpoint['last_step']))
    num_steps, rewards = play(network, envs, args, is_recurrent=use_lstm)
    print('Perfromed {0} tests for {1}.'.format(args.test_count, args.game))
    print('Mean number of steps: {0:.3f}'.format(np.mean(num_steps)))
    print('Mean R: {0:.2f}'.format(np.mean(rewards)), end=' | ')
    print('Max R: {0:.2f}'.format(np.max(rewards)), end=' | ')
    print('Min R: {0:.2f}'.format(np.min(rewards)), end=' | ')
    print('Std of R: {0:.2f}'.format(np.std(rewards)))
