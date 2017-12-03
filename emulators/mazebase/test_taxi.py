#from emulators.mazebase.multi_task_taxi import console_test_play
import numpy as np
from six.moves import xrange
from emulators import TaxiEmulator


def print_few_hot(state, encoder, cell_len=35):
    state = state.transpose((1, 2, 0))  # from [C,H,W] shape to a [H,W,C] shape
    line_sep = '\n' + '-' * (cell_len + 1) * state.shape[0]
    cell_str = '{0:^' + str(cell_len) + '}'
    legend = sorted((i, f) for f, i in encoder.feat2id.items())
    print('Legend:', legend)

    xd, yd, zd = state.shape
    # transpose first two dimentions
    for y in reversed(xrange(yd)):  # rows goes from lowest to highest
        for x in xrange(xd):
            value = ''.join(map(lambda x: str(int(x)), state[x, y]))
            print(cell_str.format(value), end='|')
        print(line_sep)

def user_action(actions):
    act = None
    while act not in actions:
        if act is not None:
            print("{0} is not a valid action! Valid actions are: {1}".format(act, actions))

        act = input('Input your action:\n')

    return act


if __name__ == '__main__':
    import train_multi_task as tr
    #args_line = '-g taxi_multi_task -d cpu -ew 1 -ec 2 ' + \
    #    "--max_global_steps 500"
    args_line = '-g taxi_game -d cpu -ew 1 -ec 2 --max_global_steps 500 -df debug_logs -m 6 6 6 6'
    print('Taxi Emulator:', TaxiEmulator.available_games())
    args = tr.get_arg_parser().parse_args(args_line.split())

    _, env_creator = tr.get_network_and_environment_creator(args)
    print('args:')
    print(tr.args_to_str(args))

    preprocess_states = env_creator.preprocess_states
    obs_shape = env_creator.obs_shape
    print('env num_actions:', env_creator.num_actions)

    envs = [env_creator.create_environment(i) for i in xrange(args.emulator_counts)]
    env = envs[0]
    if args.game == 'taxi_multi_task':
        print('All possible episode configurations:')
        for i, conf in enumerate(env.game.episode_configs):
            print(i, conf)
        n_tasks_mean =np.mean([len(tasks) for state, tasks in env.game.episode_configs])
        print('Mean number of tasks per episode:', n_tasks_mean)


    state = env.get_initial_state()
    state, info = preprocess_states(state[np.newaxis,:], obs_shape)
    print('reward:', 0, 'task:', env.game.task(), end=' ')
    print('state_shape:', state.shape, 'task_id:', info)
    env.game.display()
    print('few_hot_encoding:')
    print_few_hot(state[0], env._encoder)

    is_done = False
    action_vectors = np.eye(env_creator.num_actions)
    act2vec = {act:vec for act,vec in zip(env.legal_actions, action_vectors)}
    while True:
        a = act2vec[user_action(env.legal_actions)]
        state, r, is_done = env.next(a)
        state, info = preprocess_states(state[np.newaxis,:], obs_shape)
        print('reward:', r, 'task:', env.game.task(), end=' ')
        print('state_shape:', state.shape, 'task_id:', info)
        print('display:')
        env.game.display()
        print('few_hot_encoding')
        print_few_hot(state[0], env._encoder)

        if is_done:
            break

    print('done!!')