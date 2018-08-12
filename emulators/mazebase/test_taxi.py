#from emulators.mazebase.multi_task_taxi import console_test_play
import numpy as np
from six.moves import xrange
from itertools import count

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

    return actions.index(act)


if __name__ == '__main__':
    import train_multi_task as tr
    #args_line = '-g taxi_multi_task -d cpu -ew 1 -ec 2 ' + \
    #    "--max_global_steps 500"
    args_line = '-g taxi_multi_task -d cpu -w 1 -n 1 --max_global_steps 500 -df debug_logs -m 6 6 6 6 --arch lstm'
    print('Taxi Emulator:', tr.TaxiGamesCreator.available_games())
    args = tr.get_arg_parser().parse_args(args_line.split())

    env_creator = tr.TaxiGamesCreator(**vars(args))
    print('args:')
    print(tr.args_to_str(args))

    obs_shape = env_creator.obs_shape
    print('num_actions:', env_creator.num_actions, 'obs_shape:', obs_shape)

    env = env_creator.create_environment(17)
    if args.game == 'taxi_multi_task':
        print('All possible episode configurations:')
        for i, conf in enumerate(env.game.episode_configs):
            print(i, conf)
        n_tasks_mean =np.mean([len(tasks) for state, tasks in env.game.episode_configs])
        print('Mean number of tasks per episode:', n_tasks_mean)
        print()

    state, info = env.reset()
    print('task[0]={} task_id[0]={}, state[0]:'.format(env.game.task(), info['task_id']))
    env.game.display()
    #print('few_hot_encoding:')
    #print_few_hot(state, env._encoder)
    is_done = False
    for t in count():
        a = user_action(env.legal_actions)
        state, r, is_done, info = env.next(a)

        print('a[{0}]={1} r[{0}]={2}'.format(t,a,r))
        print('-----------------------------------------------------------')
        print('task[{0}]={1}, task_id[{0}]={2}, state[{0}]:'.format(t+1, env.game.task(), info['task_id']))
        env.game.display()

        if is_done:
            break

    print('done!!')