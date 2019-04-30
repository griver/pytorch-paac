#from emulators.mazebase.multi_task_taxi import console_test_play
import numpy as np
from six.moves import xrange
from itertools import count

KEYS_2_ACTIONS={
    'w':'up',
    'a':'left',
    's':'down',
    'd':'right',
    'e':'pickup',
    'r':'dropoff',
    'p':'pass'
}
ALLOWED_KEYS = frozenset(KEYS_2_ACTIONS.keys())


def print_few_hot(state, encoder, cell_len=35):
    state = state.transpose((1, 2, 0))  # from [C,H,W] shape to a [H,W,C] shape
    line_sep = '\n' + '-' * (cell_len + 1) * state.shape[0]
    cell_str = '{0:^' + str(cell_len) + '}'
    legend = sorted((i, f) for f, i in encoder.feat2id.items())

    print('Legend:', ' '.join('{}:{} |'.format(i,s) for i,s in legend))

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
    #args_line = '-g taxi -d cpu -ew 1 -ec 2 ' + \
    #    "--max_global_steps 500"
    args_line = '-g taxi_plus -d cpu -w 1 -n 1 --max_global_steps 300 ' \
                '-sf debug_logs -m 12 12 --arch lstm --random_seed 17 -fr -0.8 ' \
                '--view_size 5 --max_episode_steps 40 -t find_p pickup convey_p find_c pickup_c convey_c --random_seed 17'\

    print('Taxi Emulator:', tr.TaxiGamesCreator.available_games())
    args = tr.handle_command_line(tr.get_arg_parser(), args_line)

    env_creator = tr.TaxiGamesCreator(**vars(args))
    print('main_args:')
    print(tr.args_to_str(args))

    obs_shape = env_creator.obs_shape
    print('num_actions:', env_creator.num_actions, 'obs_shape:', obs_shape)

    env = env_creator.create_environment(17, visualize=False, wait_key=True)
    for i in range(3):
        print('================= Episode #{} ===================='.format(i+1))
        state, info = env.reset()
        env.display()
        #print('few_hot_encoding:')
        #print_few_hot(state, env._encoder)
        is_done = False
        total_r = 0
        for t in count():
            key = env.from_keyboard(ALLOWED_KEYS)
            a = env.legal_actions.index(KEYS_2_ACTIONS[key])
            #a = user_action(env.legal_actions)
            state, r, is_done, info = env.next(a)
            total_r += r
            print('a[{0}]={1} r[{0}]={2:.2f}'.format(t,KEYS_2_ACTIONS[key], r))
            #print('-----------------------------------------------------------')
            #print('task[{0}]={1} | info[{0}]={2} | state[{0}]:'.format(t+1, env.game.task(), env.game.agent.location))
            env.display()
            # print('few_hot_encoding:')
            # print_few_hot(state, env._encoder)

            if is_done:
                break
        if env.map_viewer:
            env.map_viewer.save_episode('taxi_plus_12x12_ep{}.gif'.format(i + 1))

        print('done!!')
        print('total_steps:', t+1, 'score:', env.game.reward_so_far(), total_r)