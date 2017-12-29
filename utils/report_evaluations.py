import copy, logging, itertools
import numpy as np
import torch as th
from .utils import  BinaryClassificationStats
from .eval_multi_task import choose_action, set_user_defined_episodes


def create_custom_config(game_cls, taxi_passenger_relation, task_seq):
    """
    Builds a proper episode config from taxi_passenger_relation and task_seq
    if one of the arguments is not specified the function creates a console dialog for the config specification.
    """
    if taxi_passenger_relation is None:
        print('Select one of the following passenger-taxi placement configurations: FAR, NEAR, INSIDE')
        pRt = input('Confguration:')
    pRt = game_cls.ItemRelation[pRt.upper()]
    far = game_cls.ItemRelation['FAR']
    init_state = game_cls.InitState(pRt=pRt, pRd=far, tRd=far)

    if task_seq is None:
        print('All possible tasks:', [t.name for t in game_cls.Tasks])
        task_seq = input('Input a sequence of tasks:')
    task_seq = [game_cls.Tasks[t.upper()] for t in task_seq.split()]

    return (init_state, task_seq)


def custom_task_eval(network, env_creator, test_count, **kwargs):
    """
    Evaluates the network on the taxi_multi_task environment with a specified episode configuration.
    To specify an episode config one should pass additional named args:
      taxi_passenger_relation - one of the possible values: "inside", "near", "far",
      tasks - a sequence of tasks names.
    if one of the arguments is not specified the function creates a console dialog for the config specification.
    """
    assert network.training == False, 'You should set the network to eval mode before testing!'
    greedy = kwargs.get('greedy', False)
    is_recurrent = kwargs.get('is_recurrent', False)
    termination_threshold = kwargs.get('termination_threshold', 0.5)
    taxi_passenger_relation = kwargs.get('taxi_passenger_relation', None)
    tasks = kwargs.get('tasks', None)

    envs = [env_creator.create_environment(i) for i in range(test_count)]

    preprocess_states = env_creator.preprocess_states
    Ttypes = network._intypes
    max_local_steps = 10

    custom_conf = create_custom_config(envs[0].game, taxi_passenger_relation, tasks)
    for env in envs:
        env.game.episode_configs = [copy.copy(custom_conf)]

    terminated = th.zeros(test_count).type(Ttypes.ByteTensor)
    rewards = np.zeros(test_count, dtype=np.float32)
    num_steps = np.full(len(envs), float('inf'))
    action_codes = np.eye(env_creator.num_actions)
    # data for termination prediction model:
    targets = th.zeros((max_local_steps, test_count)).type(Ttypes.LongTensor)
    preds = th.zeros((max_local_steps, test_count)).type(Ttypes.LongTensor)
    not_done_mask = th.zeros((max_local_steps, test_count)).type(Ttypes.ByteTensor)

    extra_inputs = {
        'greedy': greedy,
        'termination_threshold': termination_threshold
    }
    if is_recurrent:
        rnn_init = network.get_initial_state(len(envs))
        extra_inputs['rnn_inputs'] = rnn_init
    else:
        extra_inputs['rnn_inputs'] = None

    logging.info('Use stochasitc policy' if not greedy else 'Use deterministic policy')

    termination_model_stats = BinaryClassificationStats()
    states = [env.get_initial_state() for env in envs]
    states = np.array(states, dtype=np.uint8)
    obs_t, task_t = preprocess_states(states, env_creator.obs_shape)

    for T in itertools.count():
        for t in range(max_local_steps):
            net_output = choose_action(network, obs_t, task_t, **extra_inputs)
            vals, acts, done_preds, extra_inputs['rnn_inputs'], _ = net_output
            old_task = task_t.copy()
            not_done_mask[t] = (1 - terminated)  # game should be not done in the moment of prediction

            acts_one_hot = action_codes[acts.data.cpu().view(-1).numpy(), :]
            for env_id, env in enumerate(envs):
                if not terminated[env_id]:
                    s, r, is_done = env.next(acts_one_hot[env_id])
                    states[env_id] = s
                    rewards[env_id] += r
                    terminated[env_id] = is_done
                    num_steps[env_id] = T * max_local_steps + t + 1

            obs_t, task_t = preprocess_states(states, env_creator.obs_shape)

            targets_t = (task_t != old_task).astype(int)
            targets[t, :] = th.from_numpy(targets_t).type(Ttypes.LongTensor)
            preds[t, :] = done_preds.data.type(Ttypes.LongTensor)

        termination_model_stats.add_batch(
            preds=th.masked_select(preds, not_done_mask),
            targets=th.masked_select(targets, not_done_mask)
        )
        if terminated.all(): break

    return num_steps, rewards, termination_model_stats



class TaskLenStats(dict):
    def update_stats(self, episode_tasks):
        for i, (k, g) in enumerate(itertools.groupby(episode_tasks)):
            num_steps = len(list(g))
            self.setdefault((i+1,k), []).append(num_steps)

    def pretty_print(self):
        print('TaskLenStats:')
        max_i = max(i for (i,task) in self.keys())
        tasks = set(task for (i,task) in self.keys())
        for t in sorted(tasks, key=lambda t: t.name):
            print('=========== {} ========='.format(t.name))
            for i in range(0, max_i):
                steps = self.get((i+1, t), None)
                if steps:
                    print('{}:{} '.format(i+1,np.mean(steps)), end='')
            print()



def fixed_episode_eval(network, env_creator, test_count, **kwargs):

    assert network.training == False, 'You should set the network to eval mode before testing!'
    greedy = kwargs.get('greedy', False)
    is_recurrent = kwargs.get('is_recurrent', False)
    termination_threshold = kwargs.get('termination_threshold', 0.5)
    taxi_passenger_relation = kwargs.get('taxi_passenger_relation', None)
    tasks = kwargs.get('tasks', None)
    verbose = kwargs.get('verbose', False)
    repeat_episode = kwargs.get('repeat_episode', True)

    env = env_creator.create_environment(2017)
    assert hasattr(env.game, 'repeat_episode'), 'the game should be able to repeat the same episode!'
    custom_conf = create_custom_config(env.game, taxi_passenger_relation, tasks)
    env.game.episode_configs = [copy.copy(custom_conf)]
    env.get_initial_state()
    env.game.repeat_episode = repeat_episode
    if repeat_episode:
        env.game.display()

    rewards = np.zeros(test_count, dtype=np.float32)
    num_steps = np.full(test_count, float('inf'))
    action_codes = np.eye(env_creator.num_actions)
    preprocess_states = env_creator.preprocess_states
    display = env.game.display
    task = lambda: env.game.task()
    tasklen_stats = TaskLenStats()

    extra_inputs = {
      'greedy': greedy,
      'termination_threshold': termination_threshold
    }

    msg = '#{0} V(s_t): {1:.3f} a_t: {2} P(done) {0:.3f}'
    logging.info('Use stochasitc policy' if not greedy else 'Use deterministic policy')
    for i in range(test_count):
        episode_tasks = []

        extra_inputs['rnn_inputs'] = network.get_initial_state(1) if is_recurrent else None
        state = env.get_initial_state()[np.newaxis, :].astype(np.uint8)
        if verbose:
            print('=============== Episode #{} ================='.format(i + 1))
            display()
            print('current task:', task())

        for t in itertools.count():
            episode_tasks.append(task())
            inputs = preprocess_states(state, env_creator.obs_shape)
            net_output = choose_action(network, *inputs, **extra_inputs)
            V, acts, done_preds, extra_inputs['rnn_inputs'], info = net_output
            act = acts.data[0, 0]

            if verbose:
                act_name = env.legal_actions[act]
                done_prob = info['done_probs'].data[0, 1]
                print(msg.format(t, V.data[0, 0], act_name, done_prob))
                input('Press any button to continue..\n')

            act_one_hot = action_codes[act, :]
            s, r, is_done = env.next(act_one_hot)
            state[0] = s
            rewards[i] += r
            num_steps[i] = t + 1

            if verbose:
                print('reward:', r)
                display()
                print('current task:', task())

            if is_done: break

        tasklen_stats.update_stats(episode_tasks)

    return num_steps, rewards, tasklen_stats