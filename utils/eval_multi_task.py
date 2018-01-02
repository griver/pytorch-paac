import numpy as np
import itertools, copy
from torch.nn import functional as F
import torch as th
import logging, copy
from .utils import BinaryClassificationStats

def set_user_defined_episodes(env):
    """
    Modify env so that every new episode starts with console dialog
    that asks the user to specify a new episode configuration.
    :param env: a TaxiEmulator instance to update
    :return: None
    """
    init_confs = list({init for init, tasks in env.game.episode_configs})
    tasks = list({t for _, tasks in env.game.episode_configs for t in tasks})
    task_cls = type(tasks[0])

    def choose_new_config():
        print('Select one of the following initial configurations:')
        for i, init in enumerate(init_confs):
           print(i, init)
        init_id = int(input('Input the index of chosen config:'))
        init_config = init_confs[init_id]
        print('All possible tasks:', [t.name for t in tasks])
        seq = input('Input a sequence of tasks:')
        task_seq = [task_cls[t.upper()] for t in seq.split()]
        print('Your episode configuration:')
        print('init:', init_config)
        print('tasks:', task_seq)
        return (init_config, task_seq)

    env.game._get_new_config = choose_new_config


def interactive_eval(network, env_creator, test_count, **kwargs):
    """
    In this evaluation mode you can choose initial configuration
    and list a sequence of tasks at the beginning of each episode

    :return: the list of episodes cumulative rewards
             and the list of episodes lengths
    """
    assert network.training == False, 'You should set the network to eval mode before testing!'
    greedy = kwargs.get('greedy', False)
    is_recurrent = kwargs.get('is_recurrent', False)
    termination_threshold = kwargs.get('termination_threshold', 0.5)

    import time, os
    rewards = np.zeros(test_count, dtype=np.float32)
    num_steps = np.full(test_count, float('inf'))
    action_codes = np.eye(env_creator.num_actions)
    preprocess_states = env_creator.preprocess_states

    extra_inputs = {
        'greedy': greedy,
        'termination_threshold':termination_threshold
    }
    for i in range(test_count):
        print('=============== Episode #{} ================='.format(i + 1))
        extra_inputs['rnn_inputs'] = network.get_initial_state(1) if is_recurrent else None
        env = env_creator.create_environment(i)
        display = env.game.display
        task = lambda: env.game.task()

        set_user_defined_episodes(env)

        state = env.get_initial_state()[np.newaxis, :].astype(np.uint8)
        display()
        print('current task:', task())

        for t in itertools.count():
            inputs = preprocess_states(state, env_creator.obs_shape)
            net_output = choose_action(network, *inputs, **extra_inputs)
            V, acts, done_probs, extra_inputs['rnn_inputs'], info = net_output
            act = acts.data[0, 0]
            print('#{0} V(s_t): {1:.3f} a_t: {2}'.format(t, V.data[0, 0], env.legal_actions[act]), end=' ')
            print('P(done): {0:.3f}'.format(info['done_probs'].data[0,1]))
            time.sleep(2.)
            print()
            act_one_hot = action_codes[act, :]
            s, r, is_done = env.next(act_one_hot)
            state[0] = s
            rewards[i] += r
            num_steps[i] = t + 1

            print('reward:', r)
            display()
            print('current task:', task())

            if is_done: break

    return num_steps, rewards, None


def visual_eval(network, env_creator, test_count, **kwargs):
    assert network.training == False, 'You should set the network to eval mode before testing!'
    greedy = kwargs.get('greedy', False)
    is_recurrent = kwargs.get('is_recurrent', False)
    termination_threshold = kwargs.get('termination_threshold', 0.5)

    max_local_steps = 10
    rewards = np.zeros(test_count, dtype=np.float32)
    num_steps = np.full(test_count, float('inf'))
    action_codes = np.eye(env_creator.num_actions)
    preprocess_states = env_creator.preprocess_states

    extra_inputs = {
        'greedy': greedy,
        'termination_threshold':termination_threshold
    }
    for i in range(test_count):
        print('=============== Episode #{} ================='.format(i))

        extra_inputs['rnn_inputs'] = network.get_initial_state(1) if is_recurrent else None
        env = env_creator.create_environment(i)

        display = env.game.display
        task = lambda: env.game.task()
        state = env.get_initial_state()[np.newaxis, :].astype(np.uint8)

        print('map_size:', (env.game.height, env.game.width))
        display()
        print('current task:', task())

        for t in itertools.count():
            inputs = preprocess_states(state, env_creator.obs_shape)
            net_output = choose_action(network, *inputs, **extra_inputs)
            V, acts, done_preds, extra_inputs['rnn_inputs'], info = net_output
            act = acts.data[0, 0]
            print('#{0} V(s_t)={1:.3f} a_t={2}'.format(t, V.data[0, 0], env.legal_actions[act]), end=' ')
            print('P(done)={0:.3f}'.format(info['done_probs'].data[0,1]))

            input('Press any button to continue..\n')

            act_one_hot = action_codes[act, :]
            s, r, is_done = env.next(act_one_hot)
            state[0] = s
            rewards[i] += r
            num_steps[i] = t + 1

            print('reward:', r)
            display()
            print('current task:', task())

            if is_done: break

    return num_steps, rewards, None


def stats_eval(network, env_creator, test_count, **kwargs):
    assert network.training == False, 'You should set the network to eval mode before testing!'
    greedy = kwargs.get('greedy', False)
    is_recurrent = kwargs.get('is_recurrent', False)
    termination_threshold = kwargs.get('termination_threshold', 0.5)

    envs = [env_creator.create_environment(i) for i in range(test_count)]
    preprocess_states = env_creator.preprocess_states
    Ttypes = network._intypes
    max_local_steps = 10

    terminated = th.zeros(test_count).type(Ttypes.ByteTensor)
    rewards = np.zeros(test_count, dtype=np.float32)
    num_steps = np.full(len(envs), float('inf'))
    action_codes = np.eye(env_creator.num_actions)
    #data for termination prediction model:
    targets = th.zeros((max_local_steps, test_count)).type(Ttypes.LongTensor)
    preds = th.zeros((max_local_steps, test_count)).type(Ttypes.LongTensor)
    not_done_mask = th.zeros((max_local_steps, test_count)).type(Ttypes.ByteTensor)

    extra_inputs = {
        'greedy': greedy,
        'termination_threshold':termination_threshold
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
            not_done_mask[t] = (1 - terminated) #game should be not done in the moment of prediction

            acts_one_hot = action_codes[acts.data.cpu().view(-1).numpy(), :]
            for env_id, env in enumerate(envs):
                if not terminated[env_id]:
                    s, r, is_done = env.next(acts_one_hot[env_id])
                    states[env_id] = s
                    rewards[env_id] += r
                    terminated[env_id] = is_done
                    num_steps[env_id] = T*max_local_steps + t + 1


            obs_t, task_t = preprocess_states(states, env_creator.obs_shape)

            targets_t = (task_t != old_task).astype(int)
            targets[t,:] = th.from_numpy(targets_t).type(Ttypes.LongTensor)
            preds[t, :] = done_preds.data.type(Ttypes.LongTensor)

        termination_model_stats.add_batch(
            preds=th.masked_select(preds, not_done_mask),
            targets=th.masked_select(targets, not_done_mask)
        )
        if terminated.all() : break

    return num_steps, rewards, termination_model_stats


def choose_action(network, *inputs, **kwargs):
    rnn_inputs = kwargs['rnn_inputs']
    if rnn_inputs is not None:
        values, a_logits, done_logits, rnn_state = network(*inputs, rnn_inputs=rnn_inputs)
    else:
        values, a_logits, done_logits = network(*inputs)
        rnn_state = None

    a_probs = F.softmax(a_logits, dim=1)
    done_probs = F.softmax(done_logits, dim=1)

    if not kwargs['greedy']:
        acts = a_probs.multinomial()
    else:
        acts = a_probs.max(1, keepdim=True)[1]
    if kwargs['termination_threshold'] is None:
        # multinomial returns a [batch_size, num_samples] shaped tensor
        done_preds = done_probs.multinomial()[:,0]
    else:
        done_preds = done_probs[:,1] > kwargs['termination_threshold']
    extra_info = {'done_probs':done_probs, 'act_probs':a_probs}
    return values, acts, done_preds, rnn_state, extra_info


