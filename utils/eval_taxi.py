import numpy as np
import itertools
from torch.nn import functional as F
import torch as th
from torch.distributions import Categorical
import logging, copy
from .utils import BinaryClassificationStats
from .evaluate import model_evaluation
from emulators.mazebase.taxi_tasks import TaskStats

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

    env.game._get_reset_config = choose_new_config


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

        state = env.reset()[np.newaxis, :].astype(np.uint8)
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


@model_evaluation
def visual_eval(network, env_creator, num_episodes=1, greedy=False,
                termination_threshold=0.5):
    """
    Plays for num_episodes episodes on a single environment.
    Renders the process. Whether it be a separate window or string representation in the console depends on the emulator.
    Returns:
         A list that stores total reward from each episode
         A list that stores length of each episode
    """
    assert network.training == False, 'You should set the network to eval mode before testing!'

    episode_rewards = []
    episode_steps = []
    logging.info('Evaluate stochastic policy' if not greedy else 'Evaluate deterministic policy')
    device = network._device

    extra_inputs = {
        'greedy': greedy,
        'termination_threshold':termination_threshold
    }

    def unsqueeze(emulator_outputs):
        outputs = list(emulator_outputs)
        state, info = outputs[0], outputs[-1]
        if state is not None:
            outputs[0] = state[np.newaxis]
        if info is not None:
            outputs[-1] = {k:np.asarray(v)[np.newaxis] for k, v in info.items()}
        return outputs

    for episode in range(num_episodes):
        if 'env' in locals():
            env.reset()
        print('=============== Episode #{} ================='.format(episode))
        env = env_creator.create_environment(np.random.randint(1000))
        try:
            display = env.game.display
            task = lambda: env.game.task()

            mask = th.zeros(1).to(device)
            rnn_state = network.init_rnn_state(1)
            state, info = unsqueeze(env.reset())
            total_r = 0

            print('map_size:', (env.game.height, env.game.width))
            display()
            print('current task:', task())

            for t in itertools.count():

                outputs = choose_action(network, state, info, mask, rnn_state, **extra_inputs)
                act, done_pred, rnn_state, model_info = outputs
                act = act.item()

                print('#{0} v_t={1:.3f} a_t={2}'.format(t, model_info['values'].item(),
                                                           env.legal_actions[act]), end=' ')
                print('P_t(done)={0:.3f}'.format(model_info['done_probs'][0,1].item()))
                #input('Press any button to continue..\n')

                state, reward, is_done, info = unsqueeze(env.next(act))
                mask[0] = 1.-is_done
                print('R_t: {:.2f} total_r: {:.2f}'.format(reward, total_r))
                total_r += reward
                print('-------------Step #{}----------------'.format(t))
                display()
                print('current task:', task())
                if is_done: break

            episode_rewards.append(total_r)
            episode_steps.append(t + 1)
        finally:
            env.close()

    return episode_steps, episode_rewards, None


@model_evaluation
def stats_eval(network, batch_emulator, num_episodes=None, greedy=False, termination_threshold=0.5):
    assert network.training == False, 'You should set the network to eval mode before testing!'

    auto_reset = getattr(batch_emulator, 'auto_reset', True)
    assert auto_reset is False, 'The way we collect statistics about subtasks will not work with emulator that automatically resets episodes!'
    device = network._device
    num_envs = batch_emulator.num_emulators
    num_episodes = num_episodes if num_episodes else num_envs
    logging.info('Evaluate stochastic policy' if not greedy else 'Evaluate deterministic policy')

    episode_rewards, episode_steps = [], []
    terminated = np.full(num_envs, False, dtype=np.bool)
    total_r = np.zeros(num_envs, dtype=np.float32)
    num_steps = np.zeros(num_envs, dtype=np.int64)
    device = network._device

    rnn_state = network.init_rnn_state(num_envs)
    mask = th.zeros(num_envs,1).to(device)
    state, info = batch_emulator.reset_all()

    extra_inputs = {
        'greedy': greedy,
        'termination_threshold':termination_threshold
    }
    termination_model_stats = BinaryClassificationStats()

    for t in itertools.count():
        outputs = choose_action(network, state, info, mask, rnn_state, **extra_inputs)
        acts, done_preds, rnn_state, _ = outputs

        state, reward, is_done, info =  batch_emulator.next(acts)

        mask[:,0] = th.from_numpy(1.-is_done).to(device) #mask isn't used anywhere else, thus we can just rewrite it.
        #determine emulators states and collect stats about episodes' rewards and lengths:
        running = np.logical_not(terminated)
        just_ended = np.logical_and(running, is_done)
        total_r[running] += reward[running]
        num_steps[running] += 1
        episode_rewards.extend(total_r[just_ended])
        episode_steps.extend(num_steps[just_ended])
        total_r[just_ended] = 0
        num_steps[just_ended] = 0

        #collect stats about predictions of tasks termination:
        pred_mask = np.logical_and(running, np.logical_not(just_ended))
        masked_done_preds = done_preds.cpu().numpy()[pred_mask]
        masked_done_targets = info['task_status'][pred_mask]
        masked_done_targets[masked_done_targets != 1] = 0 # merge Fail(2) and Running(0) statuses
        termination_model_stats.add_batch(masked_done_preds, masked_done_targets)


        if len(episode_rewards) >= num_episodes: break
        if not auto_reset:
            terminated = np.logical_or(terminated, is_done)
            if all(terminated):
                states, infos = batch_emulator.reset_all()
                terminated[:] = False

    #this won't work correctly with emulators that autreset finished episodes:
    tasks_stats = TaskStats()
    histories = batch_emulator.call_method(
        'get_tasks_history',
        [([],{}) for _ in range(num_envs)])
    for h in histories:
        tasks_stats.add_task_history(h)
    extra_stats = {'done_pred_stats':termination_model_stats, 'task_stats':tasks_stats}
    return episode_steps, episode_rewards, extra_stats


def choose_action(network, state, info, mask, rnn_state, **kwargs):
    value, act_distr, done_logits, rnn_state = network(state, info, mask, rnn_state)
    done_distr = Categorical(logits=done_logits)
    greedy = kwargs.get('greedy', False)
    done_pred_threshold = kwargs.get('termination_threshold')

    acts = act_distr.probs.argmax(dim=1) if greedy else act_distr.sample()
    if done_pred_threshold:
        done_preds = done_distr.probs[:,1].ge(done_pred_threshold)
    else:
        done_preds = done_distr.sample()

    extra_model_info = {
        'done_probs':done_distr.probs,
        'act_probs':act_distr.probs,
        'values':value,
    }
    return acts, done_preds, rnn_state, extra_model_info


