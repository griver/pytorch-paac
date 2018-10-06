import numpy as np
import itertools
from torch.nn import functional as F
import torch as th
from torch.distributions import Categorical
import logging, copy
from .utils import BinaryClassificationStats
from .evaluate import model_evaluation
from emulators.mazebase.taxi_tasks import TaskStats
import time

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

    env.game._choose_reset_config = choose_new_config


@model_evaluation
def save_coordinates_eval(network, env_creator, num_episodes=1, greedy=False,
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
                input('Press any button to continue..\n')

                state, reward, is_done, info = unsqueeze(env.next(act))
                mask[0] = 1.-is_done
                total_r += reward
                print('R_t: {:.2f} total_r: {:.2f}'.format(reward, total_r))
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
                input('Press any button to continue..\n')

                state, reward, is_done, info = unsqueeze(env.next(act))
                mask[0] = 1.-is_done
                total_r += reward
                print('R_t: {:.2f} total_r: {:.2f}'.format(reward, total_r))
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


class TerminationStats(dict):
    def update_stats(self, task, term_prob, steps_to_target):
        #print('#TS: << task: {}, prob: {:.5f}, steps: {}'.format(task, term_prob, steps_to_target))
        steps2probs = self.setdefault(task, {})
        steps2probs.setdefault(steps_to_target, []).append(term_prob)


from tqdm import tqdm
@model_evaluation
def doneprobs_eval(network, env_creator, num_episodes=1, greedy=False,
                termination_threshold=0.5, verbose=False, delay=2):
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
    term_stats = TerminationStats()

    def unsqueeze(emulator_outputs):
        outputs = list(emulator_outputs)
        state, info = outputs[0], outputs[-1]
        if state is not None:
            outputs[0] = state[np.newaxis]
        if info is not None:
            outputs[-1] = {k:np.asarray(v)[np.newaxis] for k, v in info.items()}
        return outputs

    episode_loop = range(num_episodes) if verbose else tqdm(range(num_episodes))
    for episode in episode_loop:
        if verbose:
            print('=============== Episode #{} ================='.format(episode))

        env = env_creator.create_environment(np.random.randint(1000))
        try:
            display = env.game.display
            task = lambda: env.game.task()
            task_name = lambda: type(env.game.task()).__name__
            steps_to_complete = lambda:task().min_steps_to_complete(env.game)

            mask = th.zeros(1).to(device)
            rnn_state = network.init_rnn_state(1)
            state, info = unsqueeze(env.reset())
            total_r = 0
            if verbose:
                print('map_size:', (env.game.height, env.game.width))
                display()
                print('current task:', task())

            for t in itertools.count():

                outputs = choose_action(network, state, info, mask, rnn_state, **extra_inputs)
                act, done_pred, rnn_state, model_info = outputs
                act = act.item()
                done_prob = model_info['done_probs'][0, 1].item()

                term_stats.update_stats(
                    task_name(),
                    done_prob,
                    steps_to_complete()
                )
                if verbose:
                    print('#{0} v_t={1:.3f} a_t={2}'.format(t, model_info['values'].item(),
                                                               env.legal_actions[act]), end=' ')
                    print('P_t(done)={0:.3f}'.format(model_info['done_probs'][0,1].item()))
                    #input('Press any button to continue..\n')
                    time.sleep(delay)

                state, reward, is_done, info = unsqueeze(env.next(act))
                mask[0] = 1.-is_done
                total_r += reward
                if verbose:
                    print('R_t: {:.2f} total_r: {:.2f}'.format(reward, total_r))
                    print('-------------Step #{}----------------'.format(t))
                    display()
                    print('current task:', task())
                if is_done: break

            episode_rewards.append(total_r)
            episode_steps.append(t + 1)
        finally:
            env.close()
        if verbose:
            time.sleep(3 * delay)

    return episode_steps, episode_rewards, term_stats


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


