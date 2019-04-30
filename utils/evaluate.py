import numpy as np
import itertools
import logging
import torch
import torch.nn.functional as F
import time

def model_evaluation(num_networks=1):

    def real_decorator(eval_function):

        def wrapper(*args, **kwargs):
            networks = args[:num_networks]
            prev_modes = [net.training for net in networks]
            for net in networks:
                net.eval() #set to the inference mode

            with torch.no_grad():
                eval_stats = eval_function(*args, **kwargs)

            for net, prev_mode in zip(networks, prev_modes):
                net.train(prev_mode)

            return eval_stats

        return wrapper

    return  real_decorator

@model_evaluation()
def stats_eval(network, batch_emulator, greedy=False, num_episodes=None):
    """
    Runs play with the network for num_episodes episodes.
    Returns:
         A list that stores total reward from each episode
         A list that stores length of each episode
    """
    #auto_reset determines if batch_emulator automatically starts new episode when the previous ends
    #if num_episodes < batch_emulator.num_emulators then it is faster to run with auto_reset turned off.
    auto_reset = getattr(batch_emulator, 'auto_reset', True)
    num_envs = batch_emulator.num_emulators
    num_episodes = num_episodes if num_episodes else num_envs
    logging.info('Evaluate stochastic policy' if not greedy else 'Evaluate deterministic policy')

    episode_rewards, episode_steps = [], []
    terminated = np.full(num_envs, False, dtype=np.bool)
    total_r = np.zeros(num_envs, dtype=np.float32)
    num_steps = np.zeros(num_envs, dtype=np.int64)
    device = network._device

    rnn_state = network.init_rnn_state(num_envs)
    mask = torch.zeros(num_envs,1).to(device)
    states, infos = batch_emulator.reset_all()

    for t in itertools.count():
        acts, rnn_state = choose_action(network, states, infos, mask, rnn_state, greedy)

        states, rewards, is_done, infos =  batch_emulator.next(acts.tolist())
        mask[:,0] = torch.from_numpy(1.-is_done).to(device) #mask isn't used anywhere else, thus we can just rewrite it.

        running = np.logical_not(terminated)
        just_ended = np.logical_and(running, is_done)
        total_r[running] += rewards[running]
        num_steps[running] += 1
        episode_rewards.extend(total_r[just_ended])
        episode_steps.extend(num_steps[just_ended])
        total_r[just_ended] = 0
        num_steps[just_ended] = 0

        if len(episode_rewards) >= num_episodes: break
        if not auto_reset:
            terminated = np.logical_or(terminated, is_done)
            if all(terminated):
                states, infos = batch_emulator.reset_all()
                terminated[:] = False

    return episode_steps, episode_rewards


@model_evaluation()
def visual_eval(network, env_creator, greedy=False, num_episodes=1, verbose=0, delay=0.05,
                **env_kwargs):
    """
    Plays for num_episodes episodes on a single environment.
    Renders the process. Whether it be a separate window or string representation in the console depends on the emulator.
    Returns:
         A list that stores total reward from each episode
         A list that stores length of each episode
    """
    episode_rewards = []
    episode_steps = []
    logging.info('Evaluate stochastic policy' if not greedy else 'Evaluate deterministic policy')
    device = network._device

    def unsqueeze(emulator_outputs):
        outputs = list(emulator_outputs)
        state, info = outputs[0], outputs[-1]
        if state is not None:
            outputs[0] = state[np.newaxis]
        if info is not None:
            outputs[-1] = {k:v[np.newaxis] for k, v in info.items()}
        return outputs

    for episode in range(num_episodes):
        emulator = env_creator.create_environment(np.random.randint(100,1000)+episode, **env_kwargs)
        try:
            mask = torch.zeros(1).to(device)
            rnn_state = network.init_rnn_state(1)

            states, infos = unsqueeze(emulator.reset())
            total_r = 0

            for t in itertools.count():
                acts, rnn_state = choose_action(network, states, infos, mask, rnn_state, greedy)
                act = acts[0].item()

                states, reward, is_done, infos = unsqueeze(emulator.next(act))
                mask[0] = 1. - is_done
                if verbose > 0:
                    print("step#{} a_t={} r_t={}\r".format(t+1, act, reward), end="", flush=True)
                total_r += reward
                if delay: time.sleep(delay)
                if is_done: break

            if verbose > 0:
                print('Episode#{} total_steps={} score={}'.format(episode + 1, t + 1, total_r))
            episode_rewards.append(total_r)
            episode_steps.append(t + 1)
        finally:
            emulator.close()

    return episode_steps, episode_rewards


def choose_action(network, states, infos, masks, rnn_state, greedy=False):
    values, distr, rnn_state = network(states, infos, masks, rnn_state)
    acts = distr.probs.argmax(dim=1) if greedy else distr.sample()
    return acts, rnn_state