import numpy as np
import itertools
import logging
import torch.nn.functional as F
import time

def model_evaluation(eval_function):
    def wrapper(network, *args, **kwargs):
        prev_mode = network.training
        network.eval() #set to the inference mode
        eval_stats = eval_function(network, *args, **kwargs)
        network.train(prev_mode)
        return eval_stats

    return wrapper

@model_evaluation
def stats_eval(network, batch_emulator, greedy=False, is_recurrent=False,
               num_episodes=None):
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
    logging.info('Evaluate stochasitc policy' if not greedy else 'Evaluate deterministic policy')

    episode_rewards, episode_steps = [], []
    terminated = np.full(num_envs, False, dtype=np.bool)
    total_r = np.zeros(num_envs, dtype=np.float32)
    num_steps = np.zeros(num_envs, dtype=np.int64)
    action_codes = np.eye(batch_emulator.num_actions)

    extra_inputs = {'greedy': greedy}
    extra_inputs['net_state'] = network.get_initial_state(num_envs) if is_recurrent else None
    states, infos = batch_emulator.reset_all()

    for t in itertools.count():
        acts, net_state = choose_action(network, states, infos, **extra_inputs)
        extra_inputs['net_state'] = net_state
        acts_one_hot = action_codes[acts.data.cpu().view(-1).numpy(),:]

        states, rewards, is_done, infos =  batch_emulator.next(acts_one_hot)
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


@model_evaluation
def visual_eval(network, env_creator, greedy=False, is_recurrent=False,
               num_episodes=1, verbose=0, delay=0.05):
    """
    Plays for num_episodes episodes on a single environment.
    Renders the process. Whether it be a separate window or string representation in the console depends on the emulator.
    Returns:
         A list that stores total reward from each episode
         A list that stores length of each episode
    """
    print('Evaluate stochasitc policy' if not greedy else 'Evaluate deterministic policy')
    episode_rewards = []
    episode_steps = []
    action_codes = np.eye(env_creator.num_actions)

    def unsqueeze(emulator_outputs):
        outputs = list(emulator_outputs)
        state, info = outputs[0], outputs[-1]
        if state is not None:
            outputs[0] = state[np.newaxis]
        if info is not None:
            outputs[-1] = {k:v[np.newaxis] for k, v in info.items()}
        return outputs

    extra_inputs = {'greedy': greedy}
    for episode in range(num_episodes):
        emulator = env_creator.create_environment(np.random.randint(100,1000)+episode)
        try:
            extra_inputs['net_state'] = network.get_initial_state(1) if is_recurrent else None
            states, infos = unsqueeze(emulator.reset())
            total_r = 0
            for t in itertools.count():
                acts, net_state = choose_action(network, states, infos, **extra_inputs)
                extra_inputs['net_state'] = net_state
                act = acts.data.cpu().view(-1).numpy()[0]

                states, reward, is_done, infos =  unsqueeze(emulator.next(action_codes[act]))
                if verbose > 0:
                    print("step#{} a_t={} r_t={}\r".format(t+1, act, reward), end="", flush=True)
                total_r += reward
                if delay: time.sleep(delay)
                if is_done: break

            if verbose > 0:
                print('Episode#{} num_steps={} total_reward={}'.format(episode + 1, t + 1, total_r))
            episode_rewards.append(total_r)
            episode_steps.append(t + 1)
        finally:
            emulator.close()

    return episode_steps, episode_rewards


def choose_action(network, states, infos, **kwargs):
    rnn_state = kwargs['net_state']
    if rnn_state is not None:
        values, a_logits, rnn_state = network(states, infos, rnn_state)
    else:
        values, a_logits = network(states, infos)

    a_probs = F.softmax(a_logits, dim=1)
    if not kwargs['greedy']:
        acts = a_probs.multinomial()
    else:
        acts = a_probs.max(1, keepdim=True)[1]
    return acts, rnn_state