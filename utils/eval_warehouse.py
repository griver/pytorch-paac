from .evaluate import model_evaluation, logging, F
from .utils import BinaryClassificationStats
import numpy as np
import itertools
import torch as th

def _to_numpy(torch_varialbe, flatten=False):
    if flatten:
        torch_variable = torch_varialbe.view(-1)
    return torch_varialbe.data.cpu().numpy()


def stats_eval(network, batch_emulator, greedy=False, num_episodes=None, task_prediction_rule=None):
    """
    Runs play with the network for num_episodes episodes.
    Returns: a tuple of
         a list that stores total reward from each episode
         a list that stores length of each episode
         a BinaryClassificationStats object
    """
    #auto_reset determines if batch_emulator automatically starts new episode when the previous ends
    #if num_episodes < batch_emulator.num_emulators then it is faster to run with auto_reset turned off.
    auto_reset = getattr(batch_emulator, 'auto_reset', True)
    num_envs = batch_emulator.num_emulators
    num_episodes = num_episodes if num_episodes else num_envs
    is_rnn = hasattr(network, 'get_initial_state')

    logging.info('Evaluate stochasitc policy' if not greedy else 'Evaluate deterministic policy')

    episode_rewards, episode_steps = [], []
    terminated = np.full(num_envs, False, dtype=np.bool)
    total_r = np.zeros(num_envs, dtype=np.float32)
    num_steps = np.zeros(num_envs, dtype=np.int64)
    action_codes = np.eye(batch_emulator.num_actions)
    termination_model_stats = BinaryClassificationStats()
    targets = []
    preds = []

    extra_inputs = {'greedy': greedy, 'prediction_rule':task_prediction_rule}
    extra_inputs['net_state'] = network.get_initial_state(num_envs) if is_rnn else None


    states, infos = batch_emulator.reset_all()
    for t in itertools.count():
        running = np.logical_not(terminated)

        acts, task_done, net_state = choose_action(network, states, infos, **extra_inputs)

        extra_inputs['net_state'] = net_state
        targets.extend(infos['task_status'][running])
        preds.extend(_to_numpy(task_done, True)[running])
        acts_one_hot = action_codes[acts.data.cpu().view(-1).numpy(),:]
        # If is_done is True then states and infos are probably contain garbage or zeros.
        # It is because Vizdoom doesn't return game variables for a finished game.
        states, rewards, is_done, infos = batch_emulator.next(acts_one_hot)

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

    termination_model_stats.add_batch(
        preds=np.array(preds),
        targets=np.array(targets)
    )
    return episode_steps, episode_rewards, termination_model_stats


@model_evaluation
def visual_eval(network, env_creator, greedy=False,
                num_episodes=1, verbos=0, delay=0.05):
    """
        Runs play with the network for num_episodes episodes on a single environment.
        Renders the process. Whether it be a separate window or string representation in the console depends on the emulator.
        Returns:
             A list that stores total reward from each episode
             A list that stores length of each episode
    """
    pass


def choose_action(network, states, infos, **kwargs):
    rnn_state = kwargs['net_state']
    if rnn_state is not None:
        values, a_logits, done_logits, rnn_state = network(states, infos, rnn_state)
    else:
        values, a_logits, done_logits = network(states, infos)

    a_probs = F.softmax(a_logits, dim=1)
    if not kwargs['greedy']: acts = a_probs.multinomial()
    else: acts = a_probs.max(1, keepdim=True)[1]

    done_probs = F.softmax(done_logits, dim=1)
    prediction_rule = kwargs.get('prediction_rule', None)
    if prediction_rule:
        done_preds = prediction_rule(done_probs)
    else:
        done_preds = done_probs.multinomial()

    return acts, done_preds, rnn_state