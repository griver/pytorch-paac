from .evaluate import model_evaluation, logging, F
from .utils import BinaryClassificationStats
import numpy as np
import itertools
import warnings
import pandas as pd
import emulators.warehouse.warehouse_tasks as tasks

def _to_numpy(torch_variable, flatten=False):
    if flatten:
        torch_variable = torch_variable.view(-1)
    return torch_variable.data.cpu().numpy()


@model_evaluation
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
    task_stats = TaskStats('property','n_steps')
    targets = []
    preds = []

    extra_inputs = {'greedy': greedy, 'prediction_rule':task_prediction_rule}
    extra_inputs['net_state'] = network.get_initial_state(num_envs) if is_rnn else None


    states, infos = batch_emulator.reset_all()
    for t in itertools.count():
        running = np.logical_not(terminated)

        acts, task_done, net_state = choose_action(network, states, infos, **extra_inputs)
        termination_model_stats.add_batch( #need to squeese targets down to 2 labels
            preds = _to_numpy(task_done, True)[running],
            targets = infos['task_status'][running],
        )
        extra_inputs['net_state'] = net_state
        acts_one_hot = action_codes[acts.data.cpu().view(-1).numpy(),:]
        # If is_done is True then states and infos are probably contain garbage or zeros.
        # It is because Vizdoom doesn't return game variables for a finished game.
        states, rewards, is_done, infos = batch_emulator.next(acts_one_hot)
        task_stats.add_stats(is_done, **infos)

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

    return episode_steps, episode_rewards, termination_model_stats, task_stats


def choose_action(network, states, infos, **kwargs):
    rnn_state = kwargs['net_state']
    if rnn_state is not None:
        values, a_logits, done_logits, rnn_state = network(states, infos, rnn_state)
    else:
        values, a_logits, done_logits = network(states, infos)

    a_probs = F.softmax(a_logits, dim=1)
    if not kwargs['greedy']: acts = a_probs.multinomial(1)
    else: acts = a_probs.max(1, keepdim=True)[1]

    done_probs = F.softmax(done_logits, dim=1)
    prediction_rule = kwargs.get('prediction_rule', None)
    if prediction_rule:
        done_preds = prediction_rule(done_probs)
    else:
        done_preds = done_probs.multinomial(1)

    return acts, done_preds, rnn_state


class TaskStatisticsError(ValueError):
    pass


class TaskStats(pd.DataFrame):
    task_id2name = {cls.task_id:cls.__name__ for cls in tasks.WarehouseTask.__subclasses__()}
    def __init__(self, *extra_properties):
        """
        :param extra_properties: names of task properties you want to store
         aside from task_id and task's termination status.
        """
        super(TaskStats, self).__init__(columns=('task_id', 'status') + extra_properties)
        self._extra_columns = extra_properties

    def add_stats(self, episode_is_done, task_status, task_id, **task_properties):
        self._check_new_data(task_status, task_id, **task_properties)
        failed = (task_status == tasks.TaskStatus.FAIL)
        completed = (task_status == tasks.TaskStatus.SUCCESS)
        task_done = np.logical_or(failed, completed, episode_is_done)

        idx = len(self)

        for i, task_terminated in enumerate(task_done):
            if not task_terminated: continue
            d = dict(task_id=task_id[i], status=task_status[i])
            d.update({k:task_properties[k][i] for k in self._extra_columns})
            self.loc[idx] = d

    def _check_new_data(self, task_status, task_id, **task_properties):
        required = set(self._extra_columns)
        received = set(k for k in task_properties.keys())
        shortage = required.difference(received)
        if shortage:
            raise TaskStatisticsError("Expected values for {} columns but did'n get them!")
        excess = received.difference(required)
        if excess:
            warnings.warn("Received data for unspecified columns {}. The data will be discarded.", Warning)

    def report_str(self):
        success = tasks.TaskStatus.SUCCESS
        lines = []
        for i, name in sorted(self.task_id2name.items()):
            task_i = self[self['taks_id'] == i]
            total_i = len(task_i)
            succ_i = (task_i['status'] == success).mean() if total_i else 0.
            succ_i *= 100. #make %
            lines.append('{}: {:.2f}%({})'.format(name, succ_i, total_i))
        return '\n'.join(lines)