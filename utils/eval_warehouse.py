from .evaluate import model_evaluation, logging, F, time
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
         a TaskStats object
    """
    #auto_reset determines if batch_emulator automatically starts new episode when the previous ends
    #if num_episodes < batch_emulator.num_emulators then it is faster to run with auto_reset turned off.
    auto_reset = getattr(batch_emulator, 'auto_reset', True)
    if auto_reset:
        raise ValueError("stats_eval doesn't with batch emulators that reset episodes automatically!")
        # To collect task statistics we use completed_tas
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

    extra_inputs = {'greedy': greedy, 'prediction_rule':task_prediction_rule}
    extra_inputs['net_state'] = network.get_initial_state(num_envs) if is_rnn else None

    #print('================= Eval reset: ===========================')
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
        just_ended = np.logical_and(running, is_done)
        task_stats.add_stats(just_ended, **infos) #only episode 

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


@model_evaluation
def visual_eval(network, env_creator, num_episodes=1, greedy=False, verbose=0, delay=0.05, task_prediction_rule=None):
    episode_rewards = []
    episode_steps = []
    is_rnn = hasattr(network, 'get_initial_state')
    logging.info('Evaluate stochasitc policy' if not greedy else 'Evaluate deterministic policy')
    action_codes = np.eye(env_creator.num_actions)
    termination_model_stats = BinaryClassificationStats()
    extra_inputs = {'greedy':greedy, 'prediction_rule':task_prediction_rule}

    def unsqueeze(emulator_outputs):
        outputs = list(emulator_outputs)
        state, info = outputs[0], outputs[-1]
        if state is not None:
            outputs[0] = state[np.newaxis]
        if info is not None:
            outputs[-1] = {k:np.asarray(v)[np.newaxis] for k, v in info.items()}
        return outputs

    for episode in range(num_episodes):
        emulator = env_creator.create_environment(np.random.randint(100,1000)+episode)
        try:
            extra_inputs['net_state'] = network.get_initial_state(1) if is_rnn else None
            states, infos = unsqueeze(emulator.reset())
            total_r = 0
            prev_task = None
            print('==='*15)
            for t in itertools.count():
                acts, task_done, net_state = choose_action(network, states, infos, **extra_inputs)
                extra_inputs['net_state'] = net_state
                act = acts.data.cpu().view(-1).numpy()[0]
                em_output = emulator.next(action_codes[act])
                states, reward, is_done, info = unsqueeze(em_output)
                if verbose > 0:
                    print_emulator_info(emulator, em_output[-1], prev_task)

                total_r += reward
                if delay: time.sleep(delay)
                if is_done: break
                prev_task = emulator.task

            if verbose > 0:
                print('\r#{}: {} n_steps={}'.format(
                    len(emulator._completed), task_str(emulator.task, emulator), prev_task.n_steps))
                print('Episode#{} num_steps={} total_reward={}'.format(episode + 1, t + 1, total_r))
            episode_rewards.append(total_r)
            episode_steps.append(t + 1)
        finally:
            emulator.close()

    return episode_steps, episode_rewards


def print_emulator_info(emulator, info, prev_task):
    st_info = emulator._state_info
    room_id = st_info.room_id
    items = st_info.item_count
    task = emulator.task
    if prev_task and prev_task != task:
        print('\r#{}: {} n_steps={}'.format(
            len(emulator._completed), task_str(prev_task, emulator),prev_task.n_steps))

    print('\rtask={} room_id={}, num_items={} info={}'.format(
        task_str(task, emulator), room_id, items, info), end='', flush=True)


def task_str(task, emulator):
    info = emulator._map_info
    if isinstance(task, tasks.Visit):
        if task.target_id == info['entry_room'].id:
            return '[{}] Visit Entry room!'.format(task.status._name_)
        return '[{}] Visit room with {}'.format(
            task.status._name_, info['textures'][task.property][1])
    elif isinstance(task, tasks.CarryItem):
        if task.target_id == info['entry_room'].id:
            return '[{}] Bring the item in the entry room!'.format(task.status._name_)
        return '[{}] Bring the item in the room with {}'.format(
            task.status._name_, info['textures'][task.property][1])
    elif isinstance(task, tasks.Drop):
        return "[{}] Drop the carried item!".format(task.status._name_)
    elif isinstance(task, tasks.PickUp):
        return "[{}] Pick up a {}".format(
            task.status._name_, info['items'][task.target_id])


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
    # temporary properties
    _internal_names = pd.DataFrame._internal_names + ['_extra_properties']
    _internal_names_set = set(_internal_names)
    task_id2name = {cls.task_id:cls.__name__ for cls in tasks.WarehouseTask.__subclasses__()}

    def __init__(self, *extra_properties):
        """
        :param extra_properties: names of task properties you want to store
         aside from task_id and task's termination status.
        """
        super(TaskStats, self).__init__(columns=('task_id', 'status') + extra_properties)
        self._extra_properties = extra_properties


    def add_stats(self, episode_is_done, task_status, task_id, **task_properties):
        self._check_new_data(task_status, task_id, **task_properties)
        failed = (task_status == tasks.TaskStatus.FAIL)
        completed = (task_status == tasks.TaskStatus.SUCCESS)
        task_done = np.logical_or(episode_is_done, np.logical_or(failed, completed))

        for i, task_terminated in enumerate(task_done):
            if not task_terminated: continue

            d = dict(task_id=task_id[i], status=task_status[i])
            d.update({k:task_properties[k][i] for k in self._extra_properties})
            idx = len(self)
            self.loc[idx] = d

    def _check_new_data(self, task_status, task_id, **task_properties):
        required = set(self._extra_properties)
        received = set(k for k in task_properties.keys())
        shortage = required.difference(received)
        if shortage:
            raise TaskStatisticsError("Expected values for {} columns but did'n get them!".format(shortage))
        excess = received.difference(required)
        if excess:
            warnings.warn("Received data for unspecified columns {}. The data will be discarded.".format(excess), Warning)

    def pretty_repr(self):
        lines = [self.pretty_task_repr(id) for id, name in sorted(self.task_id2name.items())]
        return '\n'.join(lines)

    def pretty_task_repr(self, task_id):
        '''
        Returns a string representation of the data about an agent's performance one the specified task.
        Example of a returned string:
          'PickUpPassenger(20): succ=60.00% fail=35.00% runn=5.00%'
        '''
        data = self[self['task_id'] == task_id]
        total_num = len(data)
        counts = data['status'].value_counts()
        dur = data['n_steps'].mean() if total_num else 0
        strings = ['{:<9}(total={}, dur={:.0f}):'.format(self.task_id2name[task_id], total_num, dur)]
        if total_num:
            for status in tasks.TaskStatus:
                name = status.name[:4].lower()
                val = status.value
                num = counts[val] if val in counts else 0
                percent = (num/total_num)*100
                strings.append('{}={:.1f}%'.format(name, percent))
        return ' '.join(strings)
