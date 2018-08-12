import logging
import time
import torch as th

import numpy as np
import torch.nn.functional as F
from torch import  nn
from torch.autograd import Variable

import utils
from utils import red
from collections import namedtuple
from paac import ParallelActorCritic

TrainingStats = namedtuple("TrainingStats",
                               ['mean_r', 'max_r', 'min_r', 'std_r',
                                'mean_steps', 'term_acc', 'term_rec',
                                'term_prec', 't_ratio', 'p_ratio'])


class MultiTaskPAAC(ParallelActorCritic):

    class RolloutData(object):
        __slots__ = [
            'values','log_probs',
            'rewards', 'entropies',
            'masks', 'next_v',
            'log_terminals','tasks']

        def __init__(self, values, log_probs, rewards,
                     entropies, log_terminals, masks,
                     tasks, next_v):
            self.values = values
            self.log_probs = log_probs
            self.rewards = rewards
            self.entropies =entropies,
            self.masks = masks
            self.next_v = next_v
            self.log_terminals =log_terminals
            self.tasks = tasks

    def __init__(self, network, batch_env, args):
        super(MultiTaskPAAC, self).__init__(network, batch_env, args)
        self._term_model_coef = args.termination_model_coef
        logging.debug('Termination loss class weights = {0}'.format(args.term_weights))
        class_weights = th.tensor(args.term_weights).to(self.device, th.float32)
        self._term_model_loss = nn.NLLLoss(weight=class_weights).to(self.device)
        self.average_loss = utils.MovingAverage(0.01, ['actor', 'critic', 'entropy', 'term_loss', 'grad_norm'])

    def rollout(self, state, info, mask, rnn_state):
        """performs a rollout"""
        self.network.detach_rnn_state(rnn_state)
        values, log_probs, rewards, entropies, log_terminals, masks = [], [], [], [], [], []
        tasks = []

        for t in range(self.rollout_steps):
            #don't know but probably we need only task values at step t+1
            # i.e the values needed for prediction)
            # and not the values used as input...
            tasks.append(info['task_status'])
            outputs = self.choose_action(state, info, mask.unsqueeze(1), rnn_state)
            a_t, v_t, log_probs_t, entropy_t, log_term_t, rnn_state = outputs
            state, r, done, info = self.batch_env.next(a_t)

            tensor_rs = th.from_numpy(self.reshape_r(r)).to(self.device)
            rewards.append(tensor_rs)
            log_terminals.append(log_term_t)
            entropies.append(entropy_t)
            log_probs.append(log_probs_t)
            values.append(v_t)

            mask = 1.0 - th.from_numpy(done).to(self.device)
            masks.append(mask)  #1.0 if episode is not done, 0.0 otherwise

            done_mask = done.astype(bool)
            self.episodes_rewards += r
            if any(done_mask):
                self.reward_history.extend(self.episodes_rewards[done_mask])
                self.episodes_rewards[done_mask] = 0.

        tasks[self.rollout_steps] = info['task_status']
        next_v = self.predict_values(state, info, mask.unsqueeze(1), rnn_state).detach()

        rollout_data = self.RolloutData(values, log_probs, rewards, entropies,
                                        log_terminals, masks, tasks, next_v)
        return rollout_data, (state, info, mask, rnn_state)

    def choose_action(self, states, infos, masks, net_states):
        values, distr, done_logits, net_states = self.network(states, infos, masks, net_states)
        acts = distr.sample().detach()
        log_probs = distr.log_prob(acts)
        entropy = distr.entropy()
        log_done = F.log_softmax(done_logits, dim=1)
        return acts, values.squeeze(dim=1), log_probs, entropy, log_done, net_states

    def update_weights(self, rollout_data):
        returns = self.compute_returns(rollout_data.next_v, rollout_data.rewards, rollout_data.masks, self.gamma)

        loss, update_info = self.compute_loss(
            th.cat(returns), th.cat(rollout_data.values),
            th.cat(rollout_data.log_probs), th.cat(rollout_data.entropies)
        )
        if self._term_model_loss > 0.:
            term_loss, term_info = self.compute_termination_model_loss(rollout_data.log_terminals, rollout_data.tasks)
            loss += self._term_model_loss * term_loss
            update_info.update(**term_info)

        self.lr_scheduler.adjust_learning_rate(self.global_step)
        self.optimizer.zero_grad()
        loss.backward()
        global_norm = self.clip_gradients(self.network.parameters(), self.clip_norm)
        self.optimizer.step()

        update_info['grad_norm'] = global_norm
        return update_info

    def compute_termination_model_loss(self, log_terminals, tasks):
        #tasks_done = (tasks[:-1] != tasks[1:]).astype(int)
        tasks_done = (tasks[:-1] != 0).astype(np.int32)
        tasks_done = th.tensor(tasks_done, device=self.device, dtype=th.long)
        tasks_done = tasks_done.view(-1)
        log_terminals = th.cat(log_terminals) #.type(self._tensors.FloatTensor)
        term_loss = self._term_model_loss(log_terminals, tasks_done)
        loss_data = {'term_loss':term_loss.item()}
        return term_loss, loss_data