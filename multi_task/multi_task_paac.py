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


class MultiTaskActorCritic(ParallelActorCritic):

    class RolloutData(object):
        __slots__ = [
            'values','log_probs',
            'rewards', 'entropies',
            'masks', 'next_v',
            'log_terminals','tasks_status']

        def __init__(self, values, log_probs, rewards,
                     entropies, log_terminals, masks,
                     tasks_status, next_v):
            self.values = values
            self.log_probs = log_probs
            self.rewards = rewards
            self.entropies =entropies
            self.masks = masks
            self.next_v = next_v
            self.log_terminals =log_terminals
            self.tasks_status = tasks_status

    def __init__(self, *args, **kwargs):
        term_weights = kwargs.pop('term_weights')
        self._term_model_coef = kwargs.pop('termination_model_coef')

        super(MultiTaskActorCritic, self).__init__(*args, **kwargs)

        logging.debug('Termination loss class weights = {0}'.format(term_weights))
        class_weights = th.tensor(term_weights).to(self.device, th.float32)
        self._term_model_loss = nn.NLLLoss(weight=class_weights).to(self.device)
        self.average_loss = utils.MovingAverage(0.01, ['actor', 'critic', 'entropy', 'term_loss', 'grad_norm'])

    def rollout(self, state, info, mask, rnn_state):
        """performs a rollout"""
        self.network.detach_rnn_state(rnn_state)
        values, log_probs, rewards, entropies, log_dones, masks = [], [], [], [], [], []
        tasks_status = []

        for t in range(self.rollout_steps):
            outputs = self.choose_action(state, info, mask.unsqueeze(1), rnn_state)
            a_t, v_t, log_probs_t, entropy_t, log_done_t, rnn_state = outputs
            state, r, done, info = self.batch_env.next(a_t)
            #!!! self.batch_env returns references to arrays in shared memory,
            # always copy their values if you want to use them later,
            #  as the values will be rewritten at the next step !!!
            tasks_status.append( self._to_tensor(info['task_status'],th.long) )
            rewards.append( self._to_tensor(self.reshape_r(r)) )
            log_dones.append(log_done_t)
            entropies.append(entropy_t)
            log_probs.append(log_probs_t)
            values.append(v_t)

            mask = self._to_tensor(1. - done)
            masks.append(mask)  #1.0 if episode is not done, 0.0 otherwise

            done_mask = done.astype(bool)
            self.episodes_rewards += r
            if any(done_mask):
                self.reward_history.extend(self.episodes_rewards[done_mask])
                self.episodes_rewards[done_mask] = 0.

        next_v = self.predict_values(state, info, mask.unsqueeze(1), rnn_state).detach()

        rollout_data = self.RolloutData(values, log_probs, rewards, entropies,
                                        log_dones, masks, tasks_status, next_v)
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
        if self._term_model_coef > 0.:
            term_loss, term_info = self.compute_termination_model_loss(
                th.cat(rollout_data.log_terminals),
                th.cat(rollout_data.tasks_status),#0-running,1-success,2-fail
                th.cat(rollout_data.masks).to(th.uint8) #1-episode is not done, 0-episode is done
            )
            loss += self._term_model_coef*term_loss
            update_info.update(**term_info)

        self.lr_scheduler.adjust_learning_rate(self.global_step)
        self.optimizer.zero_grad()
        loss.backward()
        global_norm = self.clip_gradients(self.network.parameters(), self.clip_norm)
        self.optimizer.step()

        update_info['grad_norm'] = global_norm
        return update_info

    def compute_termination_model_loss(self, log_terminals, tasks_status, ep_masks):
        tasks_status[tasks_status > 1] = 0 # we are predicting only success states
        tasks_status = tasks_status[ep_masks]
        log_terminals = log_terminals[ep_masks,:]

        term_loss = self._term_model_loss(log_terminals, target=tasks_status)
        loss_data = {'term_loss':term_loss.item()}

        return term_loss, loss_data