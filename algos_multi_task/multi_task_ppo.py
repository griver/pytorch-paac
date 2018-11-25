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
from algos import ParallelActorCritic, ProximalPolicyOptimization, normalize

TrainingStats = namedtuple("TrainingStats",
                               ['mean_r', 'max_r', 'min_r', 'std_r',
                                'mean_steps', 'term_acc', 'term_rec',
                                'term_prec', 't_ratio', 'p_ratio'])


class MultiTaskPPO(ProximalPolicyOptimization):

    #class RolloutData(ProximalPolicyOptimization.RolloutData):
        # no need to save store aditional data here, as
        # ProximalPolicyOptimization.RolloutData already stores

    def __init__(self, *args, **kwargs):
        term_weights = kwargs.pop('term_weights')
        self._term_model_coef = kwargs.pop('termination_model_coef')

        super(MultiTaskPPO, self).__init__(*args, **kwargs)

        logging.debug('Termination loss class weights = {0}'.format(term_weights))
        class_weights = th.tensor(term_weights).to(self.device, th.float32)
        self._term_model_loss = nn.NLLLoss(weight=class_weights).to(self.device)

    def choose_action(self, states, infos, masks, rnn_states):
        values, distr, _, rnn_states = self.network(states, infos, masks, rnn_states)
        acts = distr.sample().detach()
        log_probs = distr.log_prob(acts)
        #we don't need to return log_done for experience sampling during training phase
        #log_done = F.log_softmax(done_logits, dim=1)
        # we don't need entropy either!
        # entropy = distr.entropy()
        return acts, values.squeeze(dim=1), log_probs, None, rnn_states

    def update_weights(self, rollout_data):
        self.lr_scheduler.adjust_learning_rate(self.global_step)
        # returns, values, log_probs, entropies):
        returns = rollout_data.compute_returns(self.gamma, use_gae=self.use_gae)
        returns = th.stack(returns, 0) #shape: [rollout_steps, num_envs]
        values =  th.stack(rollout_data.values, 0)

        advantages = returns - values
        #advantages = normalize(advantages) #mean=0, std=1.

        sum_grad_norm = sum_actor_loss = sum_critic_loss \
            = sum_entropy_loss = sum_term_loss = 0.

        num_updates = 0
        for epoch in range(self.ppo_epochs):
            for batch in self._batches_from_rollout(advantages, returns, rollout_data):
                num_updates += 1

                states_batch, infos_batch, masks_batch, init_rnn_states_batch, \
                actions_batch, old_log_probs_batch, \
                returns_batch, adv_batch = batch

                values, log_probs, entropies, log_terms = self.process_batch(
                    states_batch, infos_batch, masks_batch,
                    init_rnn_states_batch, actions_batch
                )

                loss, loss_info = self.compute_loss(
                    log_probs=log_probs,
                    old_log_probs=old_log_probs_batch,
                    advantages=adv_batch,
                    values=values,
                    returns=returns_batch,
                    entropies=entropies
                )

                #task termination predictor loss:
                if self._term_model_coef > 0.:
                    #infos[i+1] corresponds to the results of action[i]
                    #masks[i+1] is 0. if episode is ended after action[i], 1. otherwise
                    tasks_status_batch = [info['task_status'] for info in infos_batch[1:]]
                    term_loss, term_info = self.compute_termination_model_loss(
                        log_terminals=log_terms,
                        tasks_status=self._to_tensor(tasks_status_batch, th.long),
                        masks=masks_batch[1:].to(th.uint8)
                    )
                    loss += self._term_model_coef * term_loss
                    sum_term_loss += term_info['term_loss']

                #update_weights:
                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = self.clip_gradients(self.network.parameters(), self.clip_norm)
                self.optimizer.step()

                sum_actor_loss += loss_info['actor_loss']
                sum_critic_loss += loss_info['critic_loss']
                sum_entropy_loss += loss_info['entropy_loss']
                sum_grad_norm += grad_norm #

        return {
            'actor_loss':sum_actor_loss/num_updates,
            'critic_loss':sum_critic_loss/num_updates,
            'entropy_loss':sum_entropy_loss/num_updates,
            'term_loss':sum_term_loss/num_updates,
            'grad_norm': sum_grad_norm/num_updates
        }

    def eval_action(self, state, info, mask, rnn_state, action):
        value, distr, term_logits, rnn_state = self.network(state, info, mask, rnn_state)
        log_term = F.log_softmax(term_logits, dim=1)
        return value.squeeze(dim=1), distr.log_prob(action), \
               distr.entropy(), log_term, rnn_state

    def compute_termination_model_loss(self, **kwargs):
        '''
        Receives three tensors: log_terminals, tasks_status, masks
        :return: a 0-tensor with termination prediction loss
                 and a dict with it's float value
        '''
        log_terminals, tasks_status, ep_masks = kwargs['log_terminals'], kwargs['tasks_status'], kwargs['masks']
        tasks_status[tasks_status > 1] = 0 # we are predicting only success states
        tasks_status = tasks_status[ep_masks]
        log_terminals = log_terminals[ep_masks,:]

        term_loss = self._term_model_loss(log_terminals, target=tasks_status)
        loss_data = {'term_loss':term_loss.item()}

        return term_loss, loss_data