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
from ..paac import ParallelActorCritic

TrainingStats = namedtuple("TrainingStats",
                               ['mean_r', 'max_r', 'min_r', 'std_r',
                                'mean_steps', 'term_acc', 'term_rec',
                                'term_prec', 't_ratio', 'p_ratio'])

class MultiTaskPAAC(ParallelActorCritic):

    def __init__(self, network, batch_env, args):
        super(MultiTaskPAAC, self).__init__(network, batch_env, args)
        self._term_model_coef = args.termination_model_coef
        logging.debug('Termination loss class weights = {0}'.format(args.term_weights))
        class_weights = th.tensor(args.term_weights).to(self.device, th.float32)
        self._term_model_loss = nn.NLLLoss(weight=class_weights).to(self.device)


    def train(self):
        """
         Main actor learner loop for parallerl advantage actor critic learning.
         """
        logging.info('Starting training at step %d' % self.global_step)
        logging.debug('Device: {}'.format(self.device))

        num_updates = 0
        global_step_start = self.global_step
        average_loss = utils.MovingAverage(0.01, ['actor', 'critic', 'entropy', 'term_model', 'grad_norm'])
        total_rewards, training_stats = [], []
        num_emulators = self.batch_env.num_emulators
        steps_per_update = num_emulators * self.rollout_steps
        total_episode_rewards = np.zeros(num_emulators)
        best_mean_r = float('-inf')

        if self.evaluate:
            stats = self.evaluate(self.network)
            training_stats.append((self.global_step, stats))
            curr_mean_r = best_mean_r = stats.mean_r

        #stores 0.0 in i-th element if the episode in i-th emulator has just started, otherwise stores 1.0
        mask_t = th.zeros(num_emulators).to(self.device)
        tasks = np.zeros((self.rollout_steps+1, num_emulators)).to(self.device)
        #feedforward networks also use rnn_state, it's just empty!
        rnn_state = self.network.init_rnn_state(num_emulators)
        states, infos = self.batch_env.reset_all()

        start_time = time.time()
        while self.global_step < self.total_steps:

            loop_start_time = time.time()
            values, log_probs, rewards, entropies, log_terminals, masks = [],[],[],[],[],[]
            self.network.detach_rnn_state(rnn_state)

            for t in range(self.rollout_steps):
                #don't know but probably we need only task values at step t+1
                # i.e the values needed for prediction)
                # and not the values used as input...
                tasks[t] = infos['task_status']
                outputs = self.choose_action(states, infos, mask_t.unsqueeze(1), rnn_state)
                a_t, v_t, log_probs_t, entropy_t, log_term_t, rnn_state = outputs
                states, rs, dones, infos = self.batch_env.next(a_t)

                tensor_rs = th.from_numpy(self.reshape_r(rs)).to(self.device)
                rewards.append(tensor_rs)
                log_terminals.append(log_term_t)
                entropies.append(entropy_t)
                log_probs.append(log_probs_t)
                values.append(v_t)

                mask_t = 1.0 - th.from_numpy(dones).to(self.device)
                masks.append(mask_t) #1.0 if episode is not done, 0.0 otherwise

                done_mask = dones.astype(bool)
                total_episode_rewards += rs
                if any(done_mask):
                    total_rewards.extend(total_episode_rewards[done_mask])
                    total_episode_rewards[done_mask] = 0.

            tasks[self.rollout_steps] = infos['task_status']
            next_v = self.predict_values(states, infos, mask_t.unsqueeze(1), rnn_state)
            update_stats = self.update_weights(next_v, rewards, masks, values, log_probs, entropies)
            average_loss.update(**update_stats)

            self.global_step += steps_per_update
            num_updates += 1

            if num_updates % (10240 // steps_per_update) == 0:
                curr_time = time.time()
                self._training_info(
                    total_rewards=total_rewards,
                    average_speed=(self.global_step-global_step_start) / (curr_time-start_time),
                    loop_speed=steps_per_update / (curr_time-loop_start_time),
                    update_stats=average_loss)

            if num_updates % (self.eval_every // steps_per_update) == 0:
                if self.evaluate:
                    stats = self.evaluate(self.network)
                    training_stats.append((self.global_step, stats))
                    curr_mean_r = stats.mean_r

            if self.global_step - self.last_saving_step >= self.save_every:
                is_best = False
                if curr_mean_r > best_mean_r:
                    best_mean_r = curr_mean_r
                    is_best=True
                self._save_progress(self.checkpoint_dir, summaries=training_stats, is_best=is_best)
                training_stats = []
                self.last_saving_step = self.global_step

        self._save_progress(self.checkpoint_dir, is_best=False)
        logging.info('Training ended at step %d' % self.global_step)

    def choose_action(self, states, infos, masks, net_states):
        values, distr, done_logits, net_states = self.network(states, infos, masks, net_states)
        acts = distr.sample().detach()
        log_probs = distr.log_prob(acts)
        entropy = distr.entropy()
        log_done = F.log_softmax(done_logits, dim=1)
        return acts, values.squeeze(dim=1), log_probs, entropy, log_done, net_states

    def update_weights(self, next_v, rewards, masks, values, log_probs, entropies, **kwargs):
        """нужно выкинуть в другие места
        term_loss = self.compute_termination_model_loss(log_terminals, tasks)
        if self._term_model_coef > 0.:# and self.global_step >= self.args['warmup']:
            loss += self._term_model_coef * term_loss

        self.lr_scheduler.adjust_learning_rate(self.global_step)
        self.network.zero_grad()
        loss.backward()
        global_norm = self.clip_gradients(self.network.parameters(), self.clip_norm)
        self.optimizer.step()
        /нужно выкинуть в другие места"""
        pass

    def compute_termination_model_loss(self, log_terminals, tasks):
        #tasks_done = (tasks[:-1] != tasks[1:]).astype(int)
        tasks_done = (tasks[:-1] != 0).astype(np.int32)
        tasks_done = torch.from_numpy(tasks_done).type(self._tensors.LongTensor)
        tasks_done = Variable(tasks_done.view(-1))
        log_terminals = torch.cat(log_terminals, 0) #.type(self._tensors.FloatTensor)
        term_loss = self._term_model_loss(log_terminals, tasks_done)
        return term_loss