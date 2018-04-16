import logging
import time
import torch

import numpy as np
import torch.nn.functional as F
from torch import  nn
from torch.autograd import Variable

import utils
from utils import red
from collections import namedtuple
from paac.paac import PAACLearner, check_log_zero

TrainingStats = namedtuple("TrainingStats",
                               ['mean_r', 'max_r', 'min_r', 'std_r',
                                'mean_steps', 'term_acc', 'term_rec',
                                'term_prec', 't_ratio', 'p_ratio'])

class MultiTaskPAAC(PAACLearner):

    def __init__(self, network_creator, batch_env, args):
        super(MultiTaskPAAC, self).__init__(network_creator, batch_env, args)
        self._term_model_coef = args.termination_model_coef
        logging.debug('Termination loss class weights = {0}'.format(args.term_weights))
        self._term_model_loss = nn.NLLLoss(weight=torch.FloatTensor(args.term_weights))
        if self.use_cuda:
            self._term_model_loss = self._term_model_loss.cuda()

    def train(self):
        """
         Main actor learner loop for parallerl advantage actor critic learning.
         """
        logging.info('Starting training at step %d' % self.global_step)
        logging.debug('use_cuda == {}'.format(self.use_cuda))

        counter = 0
        global_step_start = self.global_step
        ma_loss = utils.MovingAverage(0.01, ['total', 'actor', 'critic', 'term_model'])
        total_rewards, training_stats = [], []

        if self.eval_func is not None:
            stats = self.eval_func(*self.eval_args, **self.eval_kwargs)
            training_stats.append((self.global_step, stats))
            curr_mean_r = best_mean_r = stats.mean_r


        num_emulators = self.args['num_envs']
        max_local_steps = self.args['max_local_steps']
        max_global_steps = self.args['max_global_steps']
        clip_norm = self.args['clip_norm']
        rollout_steps = num_emulators * max_local_steps

        # any summaries here?
        emulator_steps = np.zeros(num_emulators, dtype=int)
        total_episode_rewards = np.zeros(num_emulators)
        not_done_masks = torch.zeros(max_local_steps, num_emulators).type(self._tensors.FloatTensor)
        tasks = np.zeros((max_local_steps+1, num_emulators))

        hx, cx = None, None #for feedforward nets just ignore this argument
        if self.use_rnn:
            hx_init, cx_init = self.network.get_initial_state(num_emulators)
            hx, cx = hx_init, cx_init

        states, infos = self.batch_env.reset_all()
        start_time = time.time()
        while self.global_step < max_global_steps:
            loop_start_time = time.time()
            values, log_probs, rewards, entropies, log_terminals = [], [], [], [], []
            if self.use_rnn: hx, cx = hx.detach(), cx.detach()  # Do I really need to detach here?

            for t in range(max_local_steps):
                #don't know but probably we need only task values at step t+1
                # i.e thr values needed for prediction)
                # and not the values used as input...
                tasks[t] = infos['task_status']
                outputs = self.choose_action(states, infos, (hx, cx))
                a_t, v_t, log_probs_t, entropy_t, log_term_t, (hx, cx) = outputs
                states, rs, dones, infos = self.batch_env.next(a_t)
                #print('=======================')
                log_terminals.append(log_term_t)
                rewards.append(self.adjust_rewards(rs))
                entropies.append(entropy_t)
                log_probs.append(log_probs_t)
                values.append(v_t)
                is_done = torch.from_numpy(dones).type(self._tensors.FloatTensor)
                not_done_masks[t] = 1.0 - is_done

                done_mask = dones.astype(bool)
                total_episode_rewards += rs
                emulator_steps += 1

                total_rewards.extend(total_episode_rewards[done_mask])
                total_episode_rewards[done_mask] = 0.
                emulator_steps[done_mask] = 0
                if self.use_rnn and any(done_mask):  # we need to clear all lstm states corresponding to the terminated emulators
                    done_idx = is_done.nonzero().view(-1)
                    hx, cx = hx.clone(), cx.clone()  # hx_t, cx_t are used for backward op, so we can't modify them in-place
                    hx[done_idx, :] = hx_init[done_idx,:].detach()
                    cx[done_idx, :] = cx_init[done_idx,:].detach()

            tasks[max_local_steps] = infos['task_status']
            self.global_step += rollout_steps
            next_v = self.predict_values(states, infos, (hx,cx))
            R = next_v.detach().view(-1)

            delta_v = []
            for t in reversed(range(max_local_steps)):
                r_t = Variable(torch.from_numpy(rewards[t])).type(self._tensors.FloatTensor)
                not_done_t = Variable(not_done_masks[t])
                R = r_t + self.gamma * R * not_done_t
                delta_v_t = R - values[t].view(-1)
                delta_v.append(delta_v_t)

            loss, actor_loss, critic_loss = self.compute_loss(
                torch.cat(delta_v, 0),
                torch.cat(log_probs, 0).view(-1),
                torch.cat(entropies, 0).view(-1)
            )

            term_loss = self.compute_termination_model_loss(log_terminals, tasks)
            if self._term_model_coef > 0.:# and self.global_step >= self.args['warmup']:
                loss += self._term_model_coef * term_loss

            self.lr_scheduler.adjust_learning_rate(self.global_step)
            self.network.zero_grad()
            loss.backward()
            global_norm = self.clip_gradients(self.network.parameters(), clip_norm)
            self.optimizer.step()

            ma_loss.update(total=loss.data[0], actor=actor_loss.data[0],
                           critic=critic_loss.data[0], term_model=term_loss.data[0])
            counter += 1
            if counter % (10240 // rollout_steps) == 0:
                curr_time = time.time()
                self._training_info(
                    total_rewards=total_rewards,
                    average_speed=(self.global_step-global_step_start) / (curr_time-start_time),
                    loop_speed=rollout_steps / (curr_time-loop_start_time),
                    moving_averages=ma_loss, grad_norms=global_norm
                )
            if counter % (self.eval_every // rollout_steps) == 0:
                if (self.eval_func is not None):
                    stats = self.eval_func(*self.eval_args, **self.eval_kwargs)
                    training_stats.append((self.global_step, stats))
                    curr_mean_r = stats.mean_r

            if self.global_step - self.last_saving_step >= self.save_every:
                if curr_mean_r > best_mean_r:
                    best_mean_r = curr_mean_r
                    is_best=True
                self._save_progress(self.checkpoint_dir, summaries=training_stats, is_best=is_best)
                training_stats = []
                self.last_saving_step = self.global_step

        self._save_progress(self.checkpoint_dir, is_best=False)
        logging.info('Training ended at step %d' % self.global_step)

    def choose_action(self, states, infos, rnn_states):
        if self.use_rnn:
            values, a_logits, done_logits, rnn_states = self.network(states, infos, rnn_states)
        else:
            values, a_logits, done_logits = self.network(states, infos)

        log_done = F.log_softmax(done_logits, dim=1)
        probs = F.softmax(a_logits, dim=1)
        log_probs = F.log_softmax(a_logits, dim=1)
        entropy = torch.neg((log_probs * probs)).sum(1)
        acts = probs.multinomial().detach()
        selected_log_probs = log_probs.gather(1, acts)

        check_log_zero(log_probs.data)
        acts_one_hot = self.action_codes[acts.data.cpu().view(-1).numpy(), :]
        return acts_one_hot, values, selected_log_probs, entropy, log_done, rnn_states


    def compute_termination_model_loss(self, log_terminals, tasks):
        #tasks_done = (tasks[:-1] != tasks[1:]).astype(int)
        tasks_done = (tasks[1:] != 0).astype(np.int32)
        tasks_done = torch.from_numpy(tasks_done).type(self._tensors.LongTensor)
        tasks_done = Variable(tasks_done.view(-1))
        log_terminals = torch.cat(log_terminals, 0) #.type(self._tensors.FloatTensor)
        term_loss = self._term_model_loss(log_terminals, tasks_done)
        return term_loss