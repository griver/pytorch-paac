import copy
import logging
import shutil
import time
import torch as th

import numpy as np
import torch.nn.functional as F
from torch import optim, nn

import utils
from utils import ensure_dir, join_path, isfile, yellow, red
from utils.lr_scheduler import LinearAnnealingLR
from collections import namedtuple


def n_step_returns(next_value, rewards, masks, gamma=0.99):
    """
    Computes discounted n-step returns for rollout. Expects tensors or numpy.arrays as input parameters
    The function doesn't detach tensors, so you have to take care of the gradient flow by yourself.
    :return:
    """
    rollout_steps = len(rewards)
    returns = [None] * rollout_steps
    R = next_value
    for t in reversed(range(rollout_steps)):
        R = rewards[t] + gamma * masks[t] * R
        returns[t] = R
    return returns


class ParallelActorCritic(object):
    """
    The method is also known as A2C i.e. (Parallel) Advantage Actor Critic.
    https://blog.openai.com/baselines-acktr-a2c/
    https://arxiv.org/abs/1705.04862
    """

    CHECKPOINT_SUBDIR = 'checkpoints/'
    SUMMARY_FILE = 'summaries.pkl4' #pickle, protocol=4
    CHECKPOINT_LAST = 'checkpoint_last.pth'
    CHECKPOINT_BEST = 'checkpoint_best.pth'

    save_every = 10**6
    print_every = 10240
    eval_every = 20*10240

    def __init__(self, network, batch_env, args):
        logging.debug('PAAC init is started')
        self.checkpoint_dir = join_path(args.debugging_folder,self.CHECKPOINT_SUBDIR)
        ensure_dir(self.checkpoint_dir)

        checkpoint = self._load_latest_checkpoint(self.checkpoint_dir)
        self.last_saving_step = checkpoint['last_step'] if checkpoint else 0

        self.global_step = self.last_saving_step
        self.network = network
        self.batch_env = batch_env
        self.optimizer = optim.RMSprop(
            self.network.parameters(),
            lr=args.initial_lr,
            eps=args.e,
        ) #RMSprop defualts: momentum=0., centered=False, weight_decay=0

        if checkpoint:
            logging.info('Restoring agent variables from previous run')
            self.network.load_state_dict(checkpoint['network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.lr_scheduler = LinearAnnealingLR(
            self.optimizer,
            args.lr_annealing_steps
        )
        #pytorch documentation says:
        #In most cases it’s better to use CUDA_VISIBLE_DEVICES environmental variable
        #Therefore to specify a particular gpu one should use CUDA_VISIBLE_DEVICES.
        self.device = self.network._device

        self.gamma = args.gamma # future rewards discount factor
        self.entropy_coef = args.entropy_regularisation_strength
        self.loss_scaling = args.loss_scaling #5.
        self.critic_coef =  args.critic_coef #0.25
        self.total_steps = args.max_global_steps
        self.rollout_steps = args.rollout_steps
        self.clip_norm = args.clip_norm
        self.num_emulators = batch_env.num_emulators

        self.evaluate = None
        self.reshape_r = lambda r: np.clip(r, -1.,1.)
        self.compute_returns = n_step_returns
        if args.clip_norm_type == 'global':
            self.clip_gradients = nn.utils.clip_grad_norm_
        elif args.clip_norm_type == 'local':
            self.clip_gradients = utils.clip_local_grad_norm
        elif args.clip_norm_type == 'ignore':
            self.clip_gradients = lambda params, _: utils.global_grad_norm(params)
        else:
            raise ValueError('Norm type({}) is not recoginized'.format(args.clip_norm_type))
        logging.debug('Paac init is done')

    def train(self):
        """
        Main actor learner loop for parallerl advantage actor critic learning.
        """
        logging.info('Starting training at step %d' % self.global_step)
        logging.debug('Device: {}'.format(self.device))

        num_updates = 0
        global_step_start = self.global_step
        average_loss = utils.MovingAverage(0.01, ['actor', 'critic', 'entropy', 'grad_norm'])
        total_rewards, training_stats = [], []
        num_emulators = self.batch_env.num_emulators
        total_episode_rewards = np.zeros(num_emulators)
        steps_per_update = num_emulators * self.rollout_steps

        if self.evaluate:
            stats = self.evaluate(self.network)
            training_stats.append((self.global_step, stats))

        #stores 0.0 in i-th element if the episode in i-th emulator has just started, otherwise stores 1.0
        #mask is used to cut rnn_state and episode rewards between episodes.
        mask_t = th.zeros(num_emulators).to(self.device)

        #feedforward networks also use rnn_state, it's just empty!
        rnn_state = self.network.init_rnn_state(num_emulators)

        states, infos = self.batch_env.reset_all()

        start_time = time.time()
        while self.global_step < self.total_steps:

            loop_start_time = time.time()
            values, log_probs, rewards, entropies, masks = [], [], [], [], []
            self.network.detach_rnn_state(rnn_state)

            for t in range(self.rollout_steps):
                outputs = self.choose_action(states, infos, mask_t.unsqueeze(1), rnn_state)
                a_t, v_t, log_probs_t, entropy_t, rnn_state = outputs
                states, rs, dones, infos = self.batch_env.next(a_t)

                tensor_rs = th.from_numpy(self.reshape_r(rs)).to(self.device)
                rewards.append(tensor_rs)
                entropies.append(entropy_t)
                log_probs.append(log_probs_t)
                values.append(v_t)

                mask_t = 1.0 - th.from_numpy(dones).to(self.device) #dones.dtype == np.float32
                masks.append(mask_t) #1.0 if episode is not done, 0.0 otherwise

                done_mask = dones.astype(bool)
                total_episode_rewards += rs

                if any(done_mask):
                    total_rewards.extend(total_episode_rewards[done_mask])
                    total_episode_rewards[done_mask] = 0.

            next_v = self.predict_values(states, infos, mask_t.unsqueeze(1), rnn_state)

            update_stats = self.update_weights(next_v, rewards, masks, values, log_probs, entropies)
            average_loss.update(**update_stats)

            self.global_step += steps_per_update
            num_updates += 1

            if num_updates % (self.print_every // steps_per_update) == 0:
                curr_time = time.time()
                self._training_info(
                    total_rewards=total_rewards,
                    average_speed=(self.global_step - global_step_start) / (curr_time - start_time),
                    loop_speed=steps_per_update / (curr_time - loop_start_time),
                    update_stats=average_loss)

            if num_updates % (self.eval_every // steps_per_update) == 0:
                if self.evaluate:
                    stats = self.evaluate(self.network)
                    training_stats.append((self.global_step, stats))

            if self.global_step - self.last_saving_step >= self.save_every:
                self._save_progress(self.checkpoint_dir, summaries=training_stats, is_best=False)
                training_stats = []
                self.last_saving_step = self.global_step

        self._save_progress(self.checkpoint_dir, is_best=False)
        logging.info('Training ended at step %d' % self.global_step)

    def choose_action(self, states, infos, masks, rnn_states):
        values, distr, rnn_states = self.network(states, infos, masks, rnn_states)
        acts = distr.sample().detach()
        log_probs = distr.log_prob(acts)
        entropy = distr.entropy()
        return acts, values.squeeze(dim=1), log_probs, entropy, rnn_states

    def predict_values(self, states, infos, masks, rnn_states):
        values = self.network(states, infos, masks, rnn_states)[0]
        return values.squeeze(dim=1)

    def update_weights(self, next_v, rewards, masks, values, log_probs, entropies):
        returns = self.compute_returns(next_v.detach(), rewards, masks, self.gamma)

        loss, update_data = self.compute_loss(
            th.cat(returns), th.cat(values), th.cat(log_probs), th.cat(entropies)
        )

        self.lr_scheduler.adjust_learning_rate(self.global_step)
        self.optimizer.zero_grad()
        loss.backward()
        global_norm = self.clip_gradients(self.network.parameters(), self.clip_norm)
        self.optimizer.step()

        update_data['grad_norm'] = global_norm
        return update_data

    def compute_loss(self, returns, values, log_probs, entropies):
        advantages = returns - values

        critic_loss = self.critic_coef * advantages.pow(2).mean() #minimize
        actor_loss = th.neg(log_probs * advantages.detach()).mean() # minimize -log(policy(a))*advantage(s,a)
        entropy_loss = self.entropy_coef*entropies.mean() # maximize entropy

        loss = self.loss_scaling * (actor_loss + critic_loss - entropy_loss)

        loss_data = {'actor':actor_loss.item(), 'critic':critic_loss.item(), 'entropy':entropy_loss.item()}
        return loss, loss_data

    @classmethod
    def _load_latest_checkpoint(cls, dir):
        last_chkpt_path = join_path(dir, cls.CHECKPOINT_LAST)
        if isfile(last_chkpt_path):
            return th.load(last_chkpt_path)
        return None

    def _save_progress(self, dir, summaries=None, is_best=False):
        last_chkpt_path = join_path(dir, self.CHECKPOINT_LAST)
        state = {
            'last_step':self.global_step,
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        th.save(state, last_chkpt_path)
        logging.info('The state of the agent is saved at step #%d'%self.global_step)

        if (summaries is not None) and len(summaries) > 0:
            summaries_path = join_path(dir, self.SUMMARY_FILE)
            utils.save_summary(summaries, summaries_path)

        if is_best:
          best_chkpt_path = join_path(dir, self.CHECKPOINT_BEST)
          shutil.copyfile(last_chkpt_path, best_chkpt_path)

    def _training_info(self, total_rewards, average_speed, loop_speed, update_stats):
        last_ten = np.mean(total_rewards[-20:]) if len(total_rewards) else 0.
        logger_msg = "Ran {0} steps, at {1:.3f} fps (avg {2:.3f} fps), last 20 episodes avg {3:.5f}"

        lines = ['',]
        lines.append(logger_msg.format(self.global_step, loop_speed, average_speed, last_ten))
        lines.append(str(update_stats))
        logging.info(yellow('\n'.join(lines)))


def print_grads_norms(net):
    global_norm = utils.global_grad_norm(net.parameters())
    print('Global_grads norm: {:.8f}'.format(global_norm))
    for n, m in net.named_children():
        w_norm = 0. if m.weight.grad is None else utils.global_grad_norm([m.weight])
        b_norm = 0. if m.bias.grad is None else utils.global_grad_norm([m.bias])
        print('--'*10, n, '--'*10)
        print('W_grad norm: {:.8f}\nb_grad norm: {:.8f}'.format(w_norm, b_norm))
