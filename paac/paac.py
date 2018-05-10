import copy
import logging
import shutil
import time
import torch

import numpy as np
import torch.nn.functional as F
from torch import optim, nn
from torch.autograd import Variable

import utils
from utils import ensure_dir, join_path, isfile, yellow, red
from utils.lr_scheduler import LinearAnnealingLR
from collections import namedtuple

TrainingStats = namedtuple("TrainingStats",
                           ['mean_r', 'max_r', 'min_r', 'std_r', 'mean_steps'])

class PAACLearner(object):
    CHECKPOINT_SUBDIR = 'checkpoints/'
    SUMMARY_FILE = 'summaries.pkl4' #pickle, protocol=4
    CHECKPOINT_LAST = 'checkpoint_last.pth'
    CHECKPOINT_BEST = 'checkpoint_best.pth'

    save_every = 10**6
    print_every = 10240
    eval_every = 20*10240

    def __init__(self, network_creator, batch_env, args):
        logging.debug('PAAC init is started')
        self.args = copy.copy(vars(args))
        self.checkpoint_dir = join_path(self.args['debugging_folder'],self.CHECKPOINT_SUBDIR)
        ensure_dir(self.checkpoint_dir)

        checkpoint = self._load_latest_checkpoint(self.checkpoint_dir)
        self.last_saving_step = checkpoint['last_step'] if checkpoint else 0

        self.global_step = self.last_saving_step
        self.network = network_creator()
        self.batch_env = batch_env
        self.optimizer = optim.RMSprop(
            self.network.parameters(),
            lr=self.args['initial_lr'],
            eps=self.args['e'],
        ) #RMSprop defualts: momentum=0., centered=False, weight_decay=0

        if checkpoint:
            logging.info('Restoring agent variables from previous run')
            self.network.load_state_dict(checkpoint['network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.lr_scheduler = LinearAnnealingLR(
            self.optimizer,
            self.args['lr_annealing_steps']
        )
        #pytorch documentation says:
        #In most cases it’s better to use CUDA_VISIBLE_DEVICES environmental variable
        #Therefore to specify a particular gpu one should use CUDA_VISIBLE_DEVICES.
        self.use_cuda = self.args['device'] == 'gpu'
        self.use_rnn = hasattr(self.network, 'get_initial_state') #get_initial_state should return state of the rnn layers
        self._tensors = torch.cuda if self.use_cuda else torch

        self.action_codes = np.eye(batch_env.num_actions) #envs reveive actions in one-hot encoding!
        self.gamma = self.args['gamma'] # future rewards discount factor
        self.entropy_coef = self.args['entropy_regularisation_strength']
        self.loss_scaling = self.args['loss_scaling'] #5.
        self.critic_coef = self.args['critic_coef'] #0.25
        self.eval_func = None

        if self.args['clip_norm_type'] == 'global':
            self.clip_gradients = nn.utils.clip_grad_norm_
        elif self.args['clip_norm_type'] == 'local':
            self.clip_gradients = utils.clip_local_grad_norm
        elif self.args['clip_norm_type'] == 'ignore':
            self.clip_gradients = lambda params, _: utils.global_grad_norm(params)
        else:
            raise ValueError('Norm type({}) is not recoginized'.format(self.args['clip_norm_type']))
        logging.debug('Paac init is done')

    def train(self):
        """
        Main actor learner loop for parallerl advantage actor critic learning.
        """
        logging.info('Starting training at step %d' % self.global_step)
        logging.debug('use_cuda == {}'.format(self.use_cuda))

        counter = 0
        global_step_start = self.global_step
        average_loss = utils.MovingAverage(0.01, ['total', 'actor', 'critic'])
        total_rewards, training_stats = [], []

        if self.eval_func is not None:
            stats = self.evaluate(verbose=True)
            training_stats.append((self.global_step, stats))

        #num_actions = self.args['num_actions']
        num_emulators = self.args['num_envs']
        max_local_steps = self.args['max_local_steps']
        max_global_steps = self.args['max_global_steps']
        clip_norm = self.args['clip_norm']
        rollout_steps = num_emulators * max_local_steps

        states, infos = self.batch_env.reset_all()

        emulator_steps = np.zeros(num_emulators, dtype=int)
        total_episode_rewards = np.zeros(num_emulators)
        not_done_masks = torch.zeros(max_local_steps, num_emulators).type(self._tensors.FloatTensor)
        if self.use_rnn:
            hx_init, cx_init = self.network.get_initial_state(num_emulators)
            hx, cx = hx_init, cx_init
        else: #for feedforward nets just ignore this argument
            hx, cx = None, None

        start_time = time.time()
        while self.global_step < max_global_steps:
            loop_start_time = time.time()
            values, log_probs, rewards, entropies = [], [], [], []
            if self.use_rnn:
                hx, cx = hx.detach(), cx.detach() #Do I really need to detach here?

            for t in range(max_local_steps):
                outputs = self.choose_action(states, infos, (hx,cx))
                a_t, v_t, log_probs_t, entropy_t, (hx, cx) = outputs
                states, rs, dones, infos = self.batch_env.next(a_t)

                #actions_sum += a_t
                rewards.append(np.clip(rs, -1., 1.))
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
                if self.use_rnn and any(done_mask): # we need to clear all lstm states corresponding to the terminated emulators
                        done_idx = is_done.nonzero().view(-1)
                        hx, cx = hx.clone(), cx.clone() #hx_t, cx_t are used for backward op, so we can't modify them in-place
                        hx[done_idx,:] = hx_init[done_idx, :].detach()
                        cx[done_idx,:] = cx_init[done_idx,:].detach()

            self.global_step += rollout_steps
            next_v = self.predict_values(states, infos, (hx,cx))
            R = next_v.detach().view(-1)

            delta_v = []
            for t in reversed(range(max_local_steps)):
                rs = Variable(torch.from_numpy(rewards[t])).type(self._tensors.FloatTensor)
                not_done_t = Variable(not_done_masks[t])
                R = rs + self.gamma * R * not_done_t
                delta_v_t = R - values[t].view(-1)
                delta_v.append(delta_v_t)

            loss, actor_loss, critic_loss = self.compute_loss(
                torch.cat(delta_v,0),
                torch.cat(log_probs,0).view(-1),
                torch.cat(entropies,0).view(-1)
            )

            self.lr_scheduler.adjust_learning_rate(self.global_step)
            self.optimizer.zero_grad()
            loss.backward()
            global_norm = self.clip_gradients(self.network.parameters(), clip_norm)
            self.optimizer.step()

            average_loss.update(total=loss.data.item(),
                                actor=actor_loss.item(),
                                critic=critic_loss.item())

            counter += 1
            if counter % (self.print_every // rollout_steps) == 0:
                curr_time = time.time()
                self._training_info(
                    total_rewards=total_rewards,
                    average_speed=(self.global_step - global_step_start) / (curr_time - start_time),
                    loop_speed=rollout_steps / (curr_time - loop_start_time),
                    moving_averages=average_loss, grad_norms=global_norm)

            if counter % (self.eval_every // rollout_steps) == 0:
                if (self.eval_func is not None):
                    stats = self.evaluate(verbose=True)
                    training_stats.append((self.global_step, stats))

            if self.global_step - self.last_saving_step >= self.save_every:
                self._save_progress(self.checkpoint_dir, summaries=training_stats, is_best=False)
                training_stats = []
                self.last_saving_step = self.global_step

        self._save_progress(self.checkpoint_dir, is_best=False)
        logging.info('Training ended at step %d' % self.global_step)

    def choose_action(self, states, infos, rnn_states):
        if self.use_rnn:
            values, a_logits, rnn_states = self.network(states, infos, rnn_states)
        else:
            values, a_logits = self.network(states, infos) #without rnn_state

        probs = F.softmax(a_logits, dim=1)
        log_probs = F.log_softmax(a_logits, dim=1)
        entropy = torch.neg((log_probs * probs)).sum(1)
        acts = probs.multinomial(1).detach()
        selected_log_probs = log_probs.gather(1, acts)

        check_log_zero(log_probs.data)
        acts_one_hot = self.action_codes[acts.data.cpu().view(-1).numpy(),:]
        return acts_one_hot, values, selected_log_probs, entropy, rnn_states

    def predict_values(self, states, infos, rnn_states):
        if self.use_rnn:
            return self.network(states, infos, rnn_states)[0]
        return self.network(states, infos)[0]

    def compute_loss(self, delta_v, selected_log_probs, entropies):
        #delta_v = target_value - v_t which is basicale an advantage_t
        ##detach() prevents from providing grads from actor_loss to the critic
        advantages = delta_v.detach()
        actor_loss = selected_log_probs * advantages + self.entropy_coef * entropies
        actor_loss = torch.neg(torch.mean(actor_loss, 0)) #-1. * actor_loss
        critic_loss = self.critic_coef * torch.mean(delta_v.pow(2), 0)
        loss = self.loss_scaling * (actor_loss + critic_loss)
        return loss, actor_loss, critic_loss

    @classmethod
    def _load_latest_checkpoint(cls, dir):
        last_chkpt_path = join_path(dir, cls.CHECKPOINT_LAST)
        if isfile(last_chkpt_path):
            return torch.load(last_chkpt_path)
        return None

    def _save_progress(self, dir, summaries=None, is_best=False):
        last_chkpt_path = join_path(dir, self.CHECKPOINT_LAST)
        state = {
            'last_step':self.global_step,
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        torch.save(state, last_chkpt_path)
        logging.info('The state of the agent is saved at step #%d'%self.global_step)

        if (summaries is not None) and len(summaries) > 0:
            summaries_path = join_path(dir, self.SUMMARY_FILE)
            utils.save_summary(summaries, summaries_path)

        if is_best:
          best_chkpt_path = join_path(dir, self.CHECKPOINT_BEST)
          shutil.copyfile(last_chkpt_path, best_chkpt_path)

    def _training_info(self, total_rewards, average_speed, loop_speed, moving_averages, grad_norms):
        last_ten = np.mean(total_rewards[-10:]) if len(total_rewards) else 0.
        logger_msg = "Ran {} steps, at {} steps/s ({} steps/s avg), last 10 rewards avg {}"

        lines = ['',]
        lines.append(logger_msg.format(self.global_step, loop_speed, average_speed, last_ten))
        lines.append(str(moving_averages))
        lines.append('grad_norm: {}'.format(grad_norms))
        logging.info(yellow('\n'.join(lines)))

    def evaluate(self, verbose=True):
        num_steps, rewards = self.eval_func(*self.eval_args, **self.eval_kwargs)

        mean_steps = np.mean(num_steps)
        min_r, max_r = np.min(rewards), np.max(rewards)
        mean_r, std_r = np.mean(rewards), np.std(rewards)

        stats = TrainingStats(mean_r, max_r, min_r, std_r, mean_steps)
        if verbose:
            lines = [
                'Perfromed {0} tests:'.format(len(num_steps)),
                'Mean number of steps: {0:.3f}'.format(mean_steps),
                'Mean R: {0:.2f} | Std of R: {1:.3f}'.format(mean_r, std_r)]
            logging.info(red('\n'.join(lines)))

        return stats

    def set_eval_function(self, eval_func, *args, **kwargs):
        self.eval_func = eval_func
        self.eval_args = args
        self.eval_kwargs = kwargs


def check_log_zero(logs_results):
    #print('log_results:', logs_results, sep='\n')
    #print('-inf mask:', float('-inf') == logs_results, sep='\n')
    if any(float('-inf') == logs_results.view(-1)):
      raise ValueError(' The logarithm of zero is undefined!')


def print_grads_norms(net):
    global_norm = utils.global_grad_norm(net.parameters())
    print('Global_grads norm: {:.8f}'.format(global_norm))
    for n, m in net.named_children():
        w_norm = 0. if m.weight.grad is None else utils.global_grad_norm([m.weight])
        b_norm = 0. if m.bias.grad is None else utils.global_grad_norm([m.bias])
        print('--'*10, n, '--'*10)
        print('W_grad norm: {:.8f}\nb_grad norm: {:.8f}'.format(w_norm, b_norm))
