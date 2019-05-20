import copy
import logging
import shutil
import time
import torch as th

import numpy as np
from torch import optim, nn

import utils
from utils import ensure_dir, join_path, isfile, yellow, red
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


def gae_returns(next_value, values, rewards, masks, gamma=0.99, lam=0.95):
    """
    GAE(gamma, lam) = SUM_{t in range(0,rollout_steps)} (gamma*lam)^t delta_t,
    where delta_t = r_t + gamma * value_{t+1} - value[t]

    returns = GAE(gamma, lam) + values

    The returned values is used as targets values for the critic loss
    and for computation of advantage estimates for the actor loss.

    The function doesn't detach tensors, so you have to take care of the gradient flow by yourself.
    """
    rollout_steps = len(rewards)
    values = values + [next_value]
    returns = [None] * rollout_steps
    gae = 0.
    for t in reversed(range(rollout_steps)):
        delta_t =rewards[t] + gamma*masks[t]*values[t+1] - values[t]
        gae = delta_t + gamma*lam*masks[t]*gae
        returns[t] = gae + values[t]

    return returns


def normalize(tensor, eps=1e-6):
    return (tensor - tensor.mean()) / (tensor.std() + eps)


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

    class RolloutData(object):
        """
        RolloutData stores all data collected in the rollout that algorithm requires to update it's model
        If you want to collect additional information about the model or environments during rollout steps,
        you will have to override this class and the rollout method.
        """
        __slots__ = ['values', 'log_probs', 'rewards', 'entropies', 'masks', 'next_v',]

        def __init__(self, values, log_probs, rewards,
                     entropies, masks, next_v):
            self.values = values
            self.log_probs = log_probs
            self.rewards = rewards
            self.entropies =entropies
            self.masks = masks
            self.next_v = next_v

        def compute_returns(self, gamma=0.99, use_gae=False, lam=0.95):
            with th.no_grad():
                if use_gae:
                    return gae_returns(self.next_v, self.values, self.rewards, self.masks, gamma, lam)
                else:
                    return n_step_returns(self.next_v, self.rewards, self.masks, gamma)

    def __init__(self,
                 network,
                 optimizer,
                 lr_scheduler,
                 batch_env,
                 save_folder='test_log/',
                 global_step=0,
                 max_global_steps=80000000,
                 rollout_steps=10,
                 gamma=0.99,
                 critic_coef=0.5,
                 entropy_coef=0.01,
                 clip_norm=0.5,
                 use_gae=False,
                 ):

        logging.debug('PAAC init is started')
        self.checkpoint_dir = join_path(save_folder, self.CHECKPOINT_SUBDIR)
        ensure_dir(self.checkpoint_dir)

        self.global_step = global_step
        self.last_print = self.last_eval = self.last_save = global_step

        self.network = network
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.batch_env = batch_env
        self.num_emulators = batch_env.num_emulators
        #pytorch documentation says:
        #In most cases it’s better to use CUDA_VISIBLE_DEVICES environmental variable
        #Therefore to specify a particular gpu one should use CUDA_VISIBLE_DEVICES.
        self.device = self.network._device

        self.gamma = gamma # future rewards discount factor
        self.entropy_coef = entropy_coef
        self.critic_coef =  critic_coef #0.5
        self.total_steps = max_global_steps
        self.rollout_steps = rollout_steps
        self.clip_norm = clip_norm
        self.use_gae = use_gae
        self.evaluate = None
        self.reshape_r = lambda r: np.clip(r, -1.,1.)

        self.clip_gradients = nn.utils.clip_grad_norm_
        #self.clip_gradients = utils.clip_local_grad_norm
        #self.clip_gradients = lambda params, _: utils.global_grad_norm(params)

        self.average_loss = utils.MovingAverage(0.05)
        self.episodes_rewards = np.zeros(batch_env.num_emulators)
        self.reward_history = []
        logging.debug('Paac init is done')

    def _to_tensor(self, data, dtype=th.float32):
        return th.tensor(data, dtype=dtype, device=self.device)

    def train(self):
        """
        Main actor learner loop for parallerl advantage actor critic learning.
        """
        logging.info('Starting training at step %d' % self.global_step)
        logging.debug('Device: {}'.format(self.device))

        num_updates = 0
        global_step_start = self.global_step

        num_emulators = self.batch_env.num_emulators
        training_stats = []
        steps_per_update = num_emulators * self.rollout_steps
        curr_quality = best_quality = float('-inf')

        if self.evaluate:
            stats = self.evaluate(self.network)
            stats['num_episodes']=len(self.reward_history)
            training_stats.append((self.global_step, stats))
            curr_quality = stats['quality']

        state, info = self.batch_env.reset_all()
        #stores 0.0 in i-th element if the episode in i-th emulator has just started, otherwise stores 1.0
        mask = th.zeros(self.batch_env.num_emulators).to(self.device)
        #feedforward networks also use rnn_state, it's just empty!
        rnn_state = self.network.init_rnn_state(num_emulators)

        start_time = time.time()
        while self.global_step < self.total_steps:

            loop_start_time = time.time()
            rollout_data, finals = self.rollout(state, info, mask, rnn_state)
            #final states of environments and network at the end of rollout
            state, info, mask, rnn_state = finals

            update_stats = self.update_weights(rollout_data)
            self.average_loss.update(**update_stats)

            self.global_step += steps_per_update
            num_updates += 1

            if self.global_step - self.last_print >= self.print_every:
                curr_time = time.time()
                self.last_print = self.global_step
                self._training_info(
                    total_rewards=self.reward_history,
                    average_speed=(self.global_step - global_step_start) / (curr_time - start_time),
                    loop_speed=steps_per_update / (curr_time - loop_start_time),
                    update_stats=self.average_loss)

            if self.global_step - self.last_eval >= self.eval_every:
                self.last_eval = self.global_step
                if self.evaluate:
                    stats = self.evaluate(self.network)
                    stats['num_episodes'] = len(self.reward_history)
                    training_stats.append((self.global_step, stats))
                    curr_quality = stats['quality']

            if self.global_step - self.last_save >= self.save_every:
                is_best = False
                if curr_quality > best_quality:
                    best_quality = curr_quality
                    is_best = True
                self._save_progress(self.checkpoint_dir, summaries=training_stats, is_best=is_best)
                training_stats = []
                self.last_save = self.global_step

        if len(training_stats) and training_stats[-1][0] != self.global_step:
            #if we haven't already evaluated the network at the current step:
            training_stats.append((self.global_step, self.evaluate(self.network)))

        self._save_progress(self.checkpoint_dir, summaries=training_stats, is_best=False)
        logging.info('Training ended at step %d' % self.global_step)

    def rollout(self, state, info, mask, rnn_state):
        values, log_probs, rewards, entropies, masks = [], [], [], [], []
        self.network.detach_rnn_state(rnn_state)

        for t in range(self.rollout_steps):
            outputs = self.choose_action(state, info, mask.unsqueeze(1), rnn_state)
            a_t, v_t, log_probs_t, entropy_t, rnn_state = outputs
            state, r, done, info = self.batch_env.next(a_t.tolist())
            #!!! self.batch_env returns references to arrays in shared memory,
            # always copy their values if you want to use them later,
            #  as the values will be rewritten at the next step !!!
            rewards.append( self._to_tensor(self.reshape_r(r)) )
            entropies.append(entropy_t)
            log_probs.append(log_probs_t)
            values.append(v_t)

            mask = self._to_tensor(1.0-done) #done.dtype == np.float32
            masks.append(mask)  #1.0 if episode is not done, 0.0 otherwise

            done_mask = done.astype(bool)
            self.episodes_rewards += r

            if any(done_mask):
                self.reward_history.extend(self.episodes_rewards[done_mask])
                self.episodes_rewards[done_mask] = 0.

        next_v = self.predict_values(state, info, mask.unsqueeze(1), rnn_state).detach()

        rollout_data = self.RolloutData(values, log_probs, rewards, entropies, masks, next_v)
        return rollout_data, (state, info, mask, rnn_state)

    def choose_action(self, states, infos, masks, rnn_states):
        values, distr, rnn_states = self.network(states, infos, masks, rnn_states)
        acts = distr.sample().detach()
        log_probs = distr.log_prob(acts)
        entropy = distr.entropy()
        return acts, values.squeeze(dim=1), log_probs, entropy, rnn_states

    def predict_values(self, states, infos, masks, rnn_states):
        values = self.network(states, infos, masks, rnn_states)[0]
        return values.squeeze(dim=1)

    def update_weights(self, rollout_data):
        #next_v, rewards, masks, values, log_probs, entropies
        returns = rollout_data.compute_returns(self.gamma, use_gae=self.use_gae)

        loss, update_info = self.compute_loss(
            returns=th.cat(returns),
            values=th.cat(rollout_data.values),
            log_probs=th.cat(rollout_data.log_probs),
            entropies=th.cat(rollout_data.entropies)
        )

        self.lr_scheduler.adjust_learning_rate(self.global_step)
        self.optimizer.zero_grad()
        loss.backward()
        global_norm = self.clip_gradients(self.network.parameters(), self.clip_norm)
        self.optimizer.step()

        update_info['grad_norm'] = global_norm
        return update_info

    def compute_loss(self, **kwargs):
        returns, values = kwargs['returns'], kwargs['values']
        log_probs, entropies = kwargs['log_probs'], kwargs['entropies']

        advantages = returns - values.detach()
        #advantages = normalize(advantages)

        critic_loss = self.critic_coef * (values - returns).pow(2).mean() #advantages.pow(2).mean() #minimize
        actor_loss = th.neg(log_probs * advantages).mean() # minimize -log(policy(a))*advantage(s,a)
        entropy_loss = self.entropy_coef*entropies.mean() # maximize entropy

        loss = actor_loss + critic_loss - entropy_loss

        return loss, {
            'actor_loss':actor_loss.item(),
            'critic_loss':critic_loss.item(),
            'entropy_loss':entropy_loss.item()
        }

    @classmethod
    def update_from_checkpoint(Cls, save_folder, network, optimizer=None,
                               use_best=False, use_cpu=False, ignore_layers=tuple()):
        """
        Update network and optimizer(if specified) from the checkpoints in the save folder.
        If use_best is True then the data is loaded from the Cls.CHECKPOINT_BEST file
        otherwise the data is loaded from the Cls.CHECKPOINT_LAST file
        Returns the number of global steps past for the loaded checkpoint.
        Arguments:
            save_folder (str): a path to a folder containing summaries and weights
                of the pretrained model.
            network (torch.nn.Module): a model we want to update with weights from
                the checkpoint.
            optimizer (torch.optim.Optimizer, optional): sometimes an optimizer needs
                to be update along with the model, e.g. when continuing the previously
                stopped training procedure. Default: None
            use_best (bool, optional): whether to load the last saved model or the
                model with the best score. Default: False
            use_cpu (bool, optinoal): TODO: continue with comments and descriptions
        """
        filename = Cls.CHECKPOINT_BEST if use_best else Cls.CHECKPOINT_LAST
        chkpt_path = join_path(save_folder, Cls.CHECKPOINT_SUBDIR, filename)

        if not isfile(chkpt_path):
            checkpoint = None
        elif use_cpu:
            #avoids loading cuda tensors if the gpu memory is unavailable or too small
            checkpoint = th.load(chkpt_path, map_location='cpu')
        else:
            checkpoint = th.load(chkpt_path)

        last_saving_step = 0

        if checkpoint:
            last_saving_step = checkpoint['last_step']

            chkpt_state_dict = checkpoint['network_state_dict']
            if ignore_layers:
                ignore_layers = set(ignore_layers)
                param_names = list(chkpt_state_dict.keys())

                get_module_name = lambda name:name.rpartition('.')[0]

                for p_name in param_names:
                    if get_module_name(p_name) in ignore_layers:
                        del chkpt_state_dict[p_name]

                logging.info('Restoring model weights from the previous run,'
                             ' except layers: {}'.format(ignore_layers))
                network.load_state_dict(chkpt_state_dict, strict=False)

            else:
                logging.info('Restoring model weights from the previous run')
                network.load_state_dict(chkpt_state_dict)

            if optimizer:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return last_saving_step

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
