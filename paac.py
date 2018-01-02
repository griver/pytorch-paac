import shutil
import utils
from utils import ensure_dir, join_path, isfile
from lr_scheduler import LinearAnnealingLR
from torch import optim, nn
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np
import copy, time, logging, torch
#utside imports:
from runners import Runners
from emulator_runner import EmulatorRunner


class PAACLearner(object):
    CHECKPOINT_SUBDIR = 'checkpoints/'
    SUMMARY_FILE = 'summaries.pkl4' #pickle, protocol=4
    CHECKPOINT_LAST = 'checkpoint_last.pth'
    CHECKPOINT_BEST = 'checkpoint_best.pth'
    CHECKPOINT_INTERVAL = 10**6

    def __init__(self, network_creator, env_creator, args):
        logging.debug('PAAC init is started')
        self.args = copy.copy(vars(args))
        self.checkpoint_dir = join_path(self.args['debugging_folder'],self.CHECKPOINT_SUBDIR)
        ensure_dir(self.checkpoint_dir)

        checkpoint = self._load_latest_checkpoint(self.checkpoint_dir)
        self.last_saving_step = checkpoint['last_step'] if checkpoint else 0

        self.global_step = self.last_saving_step
        self.network = network_creator()
        self.optimizer = optim.RMSprop(
            self.network.parameters(),
            lr=self.args['initial_lr'],
            alpha=self.args['alpha'],
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
        self.use_lstm = ('ff' not in self.args['arch']) #not a feedforward
        self._modeltypes = torch.cuda if self.use_cuda else torch
        self._looptypes = self._modeltypes # model

        self.action_codes = np.eye(self.args['num_actions']) #envs reveive actions in one-hot encoding!
        self.gamma = self.args['gamma'] # future rewards discount factor
        self.entropy_coef = self.args['entropy_regularisation_strength']
        self.loss_scaling = self.args['loss_scaling'] #5.
        self.critic_coef = self.args['critic_coef'] #0.25

        self.emulators = np.asarray(
            [env_creator.create_environment(i) for i in range(self.args['emulator_counts'])]
        )

        if self.args['clip_norm_type'] == 'global':
            self.clip_gradients = nn.utils.clip_grad_norm
        elif self.args['clip_norm_type'] == 'local':
            self.clip_gradients = utils.clip_local_grad_norm
        elif self.args['clip_norm_type'] == 'ignore':
            self.clip_gradients = lambda params, _: utils.global_grad_norm(params)
        else:
            raise ValueError('Norm type({}) is not recoginized'.format(self.args['clip_norm_type']))
        logging.debug('Paac init is done!')

    def train(self):
        """
        Main actor learner loop for parallerl advantage actor critic learning.
        """
        logging.debug('Starting training at step %d' % self.global_step)
        print('Pytorch tensor type is used in the traning loop:', self._looptypes.FloatTensor)
        print('Pytorch tensor type is used in the model:', self._modeltypes.FloatTensor)
        counter = 0
        global_step_start = self.global_step

        #num_actions = self.args['num_actions']
        num_emulators = self.args['emulator_counts']
        max_local_steps = self.args['max_local_steps']
        max_global_steps = self.args['max_global_steps']
        clip_norm = self.args['clip_norm']

        self.runners = self._create_runners()
        self.runners.start()

        shared_vars = self.runners.get_shared_variables()
        shared_s, shared_r, shared_done, shared_a = shared_vars
        #any summaries here?
        #actions_sum = np.zeros((num_emulators, num_actions))
        emulator_steps = np.zeros(num_emulators, dtype=int)
        total_episode_rewards = np.zeros(num_emulators)
        not_done_masks = torch.zeros(max_local_steps, num_emulators).type(self._looptypes.FloatTensor)
        total_rewards = []
        if self.use_lstm:
            hx_init, cx_init = self.network.get_initial_state(num_emulators)
            hx, cx = hx_init.detach(), cx_init.detach() #Do I really need to detach here?

        start_time = time.time()
        while self.global_step < max_global_steps:
            loop_start_time = time.time()
            values, log_probs, rewards, entropies = [], [], [], []
            if self.use_lstm:
                hx, cx = hx.detach(), cx.detach() #Do I really need to detach here?
            #print('outer loop #{} global_step#{}'.format(counter, self.global_step))
            for t in range(max_local_steps):
                if self.use_lstm:
                    inputs = (shared_s, (hx,cx))
                    a_t, v_t, log_probs_t, entropy_t, (hx,cx) = self.choose_action(inputs)
                else:
                    a_t, v_t, log_probs_t, entropy_t = self.choose_action(shared_s)

                shared_a[:] = a_t[:]
                self.runners.update_environments()
                self.runners.wait_updated()
                #actions_sum += a_t
                rewards.append(np.clip(shared_r, -1., 1.))
                entropies.append(entropy_t.type(self._looptypes.FloatTensor))
                log_probs.append(log_probs_t.type(self._looptypes.FloatTensor))
                values.append(v_t.type(self._looptypes.FloatTensor))
                is_done = torch.from_numpy(shared_done).type(self._looptypes.FloatTensor)
                not_done_masks[t] = 1.0 - is_done

                done_mask = shared_done.astype(bool)
                total_episode_rewards += shared_r
                emulator_steps += 1
                self.global_step += num_emulators
                total_rewards.extend(total_episode_rewards[done_mask])
                total_episode_rewards[done_mask] = 0.
                emulator_steps[done_mask] = 0
                if self.use_lstm and any(done_mask): # we need to clear all lstm states corresponding to the terminated emulators
                        done_idx = is_done.nonzero().view(-1)
                        hx, cx = hx.clone(), cx.clone() #hx_t, cx_t are used for backward op, so we can't modify them in-place
                        hx[done_idx,:] = hx_init[done_idx, :].detach()
                        cx[done_idx,:] = cx_init[done_idx,:].detach()
                #print('  inner_loop #{0} global_step#{1}'.format(t, self.global_step))

            inputs = (shared_s, (hx, cx)) if self.use_lstm else shared_s
            next_v = self.network(inputs)[0]
            R = next_v.detach().view(-1).type(self._looptypes.FloatTensor)

            delta_v = []
            for t in reversed(range(max_local_steps)):
                r_t = Variable(torch.from_numpy(rewards[t])).type(self._looptypes.FloatTensor)
                not_done_t = Variable(not_done_masks[t])
                R = r_t + self.gamma * R * not_done_t
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
            #print('actor_loss:', actor_loss.data.numpy(), end=' ')
            #print('critic_loss:', critic_loss.data.numpy(), end=' ')
            #print('loss:', loss.cpu().data.numpy())
            counter += 1
            if counter % (10240 // (num_emulators * max_local_steps)) == 0:
                curr_time = time.time()
                last_ten = np.mean(total_rewards[-10:]) if len(total_rewards) else 0.
                logger_msg = "Ran {} steps, at {} steps/s ({} steps/s avg), last 10 rewards avg {}"
                steps_sec = (max_local_steps * num_emulators)/(curr_time - loop_start_time)
                avr_steps_sec = (self.global_step - global_step_start)/(curr_time - start_time)
                logging.info(logger_msg.format(self.global_step, steps_sec, avr_steps_sec, last_ten))
                logging.info('grad_norm: {}'.format(global_norm))

            if self.global_step - self.last_saving_step >= self.CHECKPOINT_INTERVAL:
                self._save_progress(self.network, self.optimizer, self.checkpoint_dir, is_best=False)
                self.last_saving_step = self.global_step

        self.cleanup()
        logging.debug('Training ended at step %d' % self.global_step)

    def choose_action(self, inputs):
        if self.use_lstm:
            values, a_logits, lstm_state = self.network(inputs)
        else:
            values, a_logits = self.network(inputs)
        probs = F.softmax(a_logits, dim=1)
        log_probs = F.log_softmax(a_logits, dim=1)
        entropy = torch.neg((log_probs * probs)).sum(1)
        acts = probs.multinomial().detach()
        selected_log_probs = log_probs.gather(1, acts)

        check_log_zero(log_probs.data)
        acts_one_hot = self.action_codes[acts.data.cpu().view(-1).numpy(),:]
        if self.use_lstm:
            return acts_one_hot, values, selected_log_probs, entropy, lstm_state
        else:
            return acts_one_hot, values, selected_log_probs, entropy

    def compute_loss(self, delta_v, selected_log_probs, entropies):
        #delta_v = target_value - v_t which is basicale an advantage_t
        ##detach() prevents from providing grads from actor_loss to the critic
        advantages = delta_v.detach()
        actor_loss = selected_log_probs * advantages + self.entropy_coef * entropies
        actor_loss = torch.neg(torch.mean(actor_loss, 0)) #-1. * actor_loss
        critic_loss = self.critic_coef * torch.mean(delta_v.pow(2), 0)
        loss = self.loss_scaling * (actor_loss + critic_loss)
        loss = loss.type(self._modeltypes.FloatTensor) #move to gpu
        return loss, actor_loss, critic_loss

    def _create_runners(self):
        num_emulators = self.args['emulator_counts']
        num_actions = self.args['num_actions']
        num_workers = self.args['emulator_workers']
        #variables = [state, reward, is_done, action]
        variables = [
            np.asarray([em.get_initial_state() for em in self.emulators], dtype=np.uint8),
            np.zeros(num_emulators, dtype=np.float32),
            np.asarray([False] * num_emulators, dtype=np.float32),
            np.zeros((num_emulators, num_actions), dtype=np.float32)
        ]
        logging.debug('Creating runners...')
        return Runners(EmulatorRunner, self.emulators, num_workers, variables)

    @classmethod
    def _load_latest_checkpoint(cls, dir):
        last_chkpt_path = join_path(dir, cls.CHECKPOINT_LAST)
        if isfile(last_chkpt_path):
            return torch.load(last_chkpt_path)
        return None

    def _save_progress(self, network, optimizer, dir, summaries=None, is_best=False):
        last_chkpt_path = join_path(dir, self.CHECKPOINT_LAST)
        state = {
            'last_step':self.global_step,
            'network_state_dict': network.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }
        torch.save(state, last_chkpt_path)
        logging.info('The state of the agent is saved at step #%d'%self.global_step)

        if (summaries is not None) and len(summaries) > 0:
            summaries_path = join_path(dir, self.SUMMARY_FILE)
            utils.save_summary(summaries, summaries_path)

        if is_best:
          best_chkpt_path = join_path(dir, self.CHECKPOINT_BEST)
          shutil.copyfile(last_chkpt_path, best_chkpt_path)

    def cleanup(self):
        if self.global_step - self.last_saving_step > 0:
            self._save_progress(self.network, self.optimizer, self.checkpoint_dir, is_best=False)
        self.runners.stop()


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
