import logging

from .paac import ParallelActorCritic, th, np, n_step_returns, gae_returns, normalize
import torch.nn.functional as F
import copy

class ProximalPolicyOptimization(ParallelActorCritic):
    save_every = 5*(10**5 )
    print_every = 10240
    eval_every = 10*10240 # * 20

    class RolloutData(object):
        """
        RolloutData stores all data collected in the rollout that algorithm requires to update it's model
        If you want to collect additional information about the model or environments during rollout steps,
        you will have to override this class and the rollout method.
        """
        __slots__ = [
            'states',
            'infos',
            'actions',
            'values',
            'log_probs',
            'rewards',
            'masks',
            'next_v',
            'initial_rnn_state',
            #do i really need the following ones?
            #'returns' #we will compute the later from rewards,masks and values
        ]
        #states = [rollout_steps + 1, num_envs, *obs_shape]
        #rnn_states = [rollout_steps + 1, num_envs, hidden_size]
        #masks = [rollout_steps + 1, num_envs, 1]
        #values = [rollout_steps + 1, num_envs, 1]
        #returns = [rollout_steps + 1, num_envs, 1]

        #rewards = [rollout_steps, num_processes, 1]
        #log_probs = [rollout_steps, num_envs, 1]
        #actions = [rollout_steps, num_envs, action_shape]


        def __init__(self, states, infos, actions,
                     values, log_probs, rewards,
                     masks, next_v, initial_rnn_state=None):
            self.states = states
            self.infos = infos
            self.actions = actions
            self.values = values
            self.log_probs = log_probs
            self.rewards = rewards
            self.masks = masks
            self.next_v = next_v

            self.initial_rnn_state = initial_rnn_state
            #self.returns = returns

        @classmethod
        def create_empty(cls):
            return cls(
                states=[],
                infos=[],
                actions=[],
                values=[],
                log_probs=[],
                rewards=[],
                masks=[],
                next_v=None,
                initial_rnn_state=None
            )

        def start_rollout(self, start_state, start_info, start_mask, start_rnn_state):
            #RolloutData is just a simple storage, so it is better to make
            # all required preprocessing with the arguments before passing them
            # into this method
            self.states.append(start_state)
            self.infos.append(start_info)
            self.masks.append(start_mask)
            self.initial_rnn_state = start_rnn_state

        def add_step(self, a_t, log_prob_t, v_t, r_t, mask_next, state_next, info_next):
            # RolloutData is just a storage, so it is better to make
            # all required preprocessing with the arguments before passing them
            # into this method
            self.actions.append(a_t)
            self.log_probs.append(log_prob_t)
            self.values.append(v_t)
            self.rewards.append(r_t)
            self.masks.append(mask_next)
            self.states.append(state_next)
            self.infos.append(info_next)

        def end_rollout(self, next_v):
            self.next_v = next_v

        def compute_returns(self, gamma=0.99,  use_gae=False, lam=0.95):
            # masks[1:] comes from the fact that for PPO we store masks
            # from before the initial state of rollout
            with th.no_grad():
                if use_gae:
                    return gae_returns(self.next_v, self.values, self.rewards, self.masks[1:], gamma, lam)
                else:
                    return n_step_returns(self.next_v, self.rewards, self.masks[1:], gamma)

    def __init__(self, *args, **kwargs):
        self.ppo_epochs = kwargs.pop('ppo_epochs', 4)
        self.ppo_batch_num = kwargs.pop('ppo_batch_num', 4)#or 32?
        self.ppo_clip = kwargs.pop('ppo_clip', 0.1)
        self.kl_threshold = kwargs.pop('kl_threshold', None)

        super(ProximalPolicyOptimization, self).__init__(*args, **kwargs)

    def rollout(self, state, info, mask, rnn_state):
        with th.no_grad():
            data = self.RolloutData.create_empty()
            #states, actions, values, log_probs, rewards, masks = [[] for _ in range(6)]
            #s_0,m_0,h_0 --> net--> a_0,v_0, h_1 |a_0 --> env --> s_1, m_1

            self.network.detach_rnn_state(rnn_state)
            data.start_rollout(state.copy(), copy.deepcopy(info),
                               mask, copy.deepcopy(rnn_state))

            for t in range(self.rollout_steps):
                outputs = self.choose_action(state, info, mask.unsqueeze(1), rnn_state)
                a_t, v_t, log_probs_t, entropy_t, rnn_state = outputs

                state, r, done, info = self.batch_env.next(a_t.tolist())
                mask = self._to_tensor(1.0 - done)  #done.dtype == np.float32
                #!!! self.batch_env returns references to arrays in shared memory,
                # always copy their values if you want to use them later,
                #  as the values will be rewritten at the next step !!!
                data.add_step(
                    a_t, log_probs_t, v_t, self._to_tensor(self.reshape_r(r)),
                    mask, state.copy(), copy.deepcopy(info)
                )

                done_mask = done.astype(bool)
                self.episodes_rewards += r

                if any(done_mask):
                    self.reward_history.extend(self.episodes_rewards[done_mask])
                    self.episodes_rewards[done_mask] = 0.

            #no need to detach with no_grad
            next_v = self.predict_values(state, info, mask.unsqueeze(1), rnn_state)
            data.end_rollout(next_v)

        return data, (state, info, mask, rnn_state)

    def update_weights(self, rollout_data):
        self.lr_scheduler.adjust_learning_rate(self.global_step)
        # returns, values, log_probs, entropies):
        returns = rollout_data.compute_returns(self.gamma, use_gae=self.use_gae)
        returns = th.stack(returns, 0) #shape: [rollout_steps, num_envs]
        values = th.stack(rollout_data.values, 0)

        advantages = normalize(returns - values) #mean=0, std=1.

        sum_grad_norm = sum_actor_loss = sum_critic_loss = sum_entropy_loss = 0.

        epoch_kl = 0.
        num_updates = 0
        for epoch in range(self.ppo_epochs):
            kl = 0.
            for batch in self._batches_from_rollout(advantages, returns, rollout_data):
                num_updates += 1

                states_batch, infos_batch, masks_batch, init_rnn_states_batch, \
                actions_batch, old_log_probs_batch, \
                returns_batch, adv_batch = batch

                values, log_probs, entropies = self.process_batch(
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

                #update_weights:
                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = self.clip_gradients(self.network.parameters(), self.clip_norm)
                self.optimizer.step()

                sum_actor_loss += loss_info['actor_loss']
                sum_critic_loss += loss_info['critic_loss']
                sum_entropy_loss += loss_info['entropy_loss']
                sum_grad_norm += grad_norm
                # kl-divergence between the old and the new policies:
                kl += (old_log_probs_batch - log_probs).mean().item()

            epoch_kl = kl / self.ppo_batch_num
            if self.kl_threshold and (epoch_kl > self.kl_threshold * 1.4):
                break

        return {
            'epochs': epoch+1,
            'kl': epoch_kl,
            'La':sum_actor_loss/num_updates,
            'Lc':sum_critic_loss/num_updates,
            'Le':sum_entropy_loss/num_updates,
            '|∇|': sum_grad_norm/num_updates
        }

    def _batches_from_rollout(self, advantages, returns, rollout_data):
        if rollout_data.initial_rnn_state:
            indices = th.randperm(self.num_emulators)
            assert self.num_emulators % self.ppo_batch_num == 0, 'we prefer to divide samples to batches of equal size'
            batch_size = self.num_emulators // self.ppo_batch_num
            masks = th.stack(rollout_data.masks,0)
            log_probs = th.stack(rollout_data.log_probs, 0)
            actions = th.stack(rollout_data.actions, 0)
            rnn_state = rollout_data.initial_rnn_state

            for l in range(0, self.num_emulators, batch_size):
                idx = indices[l:l+batch_size]

                states_batch = [s[idx,:] for s in rollout_data.states]
                infos_batch = [{k:v[idx] for k,v in i_t.items()} for i_t in rollout_data.infos]
                masks_batch = masks[:,idx]
                #rnn_state is a dict! e.g. for LSTM rnn_state={'cx':tensor1,'hx':tensor2}
                init_rnn_states_batch = {k:v[idx,:] for k, v in rnn_state.items()}

                adv_batch = advantages[:,idx]
                returns_batch = returns[:, idx]
                log_probs_batch = log_probs[:, idx]
                actions_batch = actions[:, idx]
                yield states_batch, infos_batch, masks_batch, \
                      init_rnn_states_batch, actions_batch, \
                      log_probs_batch, returns_batch, adv_batch
        else:
            raise NotImplementedError()

    def process_batch(self, states, infos, masks, rnn_states, actions):
        if rnn_states:  #recurrent model
            outputs = []
            for t in range(self.rollout_steps):
                *out_t, rnn_states = self.eval_action(
                    states[t], infos[t],
                    masks[t].unsqueeze(1),
                    rnn_states, actions[t]
                )
                outputs.append(out_t)

            return [th.stack(out,0) for out in zip(*outputs)]
        else:
            *outputs, _ = self.eval_action(states, infos, masks, rnn_states, actions)
            return outputs

    def process_batch_old(self, states, infos, masks, rnn_states, actions):
        if rnn_states:  #recurrent model
            values, log_probs, entropies = [], [], []
            for t in range(self.rollout_steps):
                v_t, log_prob_t, entropy_t, rnn_states = self.eval_action(
                    states[t], infos[t],
                    masks[t].unsqueeze(1),
                    rnn_states, actions[t]
                )
                values.append(v_t)
                log_probs.append(log_prob_t)
                entropies.append(entropy_t)
            return th.stack(values,0), th.stack(log_probs,0), th.stack(entropies,0)
        else:
            raise NotImplementedError()

    def eval_action(self, state, info, mask, rnn_state, action):
        """
        This function is similar to choose_action. The main difference here is that we
        only evaluate(compute value function and log prob) a previously selected action
        instead of choosing it.
        :return:
            A tuple of
                value_prediction,
                log_probability of the action,
                entropy of the current policy
                and a new rnn_state
        """
        value, distr, rnn_state = self.network(state, info, mask, rnn_state)
        return value.squeeze(dim=1), distr.log_prob(action), distr.entropy(), rnn_state

    def compute_loss(self, **kwargs):
        """
        Computes total loss for singe ppo update!
        :param log_probs: log probs for the current policy
        :param old_log_probs: log probs for the old policy that generated samples
        :param advantages: normalized advantages computed using the old critic model
        :param values: new value estimates from the current critic model
        :param returns: return estimates computed using the old critic model
        :param entropies: entropy values for the current policy
        :return: a tensor with total loss,
                 plus a dict of with actor_loss, critic_loss and entropy_loss floats
        """
        log_probs, old_log_probs = kwargs['log_probs'], kwargs['old_log_probs']
        advantages, values = kwargs['advantages'], kwargs['values']
        returns, entropies = kwargs['returns'], kwargs['entropies']

        ratio = th.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = th.clamp(ratio, 1. - self.ppo_clip, 1. + self.ppo_clip) * advantages
        actor_loss = -th.min(surr1, surr2).mean()  #maximize actor loss
        #minimize prediction error for critic:
        critic_loss = self.critic_coef * (values - returns).pow(2).mean()
        #critic_loss = F.mse_loss(values, returns) * self.critic_coef
        #critic_loss = 2.*self.critic_coef * F.smooth_l1_loss(values, returns)

        #maximize(that's were minus comes from) entropy:
        entropy_loss = - self.entropy_coef * entropies.mean()

        loss = critic_loss + actor_loss + entropy_loss

        return loss, {
            'actor_loss':actor_loss.item(),
            'critic_loss':critic_loss.item(),
            'entropy_loss':entropy_loss.item(),
        }





