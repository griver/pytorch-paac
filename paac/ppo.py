from .paac import ParallelActorCritic


class ProximalPolicyOptimization(ParallelActorCritic):

    class RolloutData(object):
        """
        RolloutData stores all data collected in the rollout that algorithm requires to update it's model
        If you want to collect additional information about the model or environments during rollout steps,
        you will have to override this class and the rollout method.
        """
        __slots__ = [
            'states',
            'actions',
            'values',
            'log_probs',
            'rewards',
            'masks',
            'next_v'
        ]

        def __init__(self, states, actions, values, log_probs, rewards, masks, next_v):
            self.states = states
            self.actions = actions
            self.values = values
            self.log_probs = log_probs
            self.rewards = rewards
            self.masks = masks
            self.next_v = next_v

    def __init__(self, *args, *kwargs):
        super(ProximalPolicyOptimization, self).__init__(*args, **kwargs)
        self.ppo_epochs=4
        self.mini_batch_size=32 #or 256?
        self.ppo_clip = 0.2

    # rollouts = RolloutStorage(args.num_steps, args.num_processes,
    #                           envs.observation_space.shape, envs.action_space,
    #                           actor_critic.recurrent_hidden_state_size)
    #
    def rollout(self, state, info, mask, rnn_state):
          pass




