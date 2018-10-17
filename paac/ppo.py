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
            'next_v',
            #do i really need the following ones?
            'rnn_states',
            'returns'
        ]

        def __init__(self, states, actions, values, log_probs, rewards, masks, next_v, rnn_states=None, returns=None):
            self.states = states
            self.actions = actions
            self.values = values
            self.log_probs = log_probs
            self.rewards = rewards
            self.masks = masks
            self.next_v = next_v
            self.rnn_states = rnn_states
            self.returns = returns

    def __init__(self, *args, **kwargs):
        self.ppo_epochs = kwargs.pop('ppo_epochs', 4)
        self.mini_batch_size = kwargs.pop('mini_batch_size', 4)#or 32?
        self.ppo_clip = kwargs.pop('ppo_clip', 0.1)
        kwargs.setdefault('critic_coef', 1.0)

        super(ProximalPolicyOptimization, self).__init__(*args, **kwargs)

    def rollout(self, state, info, mask, rnn_state):
        states, actions, values, log_probs, rewards, masks = [[] for _ in range(6)]




