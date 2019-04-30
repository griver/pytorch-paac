from .paac_nets import BaseAgentNetwork, AtariLSTM, nn, init_model_weights, Categorical, torch
SMALL_VAL = -1e9

def np2pt(*arrays, device=None):
    if device:
        return [torch.tensor(a, device=device) for a in arrays]
    return [torch.tensor(a) for a in arrays]

class MaskedActionActorCritic(nn.Module, BaseAgentNetwork):
    def __init__(self, num_actions, observation_shape, device,
                 action_mask_as_input=False,  hidden_size=256,
                 preprocess=np2pt):
        super(MaskedActionActorCritic, self).__init__()
        self._num_actions = num_actions
        self._obs_shape = observation_shape
        self._device = device
        self._hidden_size = hidden_size
        self.action_mask_as_input=action_mask_as_input
        self._preprocess = preprocess
        self._create_network()
        #recursivly traverse layers and inits weights and biases:
        self.apply(init_model_weights)
        assert self.training == True, "Model won't train If self.training is False"


    def _create_network(self):
        D, = self._obs_shape
        self.obs_encoder = nn.Sequential(
            nn.Linear(D, self._hidden_size),
            nn.ReLU(),
            nn.Linear(self._hidden_size, self._hidden_size),
            nn.ReLU(),
        )

        self.lstm = nn.LSTMCell(self._hidden_size, self._hidden_size, bias=True)

        n_features = self._hidden_size
        if self.action_mask_as_input:
            n_features += self._num_actions

        self.fc_policy = nn.Linear(n_features, self._num_actions)
        self.fc_value = nn.Linear(n_features, 1)


    def forward(self, obs, infos, masks, net_state):
        if self._preprocess:
            obs, act_mask = self._preprocess(
                obs, infos['act_mask'], device=self._device
            )

        x = self.obs_encoder(obs)

        hx, cx = net_state['hx']*masks, net_state['cx']*masks
        hx, cx = self.lstm(x, (hx,cx))

        if self.action_mask_as_input:
            state = torch.cat([hx, act_mask], dim=1)
        else:
            state = hx

        logits = self.fc_policy(state)
        logits = logits.masked_fill(act_mask==0, SMALL_VAL)
        distr = Categorical(logits=logits)

        return self.fc_value(state), distr, dict(hx=hx,cx=cx)

    def init_rnn_state(self, batch_size=None):
        '''
        Returns initial lstm state as a dict(hx=hidden_state, cx=cell_state).
        Intial lstm state is supposed to be used at the begging of an episode.
        '''
        shape = (batch_size, self.lstm.hidden_size) if batch_size else (self.lstm.hidden_size,)
        hx = torch.zeros(*shape, dtype=torch.float32, device=self._device)
        cx = torch.zeros(*shape, dtype=torch.float32, device=self._device)
        return dict(hx=hx, cx=cx)

    def detach_rnn_state(self, rnn_state):
        rnn_state['hx'] = rnn_state['hx'].detach()
        rnn_state['cx'] = rnn_state['cx'].detach()

networks = {
    'masked':MaskedActionActorCritic
}