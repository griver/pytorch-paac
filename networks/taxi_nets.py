from .paac_nets import torch, nn, F, np, init_model_weights, calc_output_shape
from .paac_nets import Categorical, BaseAgentNetwork

def preprocess_taxi_input(obs, tasks_ids, t_device):
    obs = torch.from_numpy(np.ascontiguousarray(obs, dtype=np.float32))
    obs = torch.tensor(obs, device=t_device, dtype=torch.float32)
    tasks_ids = torch.tensor(tasks_ids.tolist(), device=t_device, dtype=torch.long)
    return obs, tasks_ids


class MultiTaskFFNetwork(nn.Module, BaseAgentNetwork):

    def __init__(self, num_actions, observation_shape, device,
                 num_tasks=5, task_embed_dim=32, preprocess=None):
        super(MultiTaskFFNetwork, self).__init__()
        self._num_actions = num_actions
        self._device = device
        self._obs_shape = observation_shape
        self._task_embed_dim = task_embed_dim
        self._num_tasks = num_tasks
        self._preprocess = preprocess if preprocess is not None else lambda *args: args
        self._create_network()

        self.apply(init_model_weights)
        assert self.training == True, "Model won't train if self.training is False"

    def _create_network(self):
        C, H, W = self._obs_shape #(channels, height, width)
        self.conv1 = nn.Conv2d(C, 16, (3,3), stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, (3,3), stride=1, padding=1)
        self.embed1 = nn.Embedding(self._num_tasks, self._task_embed_dim)

        C_out, H_out, W_out = calc_output_shape((C,H,W),[self.conv1, self.conv2])
        #fc3 receives flattened conv network output + current task embedding
        self.fc3 = nn.Linear(C_out*H_out*W_out + self._task_embed_dim, 256)
        self.fc_policy = nn.Linear(256, self._num_actions)
        self.fc_value = nn.Linear(256, 1)
        self.fc_terminal = nn.Linear(256, 2) # two classes: is_done, not is_done.

    def forward(self, obs, infos, masks, net_state):
        obs, task_ids = self._preprocess(obs, infos['task_id'], self._device)
        #conv
        x = F.relu(self.conv1(obs))
        x = F.relu(self.conv2(x))
        x = x.view(x.size()[0], -1)
        #task embed:
        task_vecs = self.embed1(task_ids)
        x = torch.cat((x, task_vecs), 1)
        #fc3(conv(obs) + embed(tasks))
        x = F.relu(self.fc3(x))

        action_logits = self.fc_policy(x)
        action_distr = Categorical(logits=action_logits)
        state_value = self.fc_value(x)
        terminal_logits = self.fc_terminal(x)

        return state_value, action_distr, terminal_logits, {}

    def terminal_prediction_params(self):
        for name, param in self.named_parameters():
            if 'terminal' in name:
                yield param

    def actor_critic_params(self):
        for name, param, in self.named_parameters():
            if 'terminal' not in name:
                yield param


class MultiTaskLSTMNetwork(nn.Module, BaseAgentNetwork):
    def __init__(self, num_actions, observation_shape, device,
                 num_tasks=5, task_embed_dim=32, preprocess=None):
        super(MultiTaskLSTMNetwork, self).__init__()
        self._num_actions = num_actions
        self._device = device
        self._obs_shape = observation_shape
        self._task_embed_dim = task_embed_dim
        self._num_tasks = num_tasks
        self._preprocess = preprocess if preprocess is not None else lambda *args: args
        self._create_network()

        self.apply(init_model_weights)
        assert self.training == True, "Model won't train If self.training is False"

    def _create_network(self):
        C,H,W = self._obs_shape
        self.conv1 = nn.Conv2d(C, 16, (3,3), stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, (3,3), stride=1, padding=1)
        self.embed1 = nn.Embedding(self._num_tasks, self._task_embed_dim)

        C_out,H_out,W_out = calc_output_shape((C,H,W),[self.conv1, self.conv2])

        self.lstm = nn.LSTMCell(C_out*H_out*W_out + self._task_embed_dim, 256, bias=True)
        self.fc_policy = nn.Linear(256, self._num_actions)
        self.fc_value = nn.Linear(256, 1)
        self.fc_terminal = nn.Linear(256, 2) #  two classes: is_done, not_done.

    def forward(self, obs, infos, masks, net_state):
        obs, task_ids = self._preprocess(obs, infos['task_id'], self._device)
        #obs embeds:
        x = F.relu(self.conv1(obs))
        x = F.relu(self.conv2(x))
        x = x.view(x.size()[0], -1)
        #task embeds:
        task_ids = self.embed1(task_ids)
        x = torch.cat((x, task_ids), 1)
        #lstm and last layers:
        hx, cx = net_state['hx'] * masks, net_state['cx'] * masks
        hx, cx = self.lstm(x, (hx,cx))

        state_value = self.fc_value(hx)
        act_logits = self.fc_policy(hx)
        act_distr = Categorical(logits=act_logits)
        terminal_logits = self.fc_terminal(hx)

        return state_value, act_distr, terminal_logits, dict(hx=hx,cx=cx)

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

    def terminal_prediction_params(self):
        for name, param in self.named_parameters():
            if 'terminal' in name:
                yield param

    def actor_critic_params(self):
        for name, param, in self.named_parameters():
            if 'terminal' not in name:
                yield param


class TaxiLSTMNetwork(MultiTaskLSTMNetwork):
    def _create_network(self):
        C, H, W = self._obs_shape
        self.conv1 = nn.Conv2d(C, 16, (3, 3), stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, (3, 3), stride=1, padding=1)
        self.embed1 = nn.Embedding(self._num_tasks, self._task_embed_dim)

        C_out,H_out,W_out = calc_output_shape((C, H, W), [self.conv1, self.conv2])

        self.lstm = nn.LSTMCell(C_out*H_out*W_out, 256, bias=True)
        self.fc_policy = nn.Linear(256, self._num_actions)
        self.fc_value = nn.Linear(256, 1)
        self.fc_terminal = nn.Linear(256, 2)

    def forward(self, obs, infos, masks, net_state):
        obs, task_ids = self._preprocess(obs, infos['task_id'], self._device)
        # obs embeds:
        x = F.relu(self.conv1(obs))
        x = F.relu(self.conv2(x))
        x = x.view(x.size()[0], -1)
        # lstm and last layers:
        hx, cx = net_state['hx'] * masks, net_state['cx'] * masks
        hx, cx = self.lstm(x, (hx, cx))

        state_value = self.fc_value(hx)
        action_logits = self.fc_policy(hx)
        action_distr = Categorical(logits=action_logits)
        terminal_logits = self.fc_terminal(hx)

        return state_value, action_distr, terminal_logits, dict(hx=hx, cx=cx)


class MultiTaskLSTMNew(MultiTaskLSTMNetwork):
    def _create_network(self):
        C, H, W = self._obs_shape
        self.conv1 = nn.Conv2d(C, 16, (3, 3), stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, (3, 3), stride=1, padding=1)
        self.embed1 = nn.Embedding(self._num_tasks, self._task_embed_dim)

        conv_and_embed_dim = 32 * H * W + self._task_embed_dim

        self.lstm = nn.LSTMCell(conv_and_embed_dim, 256, bias=True)
        self.fc_policy = nn.Linear(256, self._num_actions)
        self.fc_value = nn.Linear(256, 1)
        #termination classifier part:
        self.fc_terminal1 = nn.Linear(conv_and_embed_dim, 256)
        self.fc_terminal2 = nn.Linear(256 + self._num_actions, 2)

    def forward(self, obs, infos, masks, net_state):
        obs, task_ids = self._preprocess(obs, infos['task_id'], self._device)
        # obs embeds:
        x = F.relu(self.conv1(obs))
        x = F.relu(self.conv2(x))
        x = x.view(x.size()[0], -1)
        # task embeds:
        task_ids = self.embed1(task_ids)
        x = torch.cat((x, task_ids), 1)
        #lstm and actor-critic outputs:
        hx, cx = net_state['hx'] * masks, net_state['cx'] * masks
        hx, cx = self.lstm(x, (hx, cx))

        state_value = self.fc_value(hx)
        act_logits = self.fc_policy(hx)
        act_distr = Categorical(logits=act_logits)
        #termination prediction classifier:
        t = F.relu(self.fc_terminal1(x))
        t = torch.cat((t, act_logits.detach()), 1)
        termination_logits = self.fc_terminal2(t)

        return state_value, act_distr, termination_logits, dict(hx=hx, cx=cx)


taxi_nets = {
    'mt_lstm': MultiTaskLSTMNetwork,
    'mt_lstm_new': MultiTaskLSTMNew,
    'lstm': TaxiLSTMNetwork,
    'mt_ff': MultiTaskFFNetwork,
}