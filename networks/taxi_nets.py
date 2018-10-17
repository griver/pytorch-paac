from .paac_nets import torch, nn, F, np, init_model_weights, calc_output_shape
from .paac_nets import Categorical, BaseAgentNetwork
from .factorized_rnn import FactorizedLSTMCell


def preprocess_taxi_input(obs, infos, t_device):

    obs = np.ascontiguousarray(obs, dtype=np.float32)
    obs = torch.tensor(obs, device=t_device, dtype=torch.float32)

    coords = torch.tensor(infos['agent_loc'], device=t_device, dtype=torch.float32)
    tasks_ids = torch.tensor(infos['task_id'].tolist(), device=t_device, dtype=torch.long)
    task_status = torch.tensor(infos['task_status']).to(t_device)
    task_mask = (task_status == 0.).to(torch.float32)
    return obs, tasks_ids, coords, task_mask.unsqueeze(-1)


class MultiTaskLSTMNetwork(nn.Module, BaseAgentNetwork):
    def __init__(self, num_actions, observation_shape, device,
                 num_tasks=12, task_embed_dim=128, preprocess=None,
                 use_location=False, hidden_size=256):
        super(MultiTaskLSTMNetwork, self).__init__()
        self._num_actions = num_actions
        self._device = device
        self._obs_shape = observation_shape
        self._task_embed_dim = task_embed_dim
        self._num_tasks = num_tasks
        self._preprocess = preprocess if preprocess is not None else lambda *args: args
        self.use_location = use_location
        self._hidden_size = hidden_size
        self._create_network()
        self.apply(init_model_weights)
        assert self.training == True, "Model won't train If self.training is False"

    def _create_network(self):
        C,H,W = self._obs_shape
        pad = 1 if H <= 5 else 0

        self.conv1 = nn.Conv2d(C, 16, (3, 3), stride=1, padding=pad)
        self.conv2 = nn.Conv2d(16, 32, (3, 3), stride=1, padding=pad)
        self.embed1 = nn.Embedding(self._num_tasks, self._task_embed_dim)

        C_out,H_out,W_out = calc_output_shape((C,H,W),[self.conv1, self.conv2])
        flatten_size = C_out * H_out * W_out + self._task_embed_dim
        if self.use_location:
            flatten_size += 2
        self.lstm = nn.LSTMCell(flatten_size, self._hidden_size, bias=True)
        self.fc_policy = nn.Linear(self._hidden_size, self._num_actions)
        self.fc_value = nn.Linear(self._hidden_size, 1)
        self.fc_terminal = nn.Linear(self._hidden_size, 2) #  two classes: is_done, not_done.

    def forward(self, obs, infos, masks, net_state):
        task_ids, coords = infos['task_id'], infos['agent_loc']
        obs, task_ids, coords, _ = self._preprocess(obs, infos, self._device)
        #obs embeds:
        x = F.relu(self.conv1(obs))
        x = F.relu(self.conv2(x))
        x = x.view(x.size()[0], -1)
        #task embeds:
        tasks = self.embed1(task_ids)
        if self.use_location:
            x = torch.cat((x, tasks, coords), 1)
        else:
            x = torch.cat((x, tasks), 1)

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

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TaskEnvRNN(MultiTaskLSTMNetwork):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('hidden_size', 208)
        #hidden_size=208 gives approximately the same number of parameters as MultiTaskLSTMNetwork(hidden_size=256)
        #self.erase_task_memory = kwargs.pop('erase_task_memory', True)
        super(TaskEnvRNN, self).__init__(*args, **kwargs)

    def _create_network(self):
        C, H, W = self._obs_shape
        pad = 1 if H <= 5 else 0

        self.conv1 = nn.Conv2d(C, 16, (3, 3), stride=1, padding=pad)
        self.conv2 = nn.Conv2d(16, 32, (3, 3), stride=1, padding=pad)

        C_out, H_out, W_out = calc_output_shape((C, H, W), [self.conv1, self.conv2])
        obs_flatten = C_out * H_out * W_out + (2 if self.use_location else 0)

        self.env_lstm = nn.LSTMCell(obs_flatten, self._hidden_size, bias=True)
        self.task_lstm = FactorizedLSTMCell(
            obs_flatten+self._hidden_size, self._hidden_size,
            self._num_tasks,
            self._task_embed_dim
        )
        self.fc_policy = nn.Linear(2*self._hidden_size, self._num_actions)
        self.fc_value = nn.Linear(2*self._hidden_size, 1)
        self.fc_terminal = nn.Linear(2*self._hidden_size, 2)  #  two classes: 0-not_done, 1-is_done

    def forward(self, obs, infos, masks, net_state):
        obs, task_ids, coords, task_masks = self._preprocess(obs, infos, self._device)
        x = F.relu(self.conv1(obs))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        if self.use_location:
            x = torch.cat([x, coords], dim=1)

        t_masks = masks*task_masks # if self.erase_task_memory else masks
        env_h, env_c = net_state['env_h']*masks, net_state['env_c']*masks
        task_h, task_c = net_state['task_h']*t_masks, net_state['task_c']*t_masks

        env_h, env_c = self.env_lstm(x, (env_h, env_c))
        x_env_h = torch.cat([x, env_h],dim=1)

        task_h, task_c = self.task_lstm(x_env_h, (task_h,task_c), task_ids)
        total_h = torch.cat([task_h, env_h], dim=1)

        net_state = {'env_h':env_h, 'env_c':env_c,
                     'task_h':task_h, 'task_c':task_c}
        state_value = self.fc_value(total_h)
        act_logits = self.fc_policy(total_h)
        act_distr = Categorical(logits=act_logits)
        done_logits = self.fc_terminal(total_h)
        return state_value, act_distr, done_logits, net_state

    def init_rnn_state(self, batch_size=None):
        def get_shape(h):
            return (batch_size, h) if batch_size else (h,)

        env_shape = get_shape(self.env_lstm.hidden_size)
        task_shape = get_shape(self.task_lstm.hidden_size)
        t,d = torch.float32, self._device

        self.task_lstm.generate_and_register_weights()
        return dict(
            env_h=torch.zeros(*env_shape, dtype=t, device=d),
            env_c=torch.zeros(*env_shape, dtype=t, device=d),
            task_h=torch.zeros(*task_shape, dtype=t, device=d),
            task_c=torch.zeros(*task_shape, dtype=t, device=d)
        )

    def detach_rnn_state(self, rnn_state):
        for k in list(rnn_state.keys()): #fixate keys before modifying the dict
            rnn_state[k] = rnn_state[k].detach()
        ##should call this after each weight change(i.e. either changed by optimizer or loaded from a file):
        self.task_lstm.generate_and_register_weights()


class TaskEnvHeavyEmbed(TaskEnvRNN):

    def _create_network(self):
        C, H, W = self._obs_shape
        pad = 1 if H <= 5 else 0

        self.conv1 = nn.Conv2d(C, 16, (3, 3), stride=1, padding=pad)
        self.conv2 = nn.Conv2d(16, 32, (3, 3), stride=1, padding=pad)

        C_out, H_out, W_out = calc_output_shape((C, H, W), [self.conv1, self.conv2])
        obs_flatten = C_out * H_out * W_out + (2 if self.use_location else 0)


        self.env_lstm = nn.LSTMCell(obs_flatten, self._hidden_size, bias=True)
        self.task_lstm = FactorizedLSTMCell(
            obs_flatten+self._hidden_size, self._hidden_size,
            self._num_tasks,
            self._task_embed_dim
        )

        self.fc_embed = nn.Embedding(self._num_tasks, self._hidden_size)
        fc_input = self._hidden_size + self._hidden_size  # fc_embed + task_rnn

        self.fc_policy = nn.Linear(fc_input, self._num_actions)
        self.fc_value = nn.Linear(fc_input, 1)
        self.fc_terminal = nn.Linear(fc_input, 2)  #  two classes: 0-not_done, 1-is_done

    def forward(self, obs, infos, masks, net_state):
        obs, task_ids, coords, task_masks = self._preprocess(obs, infos, self._device)
        x = F.relu(self.conv1(obs))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        if self.use_location:
            x = torch.cat([x, coords], dim=1)

        t_masks = masks * task_masks  # if self.erase_task_memory else masks
        env_h, env_c = net_state['env_h'] * masks, net_state['env_c'] * masks
        task_h, task_c = net_state['task_h'] * t_masks, net_state['task_c'] * t_masks

        env_h, env_c = self.env_lstm(x, (env_h, env_c))
        x_env_h = torch.cat([x, env_h], dim=1)

        task_h, task_c = self.task_lstm(x_env_h, (task_h, task_c), task_ids)

        net_state = {'env_h':env_h, 'env_c':env_c,
                     'task_h':task_h, 'task_c':task_c}

        tasks = self.fc_embed(task_ids)
        fc_input = torch.cat([task_h, tasks], dim=1)

        state_value = self.fc_value(fc_input)

        act_logits = self.fc_policy(fc_input)
        act_distr = Categorical(logits=act_logits)

        done_logits = self.fc_terminal(fc_input)

        return state_value, act_distr, done_logits, net_state


class TaskEnvSimple(MultiTaskLSTMNetwork):

    def _create_network(self):
        C, H, W = self._obs_shape
        pad = 1 if H <= 5 else 0

        self.conv1 = nn.Conv2d(C, 16, (3, 3), stride=1, padding=pad)
        self.conv2 = nn.Conv2d(16, 32, (3, 3), stride=1, padding=pad)
        self.embed1 = nn.Embedding(self._num_tasks, self._task_embed_dim)

        C_out, H_out, W_out = calc_output_shape((C, H, W), [self.conv1, self.conv2])
        obs_flatten = C_out * H_out * W_out + (2 if self.use_location else 0)

        self.env_lstm = nn.LSTMCell(obs_flatten, self._hidden_size, bias=True)
        self.task_lstm = nn.LSTMCell(self._task_embed_dim+self._hidden_size, self._hidden_size, bias=True)

        self.fc_policy = nn.Linear(self._hidden_size, self._num_actions)
        self.fc_value = nn.Linear(self._hidden_size, 1)
        self.fc_terminal = nn.Linear(self._hidden_size, 2)  #  two classes: 0-not_done, 1-is_done

    def forward(self, obs, infos, masks, net_state):
        obs, task_ids, coords, task_masks = self._preprocess(obs, infos, self._device)
        x = F.relu(self.conv1(obs))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        tasks = self.embed1(task_ids)

        if self.use_location:
            x = torch.cat([x, coords], dim=1)

        t_masks = masks*task_masks # if self.erase_task_memory else masks
        env_h, env_c = net_state['env_h']*masks, net_state['env_c']*masks
        task_h, task_c = net_state['task_h']*t_masks, net_state['task_c']*t_masks

        env_h, env_c = self.env_lstm(x, (env_h, env_c))
        x = torch.cat([tasks, env_h],dim=1)

        task_h, task_c = self.task_lstm(x, (task_h,task_c))
        total_h = task_h #torch.cat([task_h, env_h], dim=1)

        net_state = {'env_h':env_h, 'env_c':env_c,
                     'task_h':task_h, 'task_c':task_c}
        state_value = self.fc_value(total_h)
        act_logits = self.fc_policy(total_h)
        act_distr = Categorical(logits=act_logits)
        done_logits = self.fc_terminal(total_h)
        return state_value, act_distr, done_logits, net_state

    def init_rnn_state(self, batch_size=None):
        def get_shape(h):
            return (batch_size, h) if batch_size else (h,)

        env_shape = get_shape(self.env_lstm.hidden_size)
        task_shape = get_shape(self.task_lstm.hidden_size)
        t,d = torch.float32, self._device

        self.task_lstm.generate_and_register_weights()
        return dict(
            env_h=torch.zeros(*env_shape, dtype=t, device=d),
            env_c=torch.zeros(*env_shape, dtype=t, device=d),
            task_h=torch.zeros(*task_shape, dtype=t, device=d),
            task_c=torch.zeros(*task_shape, dtype=t, device=d)
        )

    def detach_rnn_state(self, rnn_state):
        for k in list(rnn_state.keys()): #fixate keys before modifying the dict
            rnn_state[k] = rnn_state[k].detach()
        ##should call this after each weight change(i.e. either changed by optimizer or loaded from a file):
        self.task_lstm.generate_and_register_weights()


class TaxiLSTMNetwork(MultiTaskLSTMNetwork):
    """
    RNN network for the full Taxi task
    """
    def _create_network(self):
        C, H, W = self._obs_shape
        pad = 1 if H <= 5 else 0

        self.conv1 = nn.Conv2d(C, 16, (3, 3), stride=1, padding=pad)
        self.conv2 = nn.Conv2d(16, 32, (3, 3), stride=1, padding=pad)
        self.embed1 = nn.Embedding(self._num_tasks, self._task_embed_dim)

        C_out,H_out,W_out = calc_output_shape((C, H, W), [self.conv1, self.conv2])
        obs_flatten = C_out*H_out*W_out + (2 if self.use_location else 0)
        self.lstm = nn.LSTMCell(obs_flatten, self._hidden_size, bias=True)
        self.fc_policy = nn.Linear(self._hidden_size, self._num_actions)
        self.fc_value = nn.Linear(self._hidden_size, 1)
        self.fc_terminal = nn.Linear(self._hidden_size, 2)

    def forward(self, obs, infos, masks, net_state):
        obs, _, coords, _ = self._preprocess(obs, infos, self._device)
        # obs embeds:
        x = F.relu(self.conv1(obs))
        x = F.relu(self.conv2(x))
        x = x.view(x.size()[0], -1)
        if self.use_location:
            x = torch.cat([x, coords], dim=1)
        # lstm and last layers:
        hx, cx = net_state['hx'] * masks, net_state['cx'] * masks
        hx, cx = self.lstm(x, (hx, cx))

        state_value = self.fc_value(hx)
        action_logits = self.fc_policy(hx)
        action_distr = Categorical(logits=action_logits)
        terminal_logits = self.fc_terminal(hx)

        return state_value, action_distr, terminal_logits, dict(hx=hx, cx=cx)


taxi_nets = {
    'mt_lstm': MultiTaskLSTMNetwork,
    'mt_task_env':TaskEnvRNN,
    'task_env_head_embed':TaskEnvHeavyEmbed,
    'task_env_simple':TaskEnvSimple,
    'lstm': TaxiLSTMNetwork,
}
