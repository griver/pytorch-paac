from .paac_nets import torch, nn, F, Variable, np, init_model_weights

def preprocess_taxi_input(obs, tasks_ids, Ttypes, volatile=False):
    obs = torch.from_numpy(np.ascontiguousarray(obs, dtype=np.float32))
    obs = Variable(obs, volatile=volatile).type(Ttypes.FloatTensor)
    tasks_ids = Variable(Ttypes.LongTensor(tasks_ids.tolist()), volatile=volatile)
    return obs, tasks_ids


class TaskTerminationPredictor(nn.Module):
    #TODO: Create base class or a mixin for both following networks
    pass


class MultiTaskFFNetwork(nn.Module):

    def __init__(self, num_actions, observation_shape, input_types,
                 num_tasks=5, task_embed_dim=32, preprocess=None):
        super(MultiTaskFFNetwork, self).__init__()
        self._num_actions = num_actions
        self._intypes = input_types
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
        #fc3 receives flattened conv network output + current task embedding
        self.fc3 = nn.Linear(32 * H * W + self._task_embed_dim, 256)
        self.fc_policy = nn.Linear(256, self._num_actions)
        self.fc_value = nn.Linear(256, 1)
        self.fc_terminal = nn.Linear(256, 2) # two classes: is_done, not is_done.

    def forward(self, obs, task_ids):
        volatile = not self.training
        obs, task_ids = self._preprocess(obs, task_ids, self._intypes, volatile)
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
        state_value = self.fc_value(x)
        terminal_logits = self.fc_terminal(x)

        return state_value, action_logits, terminal_logits

    def terminal_prediction_params(self):
        for name, param in self.named_parameters():
            if 'terminal' in name:
                yield param


class MultiTaskLSTMNetwork(nn.Module):
    def __init__(self, num_actions, observation_shape, input_types,
                 num_tasks=5, task_embed_dim=32, preprocess=None):
        super(MultiTaskLSTMNetwork, self).__init__()
        self._num_actions = num_actions
        self._intypes = input_types
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

        self.lstm = nn.LSTMCell(32*H*W + self._task_embed_dim, 256, bias=True)
        self.fc_policy = nn.Linear(256, self._num_actions)
        self.fc_value = nn.Linear(256, 1)
        self.fc_terminal = nn.Linear(256, 2) #  two classes: is_done, not_done.

    def forward(self, obs, task_ids, **kwargs):
        volatile = not self.training
        rnn_inputs = kwargs['rnn_inputs']
        obs, task_ids = self._preprocess(obs, task_ids, self._intypes, volatile)
        #obs embeds:
        x = F.relu(self.conv1(obs))
        x = F.relu(self.conv2(x))
        x = x.view(x.size()[0], -1)
        #task embeds:
        task_ids = self.embed1(task_ids)
        x = torch.cat((x, task_ids), 1)
        #lstm and last layers:
        hx, cx = self.lstm(x, rnn_inputs)
        state_value = self.fc_value(hx)
        action_logits = self.fc_policy(hx)
        terminal_logits = self.fc_terminal(hx)
        return state_value, action_logits, terminal_logits, (hx,cx)

    def get_initial_state(self, batch_size):
        volatile = not self.training
        hx = torch.zeros(batch_size, self.lstm.hidden_size).type(self._intypes.FloatTensor)
        cx = torch.zeros(batch_size, self.lstm.hidden_size).type(self._intypes.FloatTensor)
        return Variable(hx, volatile=volatile), Variable(cx, volatile=volatile)

    def terminal_prediction_params(self):
        for name, param in self.named_parameters():
            if 'terminal' in name:
                yield param