from .paac_nets import torch, nn, F, Categorical
from .paac_nets import calc_output_shape, AtariLSTM
import numpy as np

def preprocess_warehouse(obs, infos, device):
    obs = (np.ascontiguousarray(obs, dtype=np.float32) / 127.5) - 1.
    obs = torch.tensor(obs, device=device, dtype=torch.float32)
    task_ids = torch.tensor(infos['task_id'], device=device, dtype=torch.long)
    props = torch.tensor(infos['property'], device=device, dtype=torch.long)
    return obs, task_ids, props


class WarehouseDefault(AtariLSTM):
    def __init__(self, num_actions, observation_shape, intput_types,
                 num_tasks=4, num_properties=14, embedding_dim=32,
                 preprocess=preprocess_warehouse, nonlinearity=F.relu):
        self._embedding_dim = embedding_dim
        self._num_tasks = num_tasks
        self._num_properties = num_properties
        self.nonlinearity = nonlinearity

        super(WarehouseDefault, self).__init__(
            num_actions, observation_shape, intput_types, preprocess
        )

    def _create_network(self):
        C, H, W = self._obs_shape
        hidden_dim = 384
        self.conv1 = nn.Conv2d(C, 16, (4, 4), stride=2)
        self.conv2 = nn.Conv2d(16, 32, (4, 4), stride=2)
        self.conv3 = nn.Conv2d(32, 32, (3, 3), stride=2)
        self.embed_task = nn.Embedding(self._num_tasks, self._embedding_dim)
        self.embed_prop = nn.Embedding(self._num_properties, self._embedding_dim)

        convs = [self.conv1, self.conv2, self.conv3]
        C_out, H_out, W_out = calc_output_shape((C, H, W), convs)
        lstm_input = C_out*H_out*W_out + 2*self._embedding_dim

        self.lstm = nn.LSTMCell(lstm_input, hidden_dim, bias=True)
        self.fc_policy = nn.Linear(hidden_dim, self._num_actions)
        self.fc_value = nn.Linear(hidden_dim, 1)
        self.fc_task_end = nn.Linear(hidden_dim, 2)

    def forward(self, states, infos, masks, net_state):
        x, task_ids, prop_ids = self._preprocess(states, infos, self._intypes)
        nl = self.nonlinearity
        x = nl(self.conv1(x))
        x = nl(self.conv2(x))
        x = nl(self.conv3(x))
        x = x.view(x.size()[0], -1)
        tasks = self.embed_task(task_ids)
        props = self.embed_prop(prop_ids)
        x = torch.cat((x,tasks, props), dim=1)

        hx, cx = net_state['hx'] * masks, net_state['cx'] * masks
        hx, cx = self.lstm(x, (hx, cx))

        act_logits = self.fc_policy(hx)
        act_distr = Categorical(logits=act_logits)
        task_end_logits = self.fc_task_end(hx)

        return self.fc_value(hx), act_distr, task_end_logits, dict(hx=hx, cx=cx)


warehouse_nets = dict(
    default=WarehouseDefault,
)



