from .paac_nets import torch, nn, F, Variable
from .paac_nets import calc_output_shape, AtariLSTM
import numpy as np

def preprocess_warehouse(obs, infos, t_types):
    obs = (np.ascontiguousarray(obs, dtype=np.float32) / 127.5) - 1.
    obs = Variable(t_types.FloatTensor(obs))

    task_ids = Variable(t_types.LongTensor(infos['task_id']))
    props = Variable(t_types.LongTensor(infos['property']))
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

    def forward(self, states, infos, rnn_inputs):
        x, task_ids, prop_ids = self._preprocess(states, infos, self._intypes)
        nl = self.nonlinearity
        x = nl(self.conv1(x))
        x = nl(self.conv2(x))
        x = nl(self.conv3(x))
        x = x.view(x.size()[0], -1)
        tasks = self.embed_task(task_ids)
        props = self.embed_prop(prop_ids)
        x = torch.cat((x,tasks, props), dim=1)
        hx, cx = self.lstm(x, rnn_inputs)
        task_end_logits = self.fc_task_end(hx)
        return self.fc_value(hx), self.fc_policy(hx), task_end_logits, (hx, cx)


warehouse_nets = dict(
    default=WarehouseDefault,
)



