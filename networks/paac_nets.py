import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as nn_init

from torch.autograd import Variable
import numpy as np
import functools
from collections import namedtuple

def preprocess_images(s_numpy, t_types):
    # pytorch conv layers expect inputs of shape (batch, C,H,W)
    s_numpy = (np.ascontiguousarray(s_numpy, dtype=np.float32)/127.5) - 1. #[0,255] to [-1.,1.]
    return Variable(t_types.FloatTensor(s_numpy))

def old_preprocess_images(s_numpy, t_types):
    # pytorch conv layers expect inputs of shape (batch, C,H,W)
    s_numpy = np.ascontiguousarray(s_numpy, dtype=np.float32)/255. #[0,255] to [0.,1.]
    return Variable(t_types.FloatTensor(s_numpy))


class AtariFF(nn.Module):
    def __init__(self, num_actions, observation_shape, input_types,
                 preprocess=preprocess_images):
        super(AtariFF, self).__init__()
        self._num_actions = num_actions
        self._intypes = input_types
        self._obs_shape = observation_shape
        self._preprocess = preprocess
        self._create_network()
        #recursivly traverse layers and inits weights and biases:
        self.apply(init_model_weights)
        assert self.training == True, "Model won't train If self.training is False"

    def _create_network(self,):
        C,H,W = self._obs_shape
        self.conv1 = nn.Conv2d(C, 16, (8,8), stride=4)
        self.conv2 = nn.Conv2d(16, 32, (4,4), stride=2)

        convs = [self.conv1, self.conv2]
        C_out, H_out, W_out = calc_output_shape((C, H, W), convs)

        self.fc3 = nn.Linear(C_out*H_out*W_out, 256)
        self.fc_policy = nn.Linear(256, self._num_actions)
        self.fc_value = nn.Linear(256, 1)

    def forward(self, states, infos):
        states = self._preprocess(states, self._intypes)
        x = F.relu(self.conv1(states))
        x = F.relu(self.conv2(x))
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc3(x))
        # in pytorch A3C an author just outputs logits(the softmax input).
        # action_probs = F.softmax(self.fc_policy(x), dim=1)
        # model outputs logits to be able to compute log_probs via log_softmax later.
        action_logits = self.fc_policy(x)
        state_value = self.fc_value(x)
        return state_value, action_logits


class AtariLSTM(nn.Module):
    def __init__(self, num_actions, observation_shape, input_types,
                 preprocess=preprocess_images):
        super(AtariLSTM, self).__init__()
        self._num_actions = num_actions
        self._obs_shape = observation_shape
        self._intypes = input_types
        self._preprocess = preprocess
        self._create_network()
        #recursivly traverse layers and inits weights and biases:
        self.apply(init_model_weights)
        assert self.training == True, "Model won't train If self.training is False"

    def _create_network(self):
        C,H,W = self._obs_shape
        self.conv1 = nn.Conv2d(C, 16, (8,8), stride=4)
        self.conv2 = nn.Conv2d(16, 32, (4,4), stride=2)

        convs = [self.conv1, self.conv2]
        C_out, H_out, W_out = calc_output_shape((C, H, W), convs)

        self.lstm = nn.LSTMCell(C_out*H_out*W_out, 256, bias=True)
        self.fc_policy = nn.Linear(256, self._num_actions)
        self.fc_value = nn.Linear(256, 1)

    def forward(self, states, infos, rnn_inputs):
        states = self._preprocess(states, self._intypes)
        x = F.relu(self.conv1(states))
        x = F.relu(self.conv2(x))
        x = x.view(x.size()[0], -1)
        hx, cx = self.lstm(x, rnn_inputs)
        return self.fc_value(hx), self.fc_policy(hx), (hx,cx)

    def get_initial_state(self, batch_size):
        '''
        Returns initial lstm state as a tuple(hidden_state, cell_state).
        Intial lstm state is supposed to be used at the begging of an episode.
        '''
        t_type = self._intypes.FloatTensor
        hx = torch.zeros(batch_size, self.lstm.hidden_size).type(t_type)
        cx = torch.zeros(batch_size, self.lstm.hidden_size).type(t_type)
        return Variable(hx), Variable(cx)

atari_nets = {
    'lstm': AtariLSTM,
    'ff':AtariFF}

class VizdoomLSTM(AtariLSTM):
    def __init__(self, *args, **kwargs):
        self.nonlinearity = kwargs.pop('nonlinearity', F.relu)
        super(VizdoomLSTM, self).__init__(*args, **kwargs)

    def _create_network(self):
        C, H, W = self._obs_shape
        hidden_dim = 256
        self.conv1 = nn.Conv2d(C, 16, (4, 4), stride=2)
        self.conv2 = nn.Conv2d(16, 32, (4, 4), stride=2)
        self.conv3 = nn.Conv2d(32,32, (3,3), stride=2)

        convs = [self.conv1, self.conv2, self.conv3]
        C_out,H_out,W_out = calc_output_shape((C,H,W), convs)

        self.lstm = nn.LSTMCell(C_out*H_out*W_out, hidden_dim, bias=True)
        self.fc_policy = nn.Linear(hidden_dim, self._num_actions)
        self.fc_value = nn.Linear(hidden_dim, 1)

    def forward(self, states, infos, rnn_inputs):
        x = self._preprocess(states, self._intypes)
        nl = self.nonlinearity
        x = nl(self.conv1(x))
        x = nl(self.conv2(x))
        x = nl(self.conv3(x))
        x = x.view(x.size()[0], -1)
        hx, cx = self.lstm(x, rnn_inputs)
        return self.fc_value(hx), self.fc_policy(hx), (hx,cx)

vizdoom_nets = {
    'lstm': VizdoomLSTM,
    'selu_lstm': functools.partial(VizdoomLSTM, nonlinearity=F.selu)
}


def init_lstm(module, forget_bias=1.0):
    """
    Initializes all bias values with zeros for all gates
    except forget gates. Initializes a forget gate bias with a given values.
    """
    biases = [module.bias_ih, module.bias_hh]
    for bias in biases:
        nn_init.constant_(bias, 0.)

    bias_size = module.bias_ih.size()[0] #4*hidden_size
    # bias values goes in order: [ingate, forgetgate, cellgate, outgate]
    # see: https://github.com/pytorch/pytorch/blob/master/torch/nn/_functions/rnn.py#L33
    forget_start, forget_end = bias_size//4, bias_size//2
    for bias in biases:
        bias.data[forget_start:forget_end].fill_(forget_bias/2.)
        # division by two is here because bias_ih and bias_hh are, in fact, one bias divided in two tensors


def init_conv2d(module):
    """
    This initialization equals to the default one in pytorch v0.2,
    but who knows what future holds.
    """
    (h, w), c = module.kernel_size, module.in_channels
    d = 1.0 / np.sqrt(c*h*w)
    nn_init.uniform_(module.weight, -d, d)
    nn_init.uniform_(module.bias, -d, d)


def init_linear(module):
    """
    This initialization equals to the default one in pytorch v0.2,
    but who knows what future holds.
    """
    d = 1.0 / np.sqrt(module.in_features)
    nn_init.uniform_(module.weight, -d, d)
    nn_init.uniform_(module.bias, -d, d)


def init_model_weights(module):
    if isinstance(module, nn.Linear):
        #print('LINEAR_INIT:', module)
        init_linear(module)
    elif isinstance(module, nn.Conv2d):
        #print('CONV2D_INIT:', module)
        init_conv2d(module)
    elif isinstance(module, nn.LSTMCell):
        #print('LSTM_INIT:', module)
        init_lstm(module)


def calc_output_shape(obs_dims, net_layers):
    with torch.no_grad():
        rnd_input = torch.randn(1, *obs_dims)  # batch_size=1
        x = Variable(rnd_input)
        for l in net_layers:
            x = l(x)
        return x.size()[1:]

