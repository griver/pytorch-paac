import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as nn_init

from torch.autograd import Variable
import numpy as np


def preprocess_states(s, volatile=False):
    # pytorch conv layers expect inputs of shape (batch, C,H,W)
    states = torch.from_numpy(np.ascontiguousarray(s ,dtype=np.float32)/255.)
    return Variable(states, volatile=volatile)


class FFNetwork(nn.Module):
    def __init__(self, num_actions, input_type, input_channels=4):
        super(FFNetwork, self).__init__()
        self._num_actions = num_actions
        self._input_type = input_type
        self._create_network(input_channels)
        #recursivly traverse layers and inits weights and biases:
        self.apply(init_model_weights)
        assert self.training == True, "Model won't train If self.training is False"

    def _create_network(self, in_channels):
        self.conv1 = nn.Conv2d(in_channels, 16, (8,8), stride=4)
        self.conv2 = nn.Conv2d(16, 32, (4,4), stride=2)
        #if input shape equals (4, 84,84) then the conv part output shape is: (32, 9, 9)
        #self.__flatten_conv_size = 32*9*9
        self.fc3 = nn.Linear(32*9*9, 256)
        self.fc_policy = nn.Linear(256, self._num_actions)
        self.fc_value = nn.Linear(256, 1)

    def forward(self, inputs):
        volatile = not self.training
        inputs = preprocess_states(inputs, volatile).type(self._input_type)
        x = F.relu(self.conv1(inputs))
        x = F.relu(self.conv2(x))
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc3(x))
        # in pytorch A3C an author just outputs logits(the softmax input).
        # action_probs = F.softmax(self.fc_policy(x), dim=1)
        # model outputs logits to be able to compute log_probs via log_softmax later.
        action_logits = self.fc_policy(x)
        state_value = self.fc_value(x)
        return state_value, action_logits


class LSTMNetwork(nn.Module):
    def __init__(self, num_actions, input_type, input_channels=1):
        super(LSTMNetwork, self).__init__()
        self._num_actions = num_actions
        self._input_type = input_type
        self._create_network(input_channels)
        #recursivly traverse layers and inits weights and biases:
        self.apply(init_model_weights)
        assert self.training == True, "Model won't train If self.training is False"

    def _create_network(self, in_channels):
        self.conv1 = nn.Conv2d(in_channels, 16, (8,8), stride=4)
        self.conv2 = nn.Conv2d(16, 32, (4,4), stride=2)
        self.lstm = nn.LSTMCell(32*9*9, 256, bias=True)
        self.fc_policy = nn.Linear(256, self._num_actions)
        self.fc_value = nn.Linear(256, 1)

    def forward(self, inputs):
        volatile = not self.training
        inputs, rnn_inputs = inputs
        inputs = preprocess_states(inputs, volatile).type(self._input_type)
        x = F.relu(self.conv1(inputs))
        x = F.relu(self.conv2(x))
        x = x.view(x.size()[0], -1)
        hx, cx = self.lstm(x, rnn_inputs)
        return self.fc_value(hx), self.fc_policy(hx), (hx,cx)

    def get_initial_state(self, batch_size):
        '''
        Returns initial lstm state as a tuple(hidden_state, cell_state).
        Intial lstm state is supposed to be used at the begging of an episode.
        '''
        volatile = not self.training
        hx = torch.zeros(batch_size, self.lstm.hidden_size).type(self._input_type)
        cx = torch.zeros(batch_size, self.lstm.hidden_size).type(self._input_type)
        return Variable(hx, volatile=volatile), Variable(cx, volatile=volatile)


def init_lstm(module, forget_bias=1.0):
    """
    Initializes all bias values with zeros for all gates
    except forget gates. Initializes a forget gate bias with a given values.
    """
    biases = [module.bias_ih, module.bias_hh]
    for bias in biases:
        nn_init.constant(bias, 0.)

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
    nn_init.uniform(module.weight, -d, d)
    nn_init.uniform(module.bias, -d, d)


def init_linear(module):
    """
    This initialization equals to the default one in pytorch v0.2,
    but who knows what future holds.
    """
    d = 1.0 / np.sqrt(module.in_features)
    nn_init.uniform(module.weight, -d, d)
    nn_init.uniform(module.bias, -d, d)


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
