from torch.nn.modules import rnn
from torch import nn
import torch as th
import math

def diag_lstm_size(input_size, hidden_size, fact_size, bias=True):
    W_ih = 4 * hidden_size * fact_size + fact_size * input_size + fact_size
    W_hh = 4 * hidden_size * fact_size + fact_size * hidden_size + fact_size
    total_size = W_hh + W_ih
    if bias:
        total_size += 4 * hidden_size * 2
    return total_size

def lstm_size(input_size, hidden_size, bias=True):
    W_ih = 4*hidden_size*input_size
    W_hh = 4*hidden_size*hidden_size
    total_size = W_hh + W_ih
    if bias:
        total_size += 4*hidden_size*2
    return total_size


class DiagonalLSTMCell(rnn.RNNCellBase):
    def __init__(self, input_size, hidden_size, num_task, task_embed_size=100, bias=True):
        super(DiagonalLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih1 = nn.Parameter(th.Tensor(4*hidden_size, task_embed_size))
        self.weight_ih2 = nn.Parameter(th.Tensor(task_embed_size, input_size))

        self.weight_hh1 = nn.Parameter(th.Tensor(4 * hidden_size, task_embed_size))
        self.weight_hh2 = nn.Parameter(th.Tensor(task_embed_size, hidden_size))

        self.embedding = nn.Embedding(num_task, task_embed_size)
        if bias:
            self.bias_ih = nn.Parameter(th.Tensor(4 * hidden_size))
            self.bias_hh = nn.Parameter(th.Tensor(4 * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)

        self.reset_parameters()


    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, hx, task_ids):
        self.check_forward_input(input)
        self.check_forward_hidden(input, hx[0], '[0]')
        self.check_forward_hidden(input, hx[1], '[1]')
        cell = self._backend.LSTMCell
        hx,cx = hx
        bias_ih = self.bias_ih
        bias_hh = self.bias_hh
        batch_size = input.size(0)
        new_hx, new_cx = [None]*batch_size, [None]*batch_size

        for t in range(batch_size):
            W_ih, W_hh = self.restore_lstm_weights(task_ids[t])
            new_hx[t], new_cx[t] = cell(
                input[t:t+1],
                (hx[t:t+1],cx[t:t+1]),
                W_ih, W_hh,
                bias_ih, bias_hh
            )
        return th.cat(new_hx), th.cat(new_cx)

    def restore_lstm_weights(self, task_ids):
        task_embed = self.embedding(task_ids)
        task_diag = th.diag(task_embed)
        W_ih = th.mm(th.mm(self.weight_ih1,task_diag),self.weight_ih2)
        W_hh = th.mm(th.mm(self.weight_hh1,task_diag),self.weight_hh2)
        return W_ih, W_hh